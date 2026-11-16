"""
documind/ingest.py — steps 1-4 of the RAG pipeline (the "ingestion" half).

  PDF -> PyPDFLoader -> text chunks -> Google embeddings -> ChromaDB

Runs once per new PDF; load_existing_vectorstore() reuses the saved result.
"""

# WHAT THIS FILE IS: turns an uploaded PDF into a searchable index on disk, or reuses a saved one.
# Example: MySQL Handbook (72 pages) -> 71 chunks -> 71 embeddings in chroma_store/<12-char hash>/
#   + index_meta.json. Same file again -> reused, 0 API calls.
# Flow: load_existing_vectorstore() first (None if missing/stale/incomplete) -> else process_pdf():
#   load -> chunk -> embed -> store

import os  # folders + paths (make, list, join)
import json  # dict <-> text, for index_meta.json
import hashlib  # hash = fixed code from content; same file -> same code

# LangChain = library of ready blocks for LLM apps (loaders, splitters, stores)
from langchain_community.document_loaders import PyPDFLoader  # PDF -> one Document per page
from langchain_text_splitters import RecursiveCharacterTextSplitter  # cuts text at paragraphs, lines, sentences
from langchain_chroma import Chroma  # LangChain wrapper for ChromaDB (vector store)
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Google API: text -> 3072 numbers

from documind.settings import (  # every number/name from one place
    CHROMA_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS,  # store folder + chunking
    EMBED_MODEL, INDEX_META_FILE,  # embedding model + meta file name
)


# ── BLOCK: helpers ──────────────────────────────────────────────────────
# IN: PDF path -> OUT: 12-char hash naming its folder; same file (even renamed) -> same folder, reused
# e.g. get_pdf_hash("C:/tmp/handbook.pdf") -> 12 hex chars (0-9, a-f)
def get_pdf_hash(pdf_path: str) -> str:
    """
    Creates a unique fingerprint for a PDF so each document gets its own
    ChromaDB folder. Prevents different PDFs from overwriting each other.
    """
    with open(pdf_path, "rb") as f:  # "rb" = read raw bytes
        return hashlib.sha256(f.read()).hexdigest()[:12]  # first 12 of 64 hex chars, ~281 trillion values

# IN: nothing -> OUT: the 3 settings an index was built with (change any -> rebuild)
# e.g. {"embed_model": "models/gemini-embedding-001", "chunk_size": 1000, "chunk_overlap": 200}
def _index_settings() -> dict:
    """Settings an existing vector store must match to be reused."""
    return {"embed_model": EMBED_MODEL, "chunk_size": CHUNK_SIZE,  # model + chunk size
            "chunk_overlap": CHUNK_OVERLAP}  # + overlap


# ── BLOCK: build a new index (steps 1-4) ────────────────────────────────
# IN: PDF path + Google API key -> OUT: ready ChromaDB store; ValueError if no text (scanned PDF)
# e.g. MySQL Handbook (72 pages) -> 71 chunks -> chroma_store/<hash>/ + index_meta.json
def process_pdf(pdf_path: str, google_api_key: str) -> Chroma:
    """
    Runs the full RAG ingestion pipeline for a new PDF.

    Pipeline:
        PDF → PyPDFLoader → Text Chunks → Google Embeddings → ChromaDB

    Args:
        pdf_path:       Local path to the PDF file.
        google_api_key: Google AI Studio key, used for embeddings.

    Returns:
        A ChromaDB vector store object ready for retrieval.
    """

    # ── Step 1: Load ────────────────────────────────────────────────────────
    print(f"[RAG Engine] Loading PDF: {pdf_path}")  # goes to terminal log, not the page
    loader = PyPDFLoader(pdf_path)  # reader for this PDF
    pages = loader.load()  # one Document per page (text + {"page": n})
    print(f"[RAG Engine] Loaded {len(pages)} pages.")  # e.g. 72 pages

    # ── Step 2: Chunk ───────────────────────────────────────────────────────
    # Small overlapping pieces, so each question sends only a few, not the whole PDF.
    splitter = RecursiveCharacterTextSplitter(  # the text cutter
        chunk_size=CHUNK_SIZE,  # max 1000 chars
        chunk_overlap=CHUNK_OVERLAP,  # 200 chars repeated
        separators=CHUNK_SEPARATORS,  # paragraph, line, sentence, word, anywhere
    )
    chunks = splitter.split_documents(pages)  # per page, keeps page number; never joins pages
    print(f"[RAG Engine] Split into {len(chunks)} chunks.")  # e.g. 71 chunks
    if not chunks:  # no text on any page = scanned image PDF
        # Fail clearly instead of an empty index that answers "could not find" to everything.
        raise ValueError("No text could be extracted from this PDF. It may be "
                         "a scanned document; OCR is not supported yet.")

    # ── Steps 3 & 4: Embed & Store ──────────────────────────────────────────
    embedding_model = GoogleGenerativeAIEmbeddings(  # client only, nothing sent yet
        model=EMBED_MODEL,  # "models/gemini-embedding-001"
        google_api_key=google_api_key  # user's Google key
    )

    pdf_hash = get_pdf_hash(pdf_path)  # e.g. 12 hex chars
    persist_dir = str(CHROMA_BASE_DIR / pdf_hash)  # "chroma_store/<hash>"
    os.makedirs(persist_dir, exist_ok=True)  # create if missing, no error if exists

    # Gotcha: a rejected old store (stale/incomplete) may still be here; from_documents would ADD to it.
    # Delete via Chroma, since deleting the folder breaks on Windows file locks.
    if os.listdir(persist_dir):  # folder not empty = old store
        Chroma(persist_directory=persist_dir,  # open the old store
               embedding_function=embedding_model).delete_collection()  # wipe its vectors

    print(f"[RAG Engine] Embedding chunks and saving to: {persist_dir}")  # log
    # Slow, quota-using step: Google free tier has per-minute + per-day limits.
    vector_store = Chroma.from_documents(  # embed every chunk, then save
        documents=chunks,  # the 71 chunks
        embedding=embedding_model,  # Google embedder
        persist_directory=persist_dir,  # persist = keep on disk after app closes
    )
    # Write meta only AFTER a full build, so a half-built store has no matching record.
    with open(os.path.join(persist_dir, INDEX_META_FILE), "w", encoding="utf-8") as f:  # index_meta.json
        json.dump({**_index_settings(), "n_chunks": len(chunks)}, f)  # settings + chunk count
    print("[RAG Engine] Vector store created successfully.")  # log
    return vector_store  # ready for retrieval


# ── BLOCK: reuse a saved index ──────────────────────────────────────────
# IN: PDF path + Google API key -> OUT: saved store, or None = "build it again"
# e.g. same handbook twice -> 2nd time saved store; CHUNK_SIZE changed to 500 -> None
def load_existing_vectorstore(pdf_path: str, google_api_key: str):
    """
    Loads an already-processed vector store from disk (avoids re-processing).
    If we already processed this exact PDF before, we skip re-embedding it
    to save time and API calls.

    A store is reused only if it was built with the current embedding model
    and chunk settings, and holds one vector per chunk. Stores created before
    index_meta.json existed have no record and are reused as before.
    """
    # ── Step 1: find this PDF's folder ──
    pdf_hash = get_pdf_hash(pdf_path)  # same file -> same hash as process_pdf
    persist_dir = str(CHROMA_BASE_DIR / pdf_hash)  # "chroma_store/<hash>"

    if not (os.path.exists(persist_dir) and os.listdir(persist_dir)):  # missing or empty?
        return None  # never processed

    # ── Step 2: check settings in index_meta.json ──
    meta = None  # stays None for old stores without meta (trusted)
    meta_path = os.path.join(persist_dir, INDEX_META_FILE)  # ".../index_meta.json"
    if os.path.exists(meta_path):  # record exists?
        with open(meta_path, encoding="utf-8") as f:  # open it
            meta = json.load(f)  # text -> dict
        if any(meta.get(k) != v for k, v in _index_settings().items()):  # e.g. saved 500 vs now 1000
            print("[RAG Engine] Existing store was built with different settings; rebuilding.")  # log
            return None  # stale -> rebuild

    # ── Step 3: open the saved store ──
    print("[RAG Engine] Found existing vector store. Loading from disk...")  # log
    embedding_model = GoogleGenerativeAIEmbeddings(  # still needed: questions get embedded the same way
        model=EMBED_MODEL,  # same model as the chunks
        google_api_key=google_api_key  # user's Google key
    )
    store = Chroma(  # open, no new embedding calls
        persist_directory=persist_dir,  # the saved folder
        embedding_function=embedding_model,  # for embedding questions
    )
    # ── Step 4: check it is complete ──
    # eval/README "Defect": a failed run left 200 of 275 vectors.
    if meta is not None and store._collection.count() != meta.get("n_chunks"):  # saved vectors != chunks?
        print("[RAG Engine] Existing store is incomplete; rebuilding.")  # log
        return None  # half-built -> rebuild
    return store  # all checks passed: reuse
