"""
documind/ingest.py — steps 1-4 of the RAG pipeline (the "ingestion" half).

  PDF -> PyPDFLoader -> text chunks -> Google embeddings -> ChromaDB

Runs once per new PDF; load_existing_vectorstore() reuses the saved result.
"""

# WHAT THIS FILE IS: the librarian's filing work. It turns an uploaded PDF into a searchable index on disk,
# and decides whether an index saved earlier can be reused.
# Real example: the MySQL Handbook (72 pages) -> 71 chunks at chunk_size 1000 -> 71 embeddings saved in
#   chroma_store/<12-character hash>/ plus index_meta.json. Uploading the same file again -> reused, 0 API calls.
# Functions, in the order app.py uses them:
#   load_existing_vectorstore() = try the saved index first (returns None if missing, stale or incomplete)
#   process_pdf()               = otherwise build it: load -> chunk -> embed -> store (steps 1-4)
#   get_pdf_hash(), _index_settings() = small helpers both of them use
#

# os = work with folders and files (make the chroma_store/<hash> folder, list it, join paths).
import os
# json = turn a Python dict into text and back. Used to write and read index_meta.json.
import json
# hashlib = makes fingerprints (hashes) of data. SHA-256 of the PDF bytes names the PDF's storage folder.
# A hash is a short code computed from the content: the same file always gives the same code,
# a different file (even one changed byte) gives a completely different code.
import hashlib

# LangChain = a Python library that gives ready-made building blocks for LLM apps (loaders, splitters,
# vector stores, prompts, chains). Each import below is one block.
# PyPDFLoader = reads a PDF with the pypdf library and returns one LangChain "Document" per page.
# A Document = page_content (the text) + metadata (extra labels, e.g. {"page": 3, "source": "file.pdf"}).
from langchain_community.document_loaders import PyPDFLoader
# RecursiveCharacterTextSplitter = cuts long text into chunks of a chosen size, preferring to cut at
# paragraph breaks first, then lines, then sentences, then spaces.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chroma = LangChain's wrapper around ChromaDB, the vector store used in this project.
from langchain_chroma import Chroma
# GoogleGenerativeAIEmbeddings = calls Google's embedding API (gemini-embedding-001) to turn text into vectors.
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# All the numbers and names come from settings.py (one place for every setting).
from documind.settings import (
    CHROMA_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS,
    EMBED_MODEL, INDEX_META_FILE,
)


# IN: path to a PDF file on disk  ->  OUT: a 12-character fingerprint (hash) of the file's bytes.
# WHY: the hash names the PDF's folder in chroma_store/. The same file uploaded again (even renamed)
# gives the same hash, so the saved embeddings are reused and no API calls are spent.
# Example: get_pdf_hash("C:/tmp/handbook.pdf") -> 12 hexadecimal characters (0-9, a-f), used as the
# folder name chroma_store/<those 12 characters>/.
def get_pdf_hash(pdf_path: str) -> str:
    """
    Creates a unique fingerprint for a PDF so each document gets its own
    ChromaDB folder. Prevents different PDFs from overwriting each other.
    """
    # Open in "rb" = read binary (raw bytes, not text). sha256() hashes the bytes, hexdigest() writes the
    # 64-character result in hexadecimal, and [:12] keeps the first 12 characters (short folder name; still
    # 16^12 = about 281 trillion possible values, so two different PDFs practically never collide).
    with open(pdf_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]

# IN: nothing  ->  OUT: the 3 settings that decide how an index (vector store) is built.
# WHY: if any of these changes, an old saved index no longer matches and must be rebuilt.
# Example: {"embed_model": "models/gemini-embedding-001", "chunk_size": 1000, "chunk_overlap": 200}
def _index_settings() -> dict:
    """Settings an existing vector store must match to be reused."""
    return {"embed_model": EMBED_MODEL, "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP}


# IN: path of an uploaded PDF + Google API key  ->  OUT: a ready ChromaDB vector store for that PDF.
# WHY: this is the "ingestion" half of RAG (steps 1-4). It runs once per new PDF; afterwards
# load_existing_vectorstore() reuses the saved result.
# Example: the MySQL Handbook (72 pages) -> 71 chunks at chunk_size 1000 (numbers from the README)
#   -> 71 embeddings saved in chroma_store/<hash>/ plus index_meta.json.
# Raises ValueError if the PDF has no extractable text (a scanned, image-only PDF).
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
    # print() lines go to the terminal/server log, not to the web page. Useful when debugging.
    print(f"[RAG Engine] Loading PDF: {pdf_path}")
    # Step 1: PyPDFLoader reads the PDF; load() returns a list with one Document per page.
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[RAG Engine] Loaded {len(pages)} pages.")

    # ── Step 2: Chunk ───────────────────────────────────────────────────────
    # We split the full document into small overlapping paragraphs.
    # Without chunking, we'd have to send the entire PDF to the AI on every
    # question, which is slow and expensive.
    # chunk_size / chunk_overlap come from settings.py (1000 / 200 characters).
    # separators = CHUNK_SEPARATORS from settings.py (paragraph, line, sentence, word, anywhere).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    # split_documents() splits each page separately and keeps each page's metadata (so every chunk
    # still knows its page number). It never joins two pages into one chunk.
    chunks = splitter.split_documents(pages)
    print(f"[RAG Engine] Split into {len(chunks)} chunks.")
    # No chunks at all = no text was found in any page (typical for a scanned PDF made of images).
    if not chunks:
        # Image-only (scanned) PDFs yield no text; say so instead of building
        # an empty index that answers every question with "could not find".
        raise ValueError("No text could be extracted from this PDF. It may be "
                         "a scanned document; OCR is not supported yet.")

    # ── Steps 3 & 4: Embed & Store ──────────────────────────────────────────
    # Create the embedding client. Nothing is sent to Google yet; that happens in from_documents() below.
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=google_api_key
    )

    # Where this PDF's index lives: chroma_store/<12-character hash>.
    # makedirs(..., exist_ok=True) = create the folder (and chroma_store/ itself) if missing; no error if it exists.
    pdf_hash = get_pdf_hash(pdf_path)
    persist_dir = str(CHROMA_BASE_DIR / pdf_hash)
    os.makedirs(persist_dir, exist_ok=True)

    # A store rejected by load_existing_vectorstore (stale settings, or
    # incomplete) still sits in this folder. from_documents would ADD to it,
    # mixing old and new vectors, so empty it first. Deleting the collection
    # through Chroma avoids Windows file locks that break deleting the folder.
    # os.listdir() lists the folder. A non-empty list = something is already there (an old or broken store).
    # delete_collection() wipes the old vectors through ChromaDB itself.
    if os.listdir(persist_dir):
        Chroma(persist_directory=persist_dir,
               embedding_function=embedding_model).delete_collection()

    print(f"[RAG Engine] Embedding chunks and saving to: {persist_dir}")
    # Steps 3 + 4: send every chunk to Google to get its embedding, then save chunk text + embedding +
    # metadata into ChromaDB in persist_directory (persist = keep it on disk after the app closes).
    # This is the slow, quota-using step (Google free tier has a per-day and per-minute limit).
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    # Only after the store is fully built, write index_meta.json with the settings + chunk count.
    # {**_index_settings(), "n_chunks": ...} = copy all keys of the settings dict and add one more key.
    with open(os.path.join(persist_dir, INDEX_META_FILE), "w", encoding="utf-8") as f:
        json.dump({**_index_settings(), "n_chunks": len(chunks)}, f)
    print("[RAG Engine] Vector store created successfully.")
    return vector_store


# IN: path of an uploaded PDF + Google API key  ->  OUT: the saved ChromaDB store, or None.
# WHY: re-embedding a PDF already seen wastes time and daily quota. Return None means "build it again".
# Example: you upload the same handbook twice -> 2nd time this returns the saved store and app.py skips
# process_pdf(). If CHUNK_SIZE was changed to 500 since, the settings differ -> None -> rebuild.
def load_existing_vectorstore(pdf_path: str, google_api_key: str):
    """
    Loads an already-processed vector store from disk (avoids re-processing).
    If we already processed this exact PDF before, we skip re-embedding it
    to save time and API calls.

    A store is reused only if it was built with the current embedding model
    and chunk settings, and holds one vector per chunk. Stores created before
    index_meta.json existed have no record and are reused as before.
    """
    # Same folder name as process_pdf() used, because the hash of the same file is the same.
    pdf_hash = get_pdf_hash(pdf_path)
    persist_dir = str(CHROMA_BASE_DIR / pdf_hash)

    # Folder missing or empty -> this PDF was never processed -> None.
    if not (os.path.exists(persist_dir) and os.listdir(persist_dir)):
        return None

    # meta stays None for old stores made before index_meta.json existed (they are trusted as before).
    meta = None
    meta_path = os.path.join(persist_dir, INDEX_META_FILE)
    # If index_meta.json exists, read it and compare.
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        # any(...) is True if at least one setting is different, e.g. saved chunk_size 500 vs current 1000.
        if any(meta.get(k) != v for k, v in _index_settings().items()):
            print("[RAG Engine] Existing store was built with different settings; rebuilding.")
            return None

    # Settings match: open the saved store. The embedding model is still needed, because every new
    # QUESTION must be embedded the same way before ChromaDB can compare it with the saved chunks.
    print("[RAG Engine] Found existing vector store. Loading from disk...")
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=google_api_key
    )
    store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )
    # _collection.count() = how many vectors are really saved. If it is not equal to n_chunks, an earlier
    # embedding run failed half-way (eval/README "Defect": an index held 200 of 275 vectors) -> rebuild.
    if meta is not None and store._collection.count() != meta.get("n_chunks"):
        print("[RAG Engine] Existing store is incomplete; rebuilding.")
        return None
    # All checks passed: reuse the store.
    return store
