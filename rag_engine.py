"""
rag_engine.py
─────────────────────────────────────────────────────────────────────────────
The BRAIN of the DocuMind AI application.

RAG Pipeline (6 Steps):
  Step 1 — LOAD    : Read the uploaded PDF and extract all text.
  Step 2 — CHUNK   : Break the text into small, overlapping paragraphs.
  Step 3 — EMBED   : Convert each paragraph into a 3072-dimensional vector
                     using Google's gemini-embedding-001 (free-tier API).
  Step 4 — STORE   : Save all vectors into ChromaDB (a Vector Database).
  Step 5 — RETRIEVE: Find the most relevant paragraphs for the user's question.
  Step 6 — GENERATE: Send the relevant paragraphs + question to Groq
                     (openai/gpt-oss-20b) and return a grounded, cited answer.

Why Groq API?
  - Free tier — no credit card needed.
  - Fast generation.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import hashlib
from pathlib import Path

# Fix Windows terminal ASCII encoding issue (allows Unicode/emoji in logs)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ── ChromaDB sqlite3 compatibility (cloud deploys) ────────────────────────────
# ChromaDB requires sqlite3 >= 3.35. Some Linux hosts - including Streamlit
# Community Cloud - ship an older system sqlite3, so `import chromadb` dies at
# startup before the app ever renders. pysqlite3-binary bundles a modern
# sqlite3; swapping it into sys.modules before any Chroma import fixes that.
# It is a linux-only wheel, so its absence on Windows/macOS is expected and
# harmless - those platforms already ship a new enough sqlite3.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# ── Constants ─────────────────────────────────────────────────────────────────

CHROMA_BASE_DIR = Path("./chroma_store")

# How many characters each text chunk should be (~150 words).
CHUNK_SIZE = 1000

# Overlap between consecutive chunks to avoid cutting context at boundaries.
CHUNK_OVERLAP = 200

# How many relevant chunks to retrieve per question.
TOP_K_RESULTS = 4

EMBED_MODEL = "models/gemini-embedding-001"
GEN_MODEL = "openai/gpt-oss-20b"

# Written next to each vector store. The store's folder is named by the PDF's
# content hash only, so without this record a change to chunking or to the
# embedding model would silently reuse an index built the old way, and a store
# left half-built by a failed embedding call would look complete.
INDEX_META_FILE = "index_meta.json"


# ── The System Prompt (Prompt Engineering) ────────────────────────────────────
# This is the instruction we give to the LLM that controls its behavior.
# The key rule: answer ONLY from the document context — no hallucination.

SYSTEM_PROMPT = """You are DocuMind AI, an expert document analyst assistant.
Your role is to provide accurate, insightful answers based STRICTLY on the
provided document context below.

RULES YOU MUST FOLLOW:
1. ONLY use information explicitly found in the provided context.
2. If the answer is not in the context, clearly say: "I could not find this information in the uploaded document."
3. Always be concise, structured, and professional.
4. For complex answers, use bullet points or numbered lists for clarity.
5. When you use information from a specific part of the document, mention the page number.

DOCUMENT CONTEXT:
{context}
"""


# ── Core RAG Functions ────────────────────────────────────────────────────────

def get_pdf_hash(pdf_path: str) -> str:
    """
    Creates a unique fingerprint for a PDF so each document gets its own
    ChromaDB folder. Prevents different PDFs from overwriting each other.
    """
    with open(pdf_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def page_label(metadata) -> str:
    """
    Human page number (1-based) from a chunk's metadata, or '?' if absent.
    PyPDFLoader stores pages 0-based. A missing page must not crash: the old
    `metadata.get('page', '?') + 1` raised TypeError on its own fallback.
    """
    page = metadata.get("page")
    return str(page + 1) if isinstance(page, int) else "?"


def format_docs(docs) -> str:
    """
    Formats retrieved document chunks into a single readable string
    for injection into the LLM prompt.
    """
    return "\n\n---\n\n".join(
        f"[Page {page_label(doc.metadata)}]\n{doc.page_content}"
        for doc in docs
    )


def _index_settings() -> dict:
    """Settings an existing vector store must match to be reused."""
    return {"embed_model": EMBED_MODEL, "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP}


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
    print(f"[RAG Engine] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[RAG Engine] Loaded {len(pages)} pages.")

    # ── Step 2: Chunk ───────────────────────────────────────────────────────
    # We split the full document into small overlapping paragraphs.
    # Without chunking, we'd have to send the entire PDF to the AI on every
    # question, which is slow and expensive.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[RAG Engine] Split into {len(chunks)} chunks.")
    if not chunks:
        # Image-only (scanned) PDFs yield no text; say so instead of building
        # an empty index that answers every question with "could not find".
        raise ValueError("No text could be extracted from this PDF. It may be "
                         "a scanned document; OCR is not supported yet.")

    # ── Steps 3 & 4: Embed & Store ──────────────────────────────────────────
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=google_api_key
    )

    pdf_hash = get_pdf_hash(pdf_path)
    persist_dir = str(CHROMA_BASE_DIR / pdf_hash)
    os.makedirs(persist_dir, exist_ok=True)

    # A store rejected by load_existing_vectorstore (stale settings, or
    # incomplete) still sits in this folder. from_documents would ADD to it,
    # mixing old and new vectors, so empty it first. Deleting the collection
    # through Chroma avoids Windows file locks that break deleting the folder.
    if os.listdir(persist_dir):
        Chroma(persist_directory=persist_dir,
               embedding_function=embedding_model).delete_collection()

    print(f"[RAG Engine] Embedding chunks and saving to: {persist_dir}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    with open(os.path.join(persist_dir, INDEX_META_FILE), "w", encoding="utf-8") as f:
        json.dump({**_index_settings(), "n_chunks": len(chunks)}, f)
    print("[RAG Engine] Vector store created successfully.")
    return vector_store


def load_existing_vectorstore(pdf_path: str, google_api_key: str):
    """
    Loads an already-processed vector store from disk (avoids re-processing).
    If we already processed this exact PDF before, we skip re-embedding it
    to save time and API calls.

    A store is reused only if it was built with the current embedding model
    and chunk settings, and holds one vector per chunk. Stores created before
    index_meta.json existed have no record and are reused as before.
    """
    pdf_hash = get_pdf_hash(pdf_path)
    persist_dir = str(CHROMA_BASE_DIR / pdf_hash)

    if not (os.path.exists(persist_dir) and os.listdir(persist_dir)):
        return None

    meta = None
    meta_path = os.path.join(persist_dir, INDEX_META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if any(meta.get(k) != v for k, v in _index_settings().items()):
            print("[RAG Engine] Existing store was built with different settings; rebuilding.")
            return None

    print(f"[RAG Engine] Found existing vector store. Loading from disk...")
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=google_api_key
    )
    store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )
    if meta is not None and store._collection.count() != meta.get("n_chunks"):
        print("[RAG Engine] Existing store is incomplete; rebuilding.")
        return None
    return store


def create_qa_chain(vector_store: Chroma, groq_api_key: str) -> dict:
    """
    Creates the Q&A chain components using LangChain LCEL (modern approach).

    Components:
    - Retriever: Finds the top-K most relevant chunks from ChromaDB.
    - LLM: Groq openai/gpt-oss-20b — free tier, fast generation.
    - Prompt: Controls how the LLM behaves (grounded, no hallucination).

    Args:
        vector_store: The ChromaDB vector store loaded with PDF data.
        groq_api_key: Groq API key.

    Returns:
        A dict of components to be used by the run_qa() function.
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    # Groq gpt-oss-20b: free tier, fast
    llm = ChatGroq(
        model=GEN_MODEL,
        groq_api_key=groq_api_key,
        temperature=0.1,
        max_tokens=1024,
    )

    # The prompt template: combines system instructions + chat history + question
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    return {
        "retriever": retriever,
        "llm": llm,
        "prompt": prompt,
    }


def run_qa(chain_components: dict, question: str, chat_history: list) -> dict:
    """
    Runs a single Q&A turn through the full RAG pipeline.

    Flow:
        Question → ChromaDB Retrieval → Context Formatting →
        LLM Prompt → Answer + Source Documents

    Args:
        chain_components: The dict returned by create_qa_chain().
        question:         The user's current question.
        chat_history:     List of (human_question, ai_answer) tuples — memory.

    Returns:
        {"answer": str, "source_documents": list}
    """
    retriever = chain_components["retriever"]
    llm       = chain_components["llm"]
    prompt    = chain_components["prompt"]

    # Step 5: RETRIEVE — find the most relevant document chunks
    source_docs = retriever.invoke(question)

    # Format them into a readable context string
    context = format_docs(source_docs)

    # Build the conversation history for the prompt (enables follow-up questions)
    history_messages = []
    for human_msg, ai_msg in chat_history:
        history_messages.append(HumanMessage(content=human_msg))
        history_messages.append(AIMessage(content=ai_msg))

    # Step 6: GENERATE — send context + question to the LLM and get the answer
    chain = prompt | llm | StrOutputParser()
    inputs = {
        "context": context,
        "chat_history": history_messages,
        "question": question,
    }
    answer = chain.invoke(inputs)

    # gpt-oss-20b is a reasoning model and, rarely (1 question in one eval run),
    # returns empty visible text. Raising max_tokens did not prevent it in
    # eval/verify_tokens.py; retrying once handles it whatever the cause.
    if not answer or not answer.strip():
        answer = chain.invoke(inputs)
    if not answer or not answer.strip():
        answer = "The model returned an empty answer. Please ask again."

    return {
        "answer": answer,
        "source_documents": source_docs,
    }


def generate_summary(pdf_path: str, groq_api_key: str) -> str:
    """
    Reads the first few pages of the document and generates a 3-bullet-point
    executive summary to instantly orient the user.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Only use the first 5 pages to save tokens and speed up summarizing
        text_to_summarize = "\n".join([page.page_content for page in pages[:5]])
        
        llm = ChatGroq(
            model=GEN_MODEL,
            groq_api_key=groq_api_key,
            temperature=0.3,
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert analyst. Provide a brief, 3-bullet-point executive summary of the following document excerpt. Be concise and professional. Do not include any introductory or concluding remarks, just the 3 bullet points."),
            ("human", "Document text:\n{text}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"text": text_to_summarize})
    except Exception as e:
        return f"- Could not generate summary. ({str(e)})"
