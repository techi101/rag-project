"""
documind/ — the BRAIN of the DocuMind AI application, as a package.

RAG Pipeline (6 Steps):
  Step 1 — LOAD    : Read the uploaded PDF and extract all text.            (ingest.py)
  Step 2 — CHUNK   : Break the text into small, overlapping paragraphs.     (ingest.py)
  Step 3 — EMBED   : Convert each paragraph into a 3072-dimensional vector
                     using Google's gemini-embedding-001 (free-tier API).   (ingest.py)
  Step 4 — STORE   : Save all vectors into ChromaDB (a Vector Database).    (ingest.py)
  Step 5 — RETRIEVE: Find the most relevant paragraphs for the user's question.   (chain.py)
  Step 6 — GENERATE: Send the relevant paragraphs + question to Groq
                     (openai/gpt-oss-20b) and return a grounded, cited answer.  (chain.py)

Files:
  settings.py — every tunable number and model name, plus the system prompt (ONE place).
  ingest.py   — steps 1-4: PDF -> chunks -> embeddings -> ChromaDB, and reuse of saved stores.
  chain.py    — steps 5-6: retriever + LLM + prompt, one Q&A turn, page labels.
  summary.py  — the 3-bullet summary shown right after upload.

Why Groq API?
  - Free tier — no credit card needed.
  - Fast generation.
"""

# WHAT THIS PACKAGE IS: the "brain" (or the librarian + the writer) of DocuMind AI. app.py is only the screen;
# every real RAG step happens in the files of this folder. RAG = Retrieval-Augmented Generation: first FIND the right pages of
# the PDF (retrieval), then let a language model WRITE an answer using only those pages (generation).
# Analogy: an open-book exam. The model is the student, the PDF is the book, and this package is the helper
# who opens the book to the right 4 pages before the student writes the answer.
# Real example: you upload the MySQL Handbook (72 pages, one of the eval documents) and ask
# "What does CREATE DATABASE do?". process_pdf() (ingest.py) cuts the pages into chunks, turns each chunk into an
# embedding with Google "models/gemini-embedding-001" and saves them in ChromaDB under chroma_store/.
# run_qa() (chain.py) then finds the 4 closest chunks (for example the page with "CREATE DATABASE startersql;")
# and asks Groq "openai/gpt-oss-20b" to answer from them, citing "[Page 5]" (PyPDFLoader stores that page as 0-based page 4).
# Words used in every file of this package:
#   chunk        = a small piece of the document text (here at most 1000 characters).
#   embedding    = a list of numbers (here 3072 numbers) that captures the MEANING of a text. Texts with
#                  similar meaning get numbers that are close to each other.
#   vector store = a database that keeps embeddings and can quickly find the ones closest to a new embedding.
#                  Here it is ChromaDB, which saves to a folder on disk (no server needed).
#   LLM          = Large Language Model, the AI that writes the answer (Groq's gpt-oss-20b here).
#   token        = a small piece of a word that LLMs read and write (roughly 3 to 4 English characters).
# Overall flow: PDF -> pages (PyPDFLoader) -> chunks (splitter) -> embeddings (Google) -> ChromaDB   [ingest.py]
#               -> question -> top 4 chunks -> prompt -> Groq LLM -> answer + page numbers    [chain.py]
# All numbers and model names used along the way live in settings.py.
#
# A package = a folder of Python files with an __init__.py. This __init__.py runs FIRST, before any file
# inside documind/ is imported, which is why the two environment fixes below live here: they must
# happen before ChromaDB is ever imported.
#
# sys = talk to the Python interpreter itself (fix the terminal encoding, swap the sqlite3 module below).
import sys

# Fix Windows terminal ASCII encoding issue (allows Unicode/emoji in logs)
# If the terminal does not use UTF-8 (old Windows consoles use other encodings), printing special
# characters could crash the program. UTF-8 = the standard way of storing any character as bytes.
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    # try: switch the output stream to UTF-8. reconfigure() exists on normal Python consoles.
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    # except: some environments (like some hosted servers) do not allow this. Then just carry on;
    # it is only about printing logs, so it must never stop the app.
    except Exception:
        pass

# ── ChromaDB sqlite3 compatibility (cloud deploys) ────────────────────────────
# ChromaDB requires sqlite3 >= 3.35. Some Linux hosts - including Streamlit
# Community Cloud - ship an older system sqlite3, so `import chromadb` dies at
# startup before the app ever renders. pysqlite3-binary bundles a modern
# sqlite3; swapping it into sys.modules before any Chroma import fixes that.
# It is a linux-only wheel, so its absence on Windows/macOS is expected and
# harmless - those platforms already ship a new enough sqlite3.
# try: import pysqlite3 (only installed on Linux, see requirements.txt) and register it under the name
# "sqlite3", so when ChromaDB later says "import sqlite3" it gets the new, modern version.
# __import__("pysqlite3") = the same as "import pysqlite3", written as a function call.
# sys.modules = Python's table of already-loaded modules; pop() takes pysqlite3 out and puts it in as sqlite3.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# except: on Windows/macOS pysqlite3 is not installed. That is fine; do nothing and use the normal sqlite3.
except ModuleNotFoundError:
    pass

# Re-export the public functions, so app.py can write "from documind import run_qa" without knowing
# which file inside the package each one lives in.
from documind.ingest import process_pdf, load_existing_vectorstore, get_pdf_hash
from documind.chain import create_qa_chain, run_qa, format_docs, page_label
from documind.summary import generate_summary
from documind.settings import SYSTEM_PROMPT

__all__ = [
    "process_pdf", "load_existing_vectorstore", "get_pdf_hash",
    "create_qa_chain", "run_qa", "format_docs", "page_label",
    "generate_summary", "SYSTEM_PROMPT",
]
