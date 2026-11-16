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

# WHAT THIS FILE IS: the package entry point; app.py is the screen, documind/ does every RAG step.
# RAG = Retrieval-Augmented Generation: FIND the right PDF pages, then an LLM WRITES the answer from them.
# Example: MySQL Handbook (72 pages), "What does CREATE DATABASE do?" -> top 4 chunks -> answer citing "[Page 5]".
# Flow: PDF -> pages -> chunks -> embeddings -> ChromaDB [ingest.py] -> question -> top 4 chunks -> Groq -> answer [chain.py]
# Terms: chunk = text piece (max 1000 chars), embedding = 3072 numbers for meaning,
#   vector store = DB that finds the closest embeddings (ChromaDB, on disk), LLM = the answer-writing AI.
# Runs FIRST on any documind import, so the 2 environment fixes below happen before ChromaDB loads.
import sys  # access Python itself: stdout, loaded modules

# ── BLOCK: force UTF-8 terminal output ──────────────────────────────────
# Old Windows consoles can crash when printing special characters.
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':  # terminal not UTF-8?
    try:  # some hosts don't allow this
        sys.stdout.reconfigure(encoding='utf-8')  # switch output to UTF-8
    except Exception:  # not allowed: only logs, so carry on
        pass  # do nothing

# ── BLOCK: ChromaDB sqlite3 fix (cloud deploys) ─────────────────────────
# ChromaDB needs sqlite3 >= 3.35; Streamlit Community Cloud ships older, so import chromadb dies.
# pysqlite3-binary is a Linux-only wheel; on Windows/macOS it is absent and that's fine.
try:  # only works where pysqlite3 is installed
    __import__("pysqlite3")  # same as "import pysqlite3"
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # sys.modules = loaded modules; serve it as "sqlite3"
except ModuleNotFoundError:  # Windows/macOS: not installed
    pass  # normal sqlite3 is new enough there

# ── BLOCK: re-export the public functions ───────────────────────────────
# So app.py can write "from documind import run_qa" without knowing the file.
from documind.ingest import process_pdf, load_existing_vectorstore, get_pdf_hash  # steps 1-4 + hash
from documind.chain import create_qa_chain, run_qa, format_docs, page_label  # steps 5-6 + helpers
from documind.summary import generate_summary  # 3-bullet upload summary
from documind.settings import SYSTEM_PROMPT  # the LLM's rules

__all__ = [  # names given by "from documind import *"
    "process_pdf", "load_existing_vectorstore", "get_pdf_hash",  # ingest.py
    "create_qa_chain", "run_qa", "format_docs", "page_label",  # chain.py
    "generate_summary", "SYSTEM_PROMPT",  # summary.py + settings.py
]
