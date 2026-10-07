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

# WHAT THIS FILE IS: the "brain" (or the librarian + the writer) of DocuMind AI. app.py is only the screen;
# every real RAG step happens here. RAG = Retrieval-Augmented Generation: first FIND the right pages of
# the PDF (retrieval), then let a language model WRITE an answer using only those pages (generation).
# Analogy: an open-book exam. The model is the student, the PDF is the book, and this file is the helper
# who opens the book to the right 4 pages before the student writes the answer.
# Real example: you upload the MySQL Handbook (72 pages, one of the eval documents) and ask
# "What does CREATE DATABASE do?". process_pdf() cuts the pages into chunks, turns each chunk into an
# embedding with Google "models/gemini-embedding-001" and saves them in ChromaDB under chroma_store/.
# run_qa() then finds the 4 closest chunks (for example the page with "CREATE DATABASE startersql;")
# and asks Groq "openai/gpt-oss-20b" to answer from them, citing "[Page 5]" (PyPDFLoader stores that page as 0-based page 4).
# Words used below:
#   chunk        = a small piece of the document text (here at most 1000 characters).
#   embedding    = a list of numbers (here 3072 numbers) that captures the MEANING of a text. Texts with
#                  similar meaning get numbers that are close to each other.
#   vector store = a database that keeps embeddings and can quickly find the ones closest to a new embedding.
#                  Here it is ChromaDB, which saves to a folder on disk (no server needed).
#   LLM          = Large Language Model, the AI that writes the answer (Groq's gpt-oss-20b here).
#   token        = a small piece of a word that LLMs read and write (roughly 3 to 4 English characters).
# Overall flow: PDF -> pages (PyPDFLoader) -> chunks (splitter) -> embeddings (Google) -> ChromaDB
#               -> question -> top 4 chunks -> prompt -> Groq LLM -> answer + page numbers
#
# os = work with folders and files (make the chroma_store/<hash> folder, list it, join paths).
import os
# sys = talk to the Python interpreter itself (fix the terminal encoding, swap the sqlite3 module below).
import sys
# json = turn a Python dict into text and back. Used to write and read index_meta.json.
import json
# hashlib = makes fingerprints (hashes) of data. SHA-256 of the PDF bytes names the PDF's storage folder.
# A hash is a short code computed from the content: the same file always gives the same code,
# a different file (even one changed byte) gives a completely different code.
import hashlib
# Path = an easy way to build file paths. CHROMA_BASE_DIR / pdf_hash makes "chroma_store/<hash>".
from pathlib import Path

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
# ChatGroq = calls an LLM hosted by Groq (a company with very fast AI chips). Used to write answers.
from langchain_groq import ChatGroq
# ChatPromptTemplate = a prompt with blanks like {context} and {question} that get filled in later.
# MessagesPlaceholder = a blank inside the prompt where a whole list of past chat messages is inserted.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# StrOutputParser = takes the LLM's reply object and gives back just the plain text string.
from langchain_core.output_parsers import StrOutputParser
# HumanMessage / AIMessage = one chat message from the user / from the AI, used to pass chat history.
from langchain_core.messages import HumanMessage, AIMessage


# ── Constants ─────────────────────────────────────────────────────────────────

# Folder where all vector stores are saved. Each PDF gets its own sub-folder: chroma_store/<pdf hash>/.
CHROMA_BASE_DIR = Path("./chroma_store")

# How many characters each text chunk should be (~150 words).
# Why 1000: big enough to hold a full idea (a paragraph or a short code example), small enough that
# 4 chunks still make a short prompt. The eval also tests 500 and 2000 (configs chunk500 / chunk2000).
CHUNK_SIZE = 1000

# Overlap between consecutive chunks to avoid cutting context at boundaries.
# Why 200 (20% of 1000): if a sentence is cut at the end of one chunk, the next chunk repeats the last
# 200 characters, so the sentence appears whole in at least one chunk.
CHUNK_OVERLAP = 200

# How many relevant chunks to retrieve per question.
# top-k = "take the k best matches". k = 4 means 4 chunks go into the prompt for each question.
# 4 chunks x up to 1000 characters = at most about 4000 characters of context: enough evidence, short prompt.
TOP_K_RESULTS = 4

# The embedding model name. "models/" is how Google's API names its models. It returns 3072 numbers per text.
EMBED_MODEL = "models/gemini-embedding-001"
# The answer-writing model on Groq: OpenAI's open-weight gpt-oss-20b (20 billion parameters).
# It is a "reasoning" model: it thinks in hidden tokens before writing the visible answer.
GEN_MODEL = "openai/gpt-oss-20b"

# Written next to each vector store. The store's folder is named by the PDF's
# content hash only, so without this record a change to chunking or to the
# embedding model would silently reuse an index built the old way, and a store
# left half-built by a failed embedding call would look complete.
# Example content of chroma_store/<hash>/index_meta.json:
# {"embed_model": "models/gemini-embedding-001", "chunk_size": 1000, "chunk_overlap": 200, "n_chunks": 71}
# (71 = the number of chunks the README reports for the MySQL Handbook at chunk size 1000.)
INDEX_META_FILE = "index_meta.json"


# ── The System Prompt (Prompt Engineering) ────────────────────────────────────
# This is the instruction we give to the LLM that controls its behavior.
# The key rule: answer ONLY from the document context — no hallucination.

# SYSTEM_PROMPT = the standing instructions sent to the LLM before every question (the "system" message).
# {context} is a blank: run_qa() fills it with the 4 retrieved chunks, each labelled "[Page N]".
# Rule 2 is the anti-hallucination rule (hallucination = the AI confidently making something up).
# The exact refusal sentence in rule 2 matters: the eval in eval/ looks for it to detect a refusal.
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


# IN: a chunk's metadata dict  ->  OUT: the page number as a human reads it ("1", "2", ...) or "?".
# WHY: PyPDFLoader counts pages from 0 (the first page is page 0). People count from 1.
# Example: page_label({"page": 3}) -> "4";  page_label({}) -> "?" (no crash when the page is missing).
def page_label(metadata) -> str:
    """
    Human page number (1-based) from a chunk's metadata, or '?' if absent.
    PyPDFLoader stores pages 0-based. A missing page must not crash: the old
    `metadata.get('page', '?') + 1` raised TypeError on its own fallback.
    """
    # .get("page") returns None (instead of an error) if there is no "page" key.
    page = metadata.get("page")
    # Only add 1 if the page really is a whole number (int). Otherwise return "?".
    return str(page + 1) if isinstance(page, int) else "?"


# IN: a list of retrieved chunks (Documents)  ->  OUT: one string to paste into {context} in the prompt.
# WHY: the LLM reads plain text. Labelling each chunk with its page lets the LLM cite pages (rule 5).
# Example: 2 chunks whose metadata says page 4 and page 65 (0-based, as PyPDFLoader stores them) ->
#   "[Page 5]\nCREATE DATABASE startersql; ...\n\n---\n\n[Page 66]\n..."
# This function is also imported by eval/ (read-only), so the eval scores the real prompt format.
def format_docs(docs) -> str:
    """
    Formats retrieved document chunks into a single readable string
    for injection into the LLM prompt.
    """
    # Build one "[Page N]\n<text>" block per chunk, then join the blocks with a "---" divider line
    # (blank lines around it) so the LLM can see where one chunk ends and the next begins.
    return "\n\n---\n\n".join(
        f"[Page {page_label(doc.metadata)}]\n{doc.page_content}"
        for doc in docs
    )


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
    # chunk_size / chunk_overlap come from the constants above (1000 / 200 characters).
    # separators = where the splitter is allowed to cut, best choice first:
    #   "\n\n" = blank line (between paragraphs), "\n" = line break, ". " = end of a sentence,
    #   " " = between words, "" = anywhere (last resort, may cut a word in half).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
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
    print(f"[RAG Engine] Found existing vector store. Loading from disk...")
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


# IN: a vector store + Groq API key  ->  OUT: dict with 3 parts: retriever, llm, prompt.
# WHY: these 3 are built once per PDF and kept in app.py's session, then reused for every question.
# Example: create_qa_chain(store, "gsk_...") -> {"retriever": ..., "llm": ChatGroq(...), "prompt": ...}
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

    # A retriever = the object that, given a question, returns the most relevant chunks.
    # search_type "similarity" = embed the question and return the k nearest chunk embeddings.
    # "Nearest" here: Chroma's default distance is squared L2 (straight-line distance). The README notes that
    # gemini-embedding-001 vectors have length 1, so this gives the same order as cosine similarity
    # (cosine similarity = how much two vectors point the same way; 1 = same direction = same meaning).
    # search_kwargs {"k": 4} = return the top 4 (TOP_K_RESULTS).
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    # Groq gpt-oss-20b: free tier, fast
    # temperature = how random the model's word choices are (0 = always the most likely word, higher =
    # more creative). 0.1 = almost fixed, good for factual answers from a document.
    # max_tokens = 1024 = the most tokens the model may produce (hidden reasoning + visible answer).
    # README: measured cost is about 170 tokens median, 391 worst case, so 1024 leaves plenty of room.
    llm = ChatGroq(
        model=GEN_MODEL,
        groq_api_key=groq_api_key,
        temperature=0.1,
        max_tokens=1024,
    )

    # The prompt template: combines system instructions + chat history + question
    # The message list sent to the LLM, in order:
    #   1. system message = SYSTEM_PROMPT with {context} filled in,
    #   2. the past chat turns (inserted where the "chat_history" placeholder is),
    #   3. the user's new question, filled into "{question}".
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Return the 3 parts as a dict; run_qa() connects them on each question.
    return {
        "retriever": retriever,
        "llm": llm,
        "prompt": prompt,
    }


# IN: the dict from create_qa_chain + a question + the past chat as (question, answer) pairs
#  ->  OUT: {"answer": text, "source_documents": the 4 chunks used}.
# WHY: one full question-answer turn: retrieve (step 5) then generate (step 6).
# Example: run_qa(parts, "How do you create a database?", [])
#   -> {"answer": "Use CREATE DATABASE ... (Page 5)", "source_documents": [4 Documents]}
#   (illustrative answer text; the real one depends on the PDF and the model.)
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
    # Take the 3 parts out of the dict.
    retriever = chain_components["retriever"]
    llm       = chain_components["llm"]
    prompt    = chain_components["prompt"]

    # Step 5: RETRIEVE — find the most relevant document chunks
    # invoke(question) embeds the question with Google and returns the 4 nearest chunks from ChromaDB.
    # Note: only the latest question is used for the search, not the chat history.
    source_docs = retriever.invoke(question)

    # Format them into a readable context string
    # Turn the 4 chunks into one "[Page N] ..." text block for the {context} blank.
    context = format_docs(source_docs)

    # Build the conversation history for the prompt (enables follow-up questions)
    # Convert chat history into LangChain message objects. Each past turn = 1 human + 1 AI message.
    # Example: [("What is SQL?", "SQL is ...")] -> [HumanMessage("What is SQL?"), AIMessage("SQL is ...")]
    history_messages = []
    # Loop over every past (question, answer) pair in order, oldest first.
    for human_msg, ai_msg in chat_history:
        history_messages.append(HumanMessage(content=human_msg))
        history_messages.append(AIMessage(content=ai_msg))

    # Step 6: GENERATE — send context + question to the LLM and get the answer
    # LCEL (LangChain Expression Language): the "|" pipe sends the output of one step into the next,
    # like an assembly line: fill the prompt -> send to the Groq LLM -> take out the plain text.
    chain = prompt | llm | StrOutputParser()
    # The values for the 3 blanks in the prompt: {context}, chat_history and {question}.
    inputs = {
        "context": context,
        "chat_history": history_messages,
        "question": question,
    }
    # invoke() runs the whole chain once and returns the answer string.
    answer = chain.invoke(inputs)

    # gpt-oss-20b is a reasoning model and, rarely (1 question in one eval run),
    # returns empty visible text. Raising max_tokens did not prevent it in
    # eval/verify_tokens.py; retrying once handles it whatever the cause.
    # Empty answer (nothing, or only spaces/newlines) -> ask the model once more with the same input.
    if not answer or not answer.strip():
        answer = chain.invoke(inputs)
    # Still empty after the retry -> show a clear message instead of a blank chat bubble.
    if not answer or not answer.strip():
        answer = "The model returned an empty answer. Please ask again."

    # Return the answer plus the chunks, so app.py can show the source pages under the answer.
    return {
        "answer": answer,
        "source_documents": source_docs,
    }


# IN: path of the PDF + Groq API key  ->  OUT: a short 3-bullet summary text (or an error line).
# WHY: right after upload the user sees what the document is about before asking anything.
# Example: for the MySQL Handbook -> "- Introduces SQL and MySQL...\n- ...\n- ..." (illustrative text).
def generate_summary(pdf_path: str, groq_api_key: str) -> str:
    """
    Reads the first few pages of the document and generates a 3-bullet-point
    executive summary to instantly orient the user.
    """
    # try: any failure (bad key, rate limit, no internet) is caught below so the upload still works.
    try:
        # Read the PDF again, one Document per page.
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Only use the first 5 pages to save tokens and speed up summarizing
        # pages[:5] = the first 5 pages only (fewer pages = fewer tokens = faster and cheaper).
        # Their texts are joined with a newline into one string.
        text_to_summarize = "\n".join([page.page_content for page in pages[:5]])
        
        # A separate LLM client for the summary. temperature 0.3 = a bit more free wording than the Q&A (0.1),
        # fine for a summary. No max_tokens set here, so Groq's default limit applies.
        llm = ChatGroq(
            model=GEN_MODEL,
            groq_api_key=groq_api_key,
            temperature=0.3,
        )
        
        # A 2-message prompt: system instructions (3 bullets, no intro) + the document text in the {text} blank.
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert analyst. Provide a brief, 3-bullet-point executive summary of the following document excerpt. Be concise and professional. Do not include any introductory or concluding remarks, just the 3 bullet points."),
            ("human", "Document text:\n{text}")
        ])
        
        # Same pipe as in run_qa(): prompt -> LLM -> plain text; then run it with the first 5 pages' text.
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"text": text_to_summarize})
    # except: return the error as a bullet line, so the sidebar shows what went wrong instead of crashing.
    except Exception as e:
        return f"- Could not generate summary. ({str(e)})"
