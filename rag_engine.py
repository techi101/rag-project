"""
rag_engine.py
─────────────────────────────────────────────────────────────────────────────
The BRAIN of the DocuMind AI application.

RAG Pipeline (6 Steps):
  Step 1 — LOAD    : Read the uploaded PDF and extract all text.
  Step 2 — CHUNK   : Break the text into small, overlapping paragraphs.
  Step 3 — EMBED   : Convert each paragraph into a math vector (embedding)
                     using HuggingFace all-MiniLM-L6-v2 (100% Free, Local).
  Step 4 — STORE   : Save all vectors into ChromaDB (a Vector Database).
  Step 5 — RETRIEVE: Find the most relevant paragraphs for the user's question.
  Step 6 — GENERATE: Send the relevant paragraphs + question to Groq (Llama 3)
                     and return a grounded, cited answer.

Why Groq API?
  - 100% FREE tier — no credit card needed.
  - Lightning fast generation.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import hashlib
from pathlib import Path

# Fix Windows terminal ASCII encoding issue (allows Unicode/emoji in logs)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
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


# ── The System Prompt (Prompt Engineering) ────────────────────────────────────
# This is the instruction we give to Gemini that controls its behavior.
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


def format_docs(docs) -> str:
    """
    Formats retrieved document chunks into a single readable string
    for injection into the Gemini prompt.
    """
    return "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?') + 1}]\n{doc.page_content}"
        for doc in docs
    )


def process_pdf(pdf_path: str, google_api_key: str) -> Chroma:
    """
    Runs the full RAG ingestion pipeline for a new PDF.

    Pipeline:
        PDF → PyPDFLoader → Text Chunks → Google Embeddings → ChromaDB

    Args:
        pdf_path: Local path to the PDF file.

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

    # ── Steps 3 & 4: Embed & Store ──────────────────────────────────────────
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
    )

    pdf_hash = get_pdf_hash(pdf_path)
    persist_dir = str(CHROMA_BASE_DIR / pdf_hash)
    os.makedirs(persist_dir, exist_ok=True)

    print(f"[RAG Engine] Embedding chunks and saving to: {persist_dir}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    print("[RAG Engine] Vector store created successfully.")
    return vector_store


def load_existing_vectorstore(pdf_path: str, google_api_key: str):
    """
    Loads an already-processed vector store from disk (avoids re-processing).
    If we already processed this exact PDF before, we skip re-embedding it
    to save time and API calls.
    """
    pdf_hash = get_pdf_hash(pdf_path)
    persist_dir = str(CHROMA_BASE_DIR / pdf_hash)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"[RAG Engine] Found existing vector store. Loading from disk...")
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key
        )
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
    return None


def create_qa_chain(vector_store: Chroma, groq_api_key: str) -> dict:
    """
    Creates the Q&A chain components using LangChain LCEL (modern approach).

    Components:
    - Retriever: Finds the top-K most relevant chunks from ChromaDB.
    - LLM: Groq Llama 3 — incredibly fast and accurate for document Q&A.
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

    # Groq Llama 3 8B: FREE tier, extremely fast
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
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
        Gemini Prompt → Answer + Source Documents

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

    # Step 6: GENERATE — send context + question to Gemini and get the answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": history_messages,
        "question": question,
    })

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
            model="llama-3.1-8b-instant",
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
