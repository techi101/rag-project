"""
documind/chain.py — steps 5-6 of the RAG pipeline (retrieve, then generate).

  question -> top 4 chunks from ChromaDB -> prompt -> Groq LLM -> answer + page numbers
"""

# WHAT THIS FILE IS: finds the 4 best chunks for a question, then has the LLM answer from them with pages.
# Example: run_qa(parts, "How do you create a database?", []) -> 4 nearest chunks (one is page 5,
#   "CREATE DATABASE startersql;") -> answer citing "[Page 5]" + the 4 chunks for the sidebar.
# Flow: create_qa_chain() once per PDF -> run_qa() per question: retrieve -> format_docs() -> prompt -> LLM

from langchain_chroma import Chroma  # ChromaDB wrapper, only a type hint here
from langchain_groq import ChatGroq  # calls an LLM hosted on Groq (fast AI chips)
# prompt with {blanks} + a slot for the past chat messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser  # LLM reply -> plain text
from langchain_core.messages import HumanMessage, AIMessage  # one user / one AI chat message

from documind.settings import TOP_K_RESULTS, GEN_MODEL, GEN_TEMPERATURE, GEN_MAX_TOKENS, SYSTEM_PROMPT  # 4, model, 0.1, 1024, rules


# ── BLOCK: chunk formatting helpers ─────────────────────────────────────
# IN: chunk metadata -> OUT: 1-based page as text, or "?" (PyPDFLoader counts from 0)
# e.g. page_label({"page": 3}) -> "4"; page_label({}) -> "?"
def page_label(metadata) -> str:
    """
    Human page number (1-based) from a chunk's metadata, or '?' if absent.
    PyPDFLoader stores pages 0-based. A missing page must not crash: the old
    `metadata.get('page', '?') + 1` raised TypeError on its own fallback.
    """
    page = metadata.get("page")  # None if no "page" key
    return str(page + 1) if isinstance(page, int) else "?"  # add 1 only to a real int


# IN: retrieved chunks -> OUT: one string for {context}, each chunk labelled so the LLM can cite pages
# e.g. pages 4 + 65 (0-based) -> "[Page 5]\nCREATE DATABASE startersql; ...\n\n---\n\n[Page 66]\n..."
# Also imported by eval/, so the eval scores the real prompt format.
def format_docs(docs) -> str:
    """
    Formats retrieved document chunks into a single readable string
    for injection into the LLM prompt.
    """
    return "\n\n---\n\n".join(  # "---" divider shows where chunks split
        f"[Page {page_label(doc.metadata)}]\n{doc.page_content}"  # "[Page N]\n<text>"
        for doc in docs  # one block per chunk
    )

# ── BLOCK: build the Q&A parts (once per PDF) ───────────────────────────
# IN: vector store + Groq key -> OUT: {"retriever", "llm", "prompt"}, kept in app.py's session
# e.g. create_qa_chain(store, "gsk_...") -> {"retriever": ..., "llm": ChatGroq(...), "prompt": ...}
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

    # ── Step 1: retriever = question in, closest chunks out ──
    # Chroma default distance = squared L2; gemini vectors have length 1, so same order as cosine (README).
    retriever = vector_store.as_retriever(  # search tool over the store
        search_type="similarity",  # embed question, take nearest chunks
        search_kwargs={"k": TOP_K_RESULTS},  # top 4
    )

    # ── Step 2: the LLM (Groq gpt-oss-20b, free tier, fast) ──
    llm = ChatGroq(  # Groq client
        model=GEN_MODEL,  # "openai/gpt-oss-20b"
        groq_api_key=groq_api_key,  # user's Groq key
        temperature=GEN_TEMPERATURE,  # 0.1: near fixed, factual
        max_tokens=GEN_MAX_TOKENS,  # 1024: hidden reasoning + answer together
    )

    # ── Step 3: prompt = system rules, then past chat, then new question ──
    prompt = ChatPromptTemplate.from_messages([  # messages in send order
        ("system", SYSTEM_PROMPT),  # rules + {context}
        MessagesPlaceholder(variable_name="chat_history"),  # past turns go here
        ("human", "{question}"),  # the new question
    ])

    return {  # run_qa() joins these per question
        "retriever": retriever,  # finds chunks
        "llm": llm,  # writes the answer
        "prompt": prompt,  # the template
    }


# ── BLOCK: one question-answer turn (steps 5-6) ─────────────────────────
# IN: parts dict + question + past (question, answer) pairs -> OUT: {"answer", "source_documents"}
# e.g. run_qa(parts, "How do you create a database?", []) -> {"answer": "Use CREATE DATABASE ... (Page 5)", ...} (illustrative)
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
    retriever = chain_components["retriever"]  # unpack the 3 parts
    llm       = chain_components["llm"]  # Groq LLM
    prompt    = chain_components["prompt"]  # template

    # ── Step 5: RETRIEVE the 4 closest chunks ──
    source_docs = retriever.invoke(question)  # embeds question; only latest question, not history

    # ── Step 5b: chunks -> "[Page N] ..." text for {context} ──
    context = format_docs(source_docs)  # one labelled string

    # ── Step 5c: past chat -> LangChain messages (for follow-ups) ──
    # e.g. [("What is SQL?", "SQL is ...")] -> [HumanMessage("What is SQL?"), AIMessage("SQL is ...")]
    history_messages = []  # filled below
    for human_msg, ai_msg in chat_history:  # oldest turn first
        history_messages.append(HumanMessage(content=human_msg))  # the user's question
        history_messages.append(AIMessage(content=ai_msg))  # the AI's answer

    # ── Step 6: GENERATE the answer ──
    chain = prompt | llm | StrOutputParser()  # LCEL: "|" pipes fill prompt -> LLM -> text
    inputs = {  # values for the prompt's blanks
        "context": context,  # the 4 labelled chunks
        "chat_history": history_messages,  # past turns
        "question": question,  # new question
    }
    answer = chain.invoke(inputs)  # run once, get answer string

    # Gotcha: reasoning model rarely returns empty text (1 question in one eval run);
    # more max_tokens didn't fix it (eval/verify_tokens.py), so retry once.
    if not answer or not answer.strip():  # empty or only spaces?
        answer = chain.invoke(inputs)  # ask once more, same input
    if not answer or not answer.strip():  # still empty?
        answer = "The model returned an empty answer. Please ask again."  # clear message, not a blank bubble

    return {  # app.py shows source pages under the answer
        "answer": answer,  # the text
        "source_documents": source_docs,  # the 4 chunks used
    }
