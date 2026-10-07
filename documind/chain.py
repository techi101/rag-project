"""
documind/chain.py — steps 5-6 of the RAG pipeline (retrieve, then generate).

  question -> top 4 chunks from ChromaDB -> prompt -> Groq LLM -> answer + page numbers
"""

# WHAT THIS FILE IS: the writer's desk. Given a ready index (from ingest.py) it finds the 4 best chunks for a
# question and asks the LLM to answer from them only, with page numbers.
# Real example: run_qa(parts, "How do you create a database?", []) -> the 4 nearest chunks (one of them is
#   page 5 with "CREATE DATABASE startersql;") -> answer text citing "[Page 5]" + those 4 chunks for the sidebar.
# Functions:
#   page_label(), format_docs() = turn chunks into "[Page N] text" blocks (also imported by eval/)
#   create_qa_chain()          = build retriever + LLM + prompt once per PDF
#   run_qa()                   = one question-answer turn (step 5 retrieve, step 6 generate)
#

# Chroma = LangChain's wrapper around ChromaDB (only used here as a type hint for the vector store).
from langchain_chroma import Chroma
# ChatGroq = calls an LLM hosted by Groq (a company with very fast AI chips). Used to write answers.
from langchain_groq import ChatGroq
# ChatPromptTemplate = a prompt with blanks like {context} and {question} that get filled in later.
# MessagesPlaceholder = a blank inside the prompt where a whole list of past chat messages is inserted.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# StrOutputParser = takes the LLM's reply object and gives back just the plain text string.
from langchain_core.output_parsers import StrOutputParser
# HumanMessage / AIMessage = one chat message from the user / from the AI, used to pass chat history.
from langchain_core.messages import HumanMessage, AIMessage

# All the numbers, names and the system prompt come from settings.py.
from documind.settings import TOP_K_RESULTS, GEN_MODEL, GEN_MAX_TOKENS, SYSTEM_PROMPT


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
    # max_tokens = GEN_MAX_TOKENS (1024) from settings.py: hidden reasoning + visible answer together.
    llm = ChatGroq(
        model=GEN_MODEL,
        groq_api_key=groq_api_key,
        temperature=0.1,
        max_tokens=GEN_MAX_TOKENS,
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
