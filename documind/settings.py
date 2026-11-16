"""
documind/settings.py — every setting of the RAG pipeline in ONE place.

app.py, the rest of documind/ and the eval harness (eval/config.py) all read
these values from here, so the evaluation always measures what the app ships.
"""

# WHAT THIS FILE IS: the control panel of the RAG pipeline: only values, no logic.
# Example: set CHUNK_SIZE = 500 -> index_meta.json still says 1000, so ingest.py rebuilds the index;
#   eval/config.py's "baseline" becomes 500 too.
# Flow: settings.py -> ingest.py / chain.py / summary.py / eval/config.py

from pathlib import Path  # easy paths: CHROMA_BASE_DIR / hash = "chroma_store/<hash>"

# ── BLOCK: retrieval settings ───────────────────────────────────────────
CHROMA_BASE_DIR = Path("./chroma_store")  # one sub-folder per PDF: chroma_store/<pdf hash>/

# 1000 chars (~150 words): holds a full idea, 4 chunks still short. Eval also tests 500 / 2000.
CHUNK_SIZE = 1000  # max characters per chunk

# 20% of 1000: a sentence cut at a chunk end appears whole in the next one.
CHUNK_OVERLAP = 200  # chars repeated between neighbour chunks

TOP_K_RESULTS = 4  # top-k = best 4 chunks per question (~4000 chars max)

# Where the splitter may cut, best first: blank line, line break, sentence end, space, anywhere.
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # "" = last resort, may split a word

# ── BLOCK: model settings ───────────────────────────────────────────────
EMBED_MODEL = "models/gemini-embedding-001"  # Google embedding model, 3072 numbers per text
GEN_MODEL = "openai/gpt-oss-20b"  # Groq answer model, 20B params, thinks in hidden tokens first

GEN_TEMPERATURE = 0.1  # temperature = randomness; 0.1 = near fixed, factual

# token = small word piece (~3 to 4 chars). README: ~170 median, 391 worst case used.
GEN_MAX_TOKENS = 1024  # cap on hidden reasoning + visible answer

SUMMARY_PAGES = 5  # summary reads only first 5 pages (fewer tokens)
SUMMARY_TEMPERATURE = 0.3  # a bit freer wording than Q&A's 0.1

# ── BLOCK: index bookkeeping ────────────────────────────────────────────
# Store folder is named by PDF hash only, so this record catches changed chunking/embed model
# and half-built stores from a failed embedding call.
# e.g. {"embed_model": "models/gemini-embedding-001", "chunk_size": 1000, "chunk_overlap": 200, "n_chunks": 71}
INDEX_META_FILE = "index_meta.json"  # saved in chroma_store/<hash>/; 71 = MySQL Handbook chunks


# ── BLOCK: the system prompt (prompt engineering) ───────────────────────
# Standing rules sent before every question. {context} = 4 retrieved chunks, each "[Page N]".
# Rule 2 blocks hallucination (AI making things up). Gotcha: eval/ detects refusals by that exact sentence.
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
