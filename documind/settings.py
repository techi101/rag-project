"""
documind/settings.py — every setting of the RAG pipeline in ONE place.

app.py, the rest of documind/ and the eval harness (eval/config.py) all read
these values from here, so the evaluation always measures what the app ships.
"""

# Path = an easy way to build file paths. CHROMA_BASE_DIR / pdf_hash makes "chroma_store/<hash>".
from pathlib import Path

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

# Where the splitter is allowed to cut, best choice first:
#   "\n\n" = blank line (between paragraphs), "\n" = line break, ". " = end of a sentence,
#   " " = between words, "" = anywhere (last resort, may cut a word in half).
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# The embedding model name. "models/" is how Google's API names its models. It returns 3072 numbers per text.
EMBED_MODEL = "models/gemini-embedding-001"
# The answer-writing model on Groq: OpenAI's open-weight gpt-oss-20b (20 billion parameters).
# It is a "reasoning" model: it thinks in hidden tokens before writing the visible answer.
GEN_MODEL = "openai/gpt-oss-20b"

# max_tokens = 1024 = the most tokens the model may produce (hidden reasoning + visible answer).
# README: measured cost is about 170 tokens median, 391 worst case, so 1024 leaves plenty of room.
GEN_MAX_TOKENS = 1024

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
