"""
eval/config.py
─────────────────────────────────────────────────────────────────────────────
Shared settings for the DocuMind AI evaluation harness.

This folder is ADDITIVE: it never imports-and-mutates the live app, never
writes to ../chroma_store/, and nothing here changes how app.py behaves.
Vector stores built for experiments live in eval/chroma_eval/.
"""
# WHAT THIS FILE IS: the "settings board" of the evaluation harness. Every eval script (eval/*.py, eval/generate/,
# eval/verify/) imports it
# (import config) so they all agree on the same folders, model names, test configurations and scoring helpers.
# An EVALUATION HARNESS = a set of scripts that test the RAG app with known questions and give it a score,
# like an exam paper plus an answer key plus a marking scheme.
# Real example: CONFIGS["baseline"] = chunk_size 1000, chunk_overlap 200, k 4, dense retrieval, which is exactly
# what documind/settings.py ships. JUDGE_MODEL "openai/gpt-oss-120b" grades the answers of GEN_MODEL "openai/gpt-oss-20b".
# Real example of a helper: question q001 "What SQL command is shown for creating the example database?" has the
# reference answer "CREATE DATABASE startersql;", which acceptable_pages() finds on page index 4 AND page index 65.
# Overall flow: folder paths -> load_env() reads API keys -> DOCUMENTS read from data/documents.json -> model settings (imported from documind/settings.py)
#   -> CONFIGS to compare -> helper functions that measure word overlap and find every page that holds an answer
#
# os = talk to the operating system; here it reads and sets environment variables (os.environ) such as GROQ_API_KEY.
import os
# sys = Python's own settings; here it is used to check and fix the text encoding of the terminal output.
import sys
# pathlib = easy file paths (join folders with "/", check if a file exists, read a text file).
import pathlib

# Windows terminals sometimes print with an old encoding (like cp1252) and crash on characters such as "─".
# If the terminal is not UTF-8 (UTF-8 = the standard way to store any character as bytes), switch it to UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    # try/except: reconfigure() does not exist on every kind of output stream; if it fails, just carry on silently.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Folder paths used by every eval script:
# EVAL_DIR = the eval/ folder (the folder this file is in). __file__ = the path of this file.
EVAL_DIR = pathlib.Path(__file__).parent
# PROJECT_DIR = the project root (rag-project/), one level up. The .env file and the documind/ package live there.
PROJECT_DIR = EVAL_DIR.parent
# Put the project root on the import path, so every eval script can "import documind" (read-only).
# sys.path = the list of folders Python searches when it sees an import.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
# DATA_DIR = eval/data/, where the question sets and the corpus list live.
DATA_DIR = EVAL_DIR / "data"
# CHROMA_EVAL_DIR = eval/chroma_eval/, where the experiment vector stores are saved (never the app's chroma_store/).
# A VECTOR STORE = a database of embeddings. An EMBEDDING = a list of numbers that captures the meaning of a text.
CHROMA_EVAL_DIR = EVAL_DIR / "chroma_eval"
# RESULTS_DIR = eval/results/, where summary.json and the raw per-question results are written.
RESULTS_DIR = EVAL_DIR / "results"
# CACHE_DIR = eval/cache/. A CACHE = saved results from earlier runs, so the same API call is never paid for twice.
CACHE_DIR = EVAL_DIR / "cache"
# QA_SET_PATH = the original ("easy") question set: 48 questions per eval/README.md.
QA_SET_PATH = DATA_DIR / "qa_set.json"
# QA_SET_HARD_PATH = the paraphrased ("hard") question set built by generate/generate_qa_hard.py.
QA_SET_HARD_PATH = DATA_DIR / "qa_set_hard.json"


# IN: nothing (reads the file ../.env)  ->  OUT: nothing; the keys are put into os.environ (the process's settings).
# WHY: the Google and Groq libraries look for GOOGLE_API_KEY / GROQ_API_KEY in os.environ. Reading .env by hand
#      avoids needing the python-dotenv package just for the eval.
# Example: a .env line "GROQ_API_KEY=gsk_abc" -> os.environ["GROQ_API_KEY"] == "gsk_abc".
def load_env() -> None:
    """Read ../.env into os.environ (no python-dotenv dependency)."""
    env_path = PROJECT_DIR / ".env"
    # No .env file at all -> stop the script with a clear message (SystemExit = exit the program with this text).
    if not env_path.exists():
        raise SystemExit(f"Missing {env_path}. Add GOOGLE_API_KEY and GROQ_API_KEY.")
    # Loop over every line of the .env file.
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # Use only lines that look like KEY=VALUE and are not comments (lines starting with "#").
        if "=" in line and not line.startswith("#"):
            # split("=", 1) splits at the FIRST "=" only, so a value that itself contains "=" stays whole.
            k, v = line.split("=", 1)
            # setdefault: only set the key if it is not already set, so a key exported in the terminal wins over .env.
            os.environ.setdefault(k.strip(), v.strip())
    # Both keys are required: Google for embeddings, Groq for answers and grading. Missing or empty -> stop.
    for required in ("GOOGLE_API_KEY", "GROQ_API_KEY"):
        if not os.environ.get(required):
            raise SystemExit(f"{required} not set in .env")


# ── The corpus ────────────────────────────────────────────────────────────────
# Three documents with deliberately different shapes, so the ablation reveals
# where a setting helps and where it hurts, instead of one global average.

# Paths live in eval/data/documents.json, which is gitignored -- the corpus is local
# study material, and hardcoding absolute paths would both leak the machine's
# directory layout and make the harness unrunnable by anyone else.
# Copy documents.example.json to documents.json and point it at your own PDFs.
DOCUMENTS_PATH = DATA_DIR / "documents.json"
DOCUMENTS_EXAMPLE = DATA_DIR / "documents.example.json"


# IN: nothing (reads eval/data/documents.json)  ->  OUT: dict of documents {doc_id: {"path", "label", "shape"}}.
# WHY: the PDFs are the owner's local study material, so their paths are kept out of git (documents.json is gitignored).
# Example (from documents.example.json): {"slides": {"path": "/path/to/a/lecture-deck.pdf", "label": "Lecture Slides", ...}}.
# The owner's local corpus uses doc ids like "mysql" (the MySQL Handbook) and "bi" (see eval/README.md and qa_set.json).
def _load_documents():
    # File exists -> read it. json is imported inside the function because only this function needs it.
    if DOCUMENTS_PATH.exists():
        import json
        return json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    # File missing -> stop and explain how to create it (copy documents.example.json and edit the paths).
    # The "%s" markers in the message are filled with the file names by the "%" operator at the end.
    raise SystemExit(
        "Missing %s\n\n"
        "The evaluation corpus is local PDFs, so paths are not committed.\n"
        "Copy the example and edit it to point at three documents of your own:\n\n"
        "    cp %s %s\n\n"
        "Pick documents with DIFFERENT shapes -- a dense prose document, a\n"
        "technical reference with short pages, and a slide deck. The contrast is\n"
        "what makes chunk-size results meaningful rather than an average."
        % (DOCUMENTS_PATH, "eval/data/" + DOCUMENTS_EXAMPLE.name, "eval/data/" + DOCUMENTS_PATH.name))


# Run the loader once, when any script does "import config". Every script then uses config.DOCUMENTS.
DOCUMENTS = _load_documents()

# ── Model settings ────────────────────────────────────────────────────────────
# EMBED_MODEL, GEN_MODEL, GEN_TEMPERATURE, GEN_MAX_TOKENS and the baseline chunking are IMPORTED from
# documind/settings.py, never retyped here, so changing the app's settings automatically changes what the
# eval measures. Scripts read them as config.GEN_MODEL etc. (that is why they are imported here).
# EMBED_MODEL turns text into 3072-number vectors. GEN_MODEL writes the answers. JUDGE_MODEL is a bigger model that grades them.
# A JUDGE MODEL (LLM-as-judge) = an LLM that reads the question, the correct answer and the app's answer and gives a grade.
from documind.settings import (
    EMBED_MODEL, GEN_MODEL, GEN_TEMPERATURE, GEN_MAX_TOKENS,
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS, TOP_K_RESULTS,
)
JUDGE_MODEL = "openai/gpt-oss-120b"   # larger model grades, to avoid self-scoring bias

# gpt-oss-* are REASONING models: hidden reasoning consumes completion tokens
# BEFORE any visible text. The app uses max_tokens=1024 (GEN_MAX_TOKENS); measured reasoning
# overhead runs 300-900 chars, so tight budgets can truncate or empty an answer.
# A TOKEN = a small piece of text (roughly 3/4 of an English word) that LLMs read and write.
# max tokens = the most tokens the model may write in one reply. Same ceiling as the app, so baseline numbers transfer.
# The judge only writes a tiny JSON grade, so 512 tokens is plenty.
JUDGE_MAX_TOKENS = 512

# ── Configurations under test ────────────────────────────────────────────────
# "baseline" is exactly what documind/settings.py ships today (CHUNK_SIZE=1000,
# CHUNK_OVERLAP=200, TOP_K_RESULTS=4, dense-only similarity search).

# The six configurations to compare. Each one changes ONE thing compared to "baseline" (hybrid_k8 changes two):
#   chunk500 / chunk2000 = smaller / bigger chunks (overlap kept at 20% of the chunk size: 100 / 400)
#   k8 = retrieve 8 chunks instead of 4. k (also called TOP-K) = how many best-matching chunks are handed to the LLM.
#   hybrid = mix keyword search (BM25) with meaning search (dense), fused with RRF (both explained in run_eval.py).
#   "dense" = search by comparing embeddings (meaning), the way the live app searches.
CONFIGS = {
    "baseline":      dict(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, k=TOP_K_RESULTS, retrieval="dense"),
    "chunk500":      dict(chunk_size=500,  chunk_overlap=100, k=4, retrieval="dense"),
    "chunk2000":     dict(chunk_size=2000, chunk_overlap=400, k=4, retrieval="dense"),
    "k8":            dict(chunk_size=1000, chunk_overlap=200, k=8, retrieval="dense"),
    "hybrid":        dict(chunk_size=1000, chunk_overlap=200, k=4, retrieval="hybrid"),
    "hybrid_k8":     dict(chunk_size=1000, chunk_overlap=200, k=8, retrieval="hybrid"),
}

# Chunking is shared across configs that agree on size/overlap, so indexes are
# keyed by the chunking pair only -- 3 distinct indexes per document, not 6.
# IN: document id + chunk size + overlap  ->  OUT: a folder name for that index.
# WHY: configs with the same chunking share one index, so only the chunking pair goes into the name.
# Example: index_key("bi", 1000, 200) -> "bi_cs1000_co200" (cs = chunk size, co = chunk overlap).
def index_key(doc_id: str, chunk_size: int, chunk_overlap: int) -> str:
    return f"{doc_id}_cs{chunk_size}_co{chunk_overlap}"


# ── Ground-truth pages ───────────────────────────────────────────────────────
# Recall@k originally scored a MISS whenever no retrieved chunk came from the
# single page a question was generated from. Adversarial testing
# (verify_method.py CHECK 2) showed some answers genuinely appear on more than
# one page -- e.g. "CREATE DATABASE startersql;" is on both page 4 and page 65 of
# the MySQL handbook. Punishing the retriever for returning page 65 is a FALSE
# NEGATIVE.
#
# So we compute an ACCEPTABLE set per question and report recall two ways:
#   strict  -- only the originating page counts   (lower bound)
#   lenient -- any page truly containing the answer (upper bound)
# The honest figure lies between them, and quoting both is the point.

# STOPWORDS = very common words ("what", "the", "of") that carry no meaning about the topic.
# They are ignored when measuring how many of a question's words are copied from its page.
# The triple-quoted text is just a list of words; .split() cuts it at spaces/newlines and set() makes lookups fast.
QUESTION_STOPWORDS = set("""what which who when where how why is are was were do
does did the a an and or of to in on for from with that this these those it its
as by at be been name given used use uses using shown show list three two one
according can could would should may might will there their they you your""".split())


# IN: a question + the full text of its source page in lowercase  ->  OUT: a number from 0.0 to 1.0.
# WHY: if a question copies most words from its page, plain keyword search finds the page without understanding,
#      so the test cannot tell a smart retriever from a dumb one (BM25 scored 97.7% on the easy set, per eval/README.md).
# Example: "What SQL command is shown for creating the example database?" -> content words like "sql", "command",
#   "creating", "example", "database"; if 4 of those 5 appear on the page -> 0.8 (80% overlap).
def question_page_overlap(question: str, page_text_lower: str) -> float:
    """
    Fraction of a question's content words that appear verbatim on its source page.

    High overlap means a keyword matcher can find the page without understanding
    anything, so recall measures the QUESTION SET rather than the retriever. This
    set measured 73% mean overlap, and BM25-only scored 97.7% recall@4 on it --
    the reason a harder, paraphrased set exists.
    """
    # re = regular expressions: a mini-language for finding text patterns.
    import re
    # Regex [a-z][a-z0-9_]{2,} piece by piece: [a-z] = one lowercase letter to start, then [a-z0-9_]{2,} = 2 or more
    # letters/digits/underscores. So it finds words of 3+ characters that start with a letter ("sql", "create_db").
    # The set {...} keeps each word once and drops stopwords.
    words = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", question.lower())
             if w not in QUESTION_STOPWORDS}
    # Only stopwords in the question -> nothing to measure; return 1.0 (treat it as fully overlapping, the worst case).
    if not words:
        return 1.0
    # Count content words found anywhere in the page text, divided by the number of content words.
    # Note: "w in page_text_lower" is a substring check, so "data" also counts if the page has "database".
    return sum(1 for w in words if w in page_text_lower) / len(words)


# Words that are long (6+ letters) but generic, so they must not count as "distinctive" answer terms.
_TERM_STOPWORDS = {
    "document", "according", "information", "provided", "because", "returns",
    "should", "between", "through", "example", "following", "statement",
    "including", "however", "therefore", "without",
}


# IN: a reference answer text  ->  OUT: up to `limit` (default 6) distinctive lowercase terms, in order of appearance.
# WHY: to check if a page really contains an answer, we look for the answer's rare, specific words, not "the" or "is".
# Example: "The page shows the command `CREATE DATABASE startersql;`." -> ["command", "create", "database", "startersql"].
def answer_key_terms(answer: str, limit: int = 6):
    """Distinctive tokens from a reference answer: long words, numbers, identifiers."""
    # re = regular expressions, imported inside the function because only these helpers need it.
    import re
    # Regex piece by piece, two alternatives joined by "|":
    #   [A-Za-z_][A-Za-z0-9_]{5,} = a letter or underscore, then 5 or more letters/digits/underscores -> words of 6+ characters
    #   \d{2,} = a number with 2 or more digits (like "65" or "2019"); single digits are too common to be distinctive.
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}|\d{2,}", answer)
    # seen = terms already taken (to skip duplicates); out = the list we return.
    seen, out = set(), []
    # Loop over each matched token in the order it appears in the answer.
    for t in toks:
        t = t.lower()
        # Skip generic words and repeats.
        if t in _TERM_STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    # Keep only the first `limit` terms (6 by default) so a long answer does not demand too many matches.
    return out[:limit]


# IN: reference answer + list of every page's lowercase text + the page the question came from
#     ->  OUT: sorted list of page indexes that contain the answer.
# WHY: an answer can appear on more than one page. Recall "lenient" credits any of these pages, so the retriever is
#      not punished for finding a correct copy elsewhere. 18.6% of answerable questions have more than one valid page
#      (eval/README.md).
# Example: q001's answer "CREATE DATABASE startersql;" -> [4, 65] (page indexes as PyPDFLoader counts them, from 0).
def acceptable_pages(answer: str, pages_lower, gt_page):
    """
    Pages that genuinely contain the reference answer.

    Requires at least 3 distinctive terms and ALL of them present on the page.
    An earlier draft allowed one missing term, which produced false matches --
    page 14 "contained" an answer on the strength of the generic words 'select'
    and 'salary' alone. Requiring every term removed those.
    """
    terms = answer_key_terms(answer)
    # Always include the original page (if the question has one; unanswerable questions have gt_page None).
    pages = {gt_page} if gt_page is not None else set()
    # Fewer than 3 distinctive terms is too weak to judge another page safely -> return only the original page.
    # 3 = minimum so that a couple of generic words cannot make a random page "match".
    if len(terms) < 3:
        return sorted(pages)
    # Check every page: enumerate gives (index, text) pairs, index starting from 0.
    for i, text in enumerate(pages_lower):
        # The page counts only if ALL the terms appear on it (one missing term was allowed before and gave false matches).
        if all(t in text for t in terms):
            pages.add(i)
    return sorted(pages)
