"""
eval/config.py
─────────────────────────────────────────────────────────────────────────────
Shared settings for the DocuMind AI evaluation harness.

This folder is ADDITIVE: it never imports-and-mutates the live app, never
writes to ../chroma_store/, and nothing here changes how app.py behaves.
Vector stores built for experiments live in eval/chroma_eval/.
"""
import os
import sys
import pathlib

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

EVAL_DIR = pathlib.Path(__file__).parent
PROJECT_DIR = EVAL_DIR.parent
CHROMA_EVAL_DIR = EVAL_DIR / "chroma_eval"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / "cache"
QA_SET_PATH = EVAL_DIR / "qa_set.json"


def load_env() -> None:
    """Read ../.env into os.environ (no python-dotenv dependency)."""
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        raise SystemExit(f"Missing {env_path}. Add GOOGLE_API_KEY and GROQ_API_KEY.")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    for required in ("GOOGLE_API_KEY", "GROQ_API_KEY"):
        if not os.environ.get(required):
            raise SystemExit(f"{required} not set in .env")


# ── The corpus ────────────────────────────────────────────────────────────────
# Three documents with deliberately different shapes, so the ablation reveals
# where a setting helps and where it hurts, instead of one global average.

DOCUMENTS = {
    "mysql": {
        "path": r"C:\Users\Lenovo\OneDrive\Desktop\Handbook & Code\MySQL Handbook.pdf",
        "label": "MySQL Handbook",
        "shape": "technical reference (code blocks, tables, short lines)",
    },
    "bi": {
        "path": r"C:\Users\Lenovo\OneDrive\Study PDFs\Business Intelligence Exam Companion (A4).pdf",
        "label": "BI Exam Companion",
        "shape": "dense continuous prose (exam notes)",
    },
    "objrec": {
        "path": r"C:\Users\Lenovo\Downloads\11.-Object-Recognition.pdf",
        "label": "Object Recognition Slides",
        "shape": "lecture slides (sparse, fragmented text)",
    },
}

# ── Model settings ────────────────────────────────────────────────────────────
EMBED_MODEL = "models/gemini-embedding-001"
GEN_MODEL = "openai/gpt-oss-20b"      # what the live app uses
JUDGE_MODEL = "openai/gpt-oss-120b"   # larger model grades, to avoid self-scoring bias

# gpt-oss-* are REASONING models: hidden reasoning consumes completion tokens
# BEFORE any visible text. rag_engine.py uses max_tokens=1024; measured reasoning
# overhead runs 300-900 chars, so tight budgets can truncate or empty an answer.
GEN_MAX_TOKENS = 1024        # matches the live app, so baseline numbers transfer
JUDGE_MAX_TOKENS = 512

# ── Configurations under test ────────────────────────────────────────────────
# "baseline" is exactly what rag_engine.py ships today (CHUNK_SIZE=1000,
# CHUNK_OVERLAP=200, TOP_K_RESULTS=4, dense-only similarity search).

CONFIGS = {
    "baseline":      dict(chunk_size=1000, chunk_overlap=200, k=4, retrieval="dense"),
    "chunk500":      dict(chunk_size=500,  chunk_overlap=100, k=4, retrieval="dense"),
    "chunk2000":     dict(chunk_size=2000, chunk_overlap=400, k=4, retrieval="dense"),
    "k8":            dict(chunk_size=1000, chunk_overlap=200, k=8, retrieval="dense"),
    "hybrid":        dict(chunk_size=1000, chunk_overlap=200, k=4, retrieval="hybrid"),
    "hybrid_k8":     dict(chunk_size=1000, chunk_overlap=200, k=8, retrieval="hybrid"),
}

# Chunking is shared across configs that agree on size/overlap, so indexes are
# keyed by the chunking pair only -- 3 distinct indexes per document, not 6.
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

QUESTION_STOPWORDS = set("""what which who when where how why is are was were do
does did the a an and or of to in on for from with that this these those it its
as by at be been name given used use uses using shown show list three two one
according can could would should may might will there their they you your""".split())


def question_page_overlap(question: str, page_text_lower: str) -> float:
    """
    Fraction of a question's content words that appear verbatim on its source page.

    High overlap means a keyword matcher can find the page without understanding
    anything, so recall measures the QUESTION SET rather than the retriever. This
    set measured 73% mean overlap, and BM25-only scored 100% recall@4 on it --
    the reason a harder, paraphrased set exists.
    """
    import re
    words = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", question.lower())
             if w not in QUESTION_STOPWORDS}
    if not words:
        return 1.0
    return sum(1 for w in words if w in page_text_lower) / len(words)


_TERM_STOPWORDS = {
    "document", "according", "information", "provided", "because", "returns",
    "should", "between", "through", "example", "following", "statement",
    "including", "however", "therefore", "without",
}


def answer_key_terms(answer: str, limit: int = 6):
    """Distinctive tokens from a reference answer: long words, numbers, identifiers."""
    import re
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}|\d{2,}", answer)
    seen, out = set(), []
    for t in toks:
        t = t.lower()
        if t in _TERM_STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:limit]


def acceptable_pages(answer: str, pages_lower, gt_page):
    """
    Pages that genuinely contain the reference answer.

    Requires at least 3 distinctive terms and ALL of them present on the page.
    An earlier draft allowed one missing term, which produced false matches --
    page 14 "contained" an answer on the strength of the generic words 'select'
    and 'salary' alone. Requiring every term removed those.
    """
    terms = answer_key_terms(answer)
    pages = {gt_page} if gt_page is not None else set()
    if len(terms) < 3:
        return sorted(pages)
    for i, text in enumerate(pages_lower):
        if all(t in text for t in terms):
            pages.add(i)
    return sorted(pages)
