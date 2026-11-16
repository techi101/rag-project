"""
eval/config.py
─────────────────────────────────────────────────────────────────────────────
Shared settings for the DocuMind AI evaluation harness.

This folder is ADDITIVE: it never imports-and-mutates the live app, never
writes to ../chroma_store/, and nothing here changes how app.py behaves.
Vector stores built for experiments live in eval/chroma_eval/.
"""
# WHAT THIS FILE IS: shared settings every eval script imports (folders, models, configs, scoring helpers).
# Eval harness = scripts that test the RAG app with known questions and score it.
# Example: CONFIGS["baseline"] = chunk 1000, overlap 200, k 4, dense (what documind/settings.py ships);
#   q001's answer "CREATE DATABASE startersql;" is found on page indexes 4 and 65.
# Flow: paths -> load_env() -> DOCUMENTS -> model settings -> CONFIGS -> overlap/answer-page helpers
import os  # reads/sets env variables like GROQ_API_KEY
import sys  # terminal encoding + import path
import pathlib  # easy file paths

# ── BLOCK: force UTF-8 output ───────────────────────────────────────────────
# Windows terminals (cp1252) crash on characters like "─".
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":  # terminal not UTF-8?
    try:  # reconfigure() missing on some streams
        sys.stdout.reconfigure(encoding="utf-8")  # switch output to UTF-8
    except Exception:  # failed -> carry on silently
        pass

# ── BLOCK: folder paths used by every eval script ──────────────────────────
EVAL_DIR = pathlib.Path(__file__).parent  # eval/ (this file's folder)
PROJECT_DIR = EVAL_DIR.parent  # rag-project/: holds .env + documind/
if str(PROJECT_DIR) not in sys.path:  # sys.path = folders Python imports from
    sys.path.insert(0, str(PROJECT_DIR))  # so scripts can "import documind" (read-only)
DATA_DIR = EVAL_DIR / "data"  # question sets + corpus list
CHROMA_EVAL_DIR = EVAL_DIR / "chroma_eval"  # vector stores = embedding DBs, never the app's chroma_store/
RESULTS_DIR = EVAL_DIR / "results"  # summary.json + raw per-question results
CACHE_DIR = EVAL_DIR / "cache"  # saved API results, never pay twice
QA_SET_PATH = DATA_DIR / "qa_set.json"  # "easy" set, 48 questions
QA_SET_HARD_PATH = DATA_DIR / "qa_set_hard.json"  # paraphrased "hard" set (generate/generate_qa_hard.py)


# IN: nothing (reads ../.env) -> OUT: nothing, keys go into os.environ
# e.g. .env line "GROQ_API_KEY=gsk_abc" -> os.environ["GROQ_API_KEY"] == "gsk_abc" (no python-dotenv needed)
def load_env() -> None:
    """Read ../.env into os.environ (no python-dotenv dependency)."""
    env_path = PROJECT_DIR / ".env"  # rag-project/.env
    if not env_path.exists():  # no .env at all
        raise SystemExit(f"Missing {env_path}. Add GOOGLE_API_KEY and GROQ_API_KEY.")  # stop with a clear message
    for line in env_path.read_text(encoding="utf-8").splitlines():  # each .env line
        line = line.strip()  # drop spaces/newline
        if "=" in line and not line.startswith("#"):  # KEY=VALUE, not a comment
            k, v = line.split("=", 1)  # split at FIRST "=" only
            os.environ.setdefault(k.strip(), v.strip())  # terminal-exported key wins over .env
    for required in ("GOOGLE_API_KEY", "GROQ_API_KEY"):  # Google = embeddings, Groq = answers + grading
        if not os.environ.get(required):  # missing or empty
            raise SystemExit(f"{required} not set in .env")  # stop


# ── The corpus ────────────────────────────────────────────────────────────────
# Three documents with deliberately different shapes, so the ablation reveals
# where a setting helps and where it hurts, instead of one global average.

# Paths live in eval/data/documents.json, which is gitignored -- the corpus is local
# study material, and hardcoding absolute paths would both leak the machine's
# directory layout and make the harness unrunnable by anyone else.
# Copy documents.example.json to documents.json and point it at your own PDFs.
DOCUMENTS_PATH = DATA_DIR / "documents.json"  # real paths (gitignored)
DOCUMENTS_EXAMPLE = DATA_DIR / "documents.example.json"  # template to copy


# IN: nothing (reads documents.json) -> OUT: {doc_id: {"path", "label", "shape"}}
# e.g. local ids "mysql" (MySQL Handbook) and "bi"; example file uses {"slides": {"path": ...}}
def _load_documents():
    if DOCUMENTS_PATH.exists():  # file present -> read it
        import json  # only needed here
        return json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))  # dict of documents
    # ── missing -> stop and explain how to create it ("%s" filled by "%" at the end) ──
    raise SystemExit(
        "Missing %s\n\n"  # the missing path
        "The evaluation corpus is local PDFs, so paths are not committed.\n"
        "Copy the example and edit it to point at three documents of your own:\n\n"
        "    cp %s %s\n\n"  # copy command
        "Pick documents with DIFFERENT shapes -- a dense prose document, a\n"
        "technical reference with short pages, and a slide deck. The contrast is\n"
        "what makes chunk-size results meaningful rather than an average."
        % (DOCUMENTS_PATH, "eval/data/" + DOCUMENTS_EXAMPLE.name, "eval/data/" + DOCUMENTS_PATH.name))  # fill blanks


DOCUMENTS = _load_documents()  # runs once on "import config"

# ── Model settings ────────────────────────────────────────────────────────────
# Imported from documind/settings.py, never retyped, so the eval follows the app.
# EMBED_MODEL = text -> 3072 numbers, GEN_MODEL = writes answers, JUDGE = LLM that grades answers.
from documind.settings import (  # the app's own settings
    EMBED_MODEL, GEN_MODEL, GEN_TEMPERATURE, GEN_MAX_TOKENS,  # model names + answer settings
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS, TOP_K_RESULTS,  # baseline chunking + k
)
JUDGE_MODEL = "openai/gpt-oss-120b"   # larger model grades, to avoid self-scoring bias

# gpt-oss-* are REASONING models: hidden reasoning uses tokens BEFORE visible text.
# App uses max_tokens=1024 (GEN_MAX_TOKENS); reasoning overhead 300-900 chars, tight budgets truncate.
# Token = ~3/4 of a word.
JUDGE_MAX_TOKENS = 512  # judge only writes a tiny JSON grade

# ── Configurations under test ────────────────────────────────────────────────
# "baseline" is exactly what documind/settings.py ships today (CHUNK_SIZE=1000,
# CHUNK_OVERLAP=200, TOP_K_RESULTS=4, dense-only similarity search).
# Each config changes ONE thing vs baseline (hybrid_k8 changes two). k = chunks handed to the LLM.
# dense = search by embedding meaning; hybrid = BM25 keywords + dense, fused with RRF (see run_eval.py).
CONFIGS = {
    "baseline":      dict(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, k=TOP_K_RESULTS, retrieval="dense"),  # app as shipped
    "chunk500":      dict(chunk_size=500,  chunk_overlap=100, k=4, retrieval="dense"),  # smaller chunks, 20% overlap
    "chunk2000":     dict(chunk_size=2000, chunk_overlap=400, k=4, retrieval="dense"),  # bigger chunks, 20% overlap
    "k8":            dict(chunk_size=1000, chunk_overlap=200, k=8, retrieval="dense"),  # 8 chunks instead of 4
    "hybrid":        dict(chunk_size=1000, chunk_overlap=200, k=4, retrieval="hybrid"),  # BM25 + dense
    "hybrid_k8":     dict(chunk_size=1000, chunk_overlap=200, k=8, retrieval="hybrid"),  # hybrid + 8 chunks
}

# Chunking is shared across configs that agree on size/overlap, so indexes are
# keyed by the chunking pair only -- 3 distinct indexes per document, not 6.
# IN: doc id, chunk size, overlap -> OUT: index folder name
# e.g. index_key("bi", 1000, 200) -> "bi_cs1000_co200" (cs = chunk size, co = overlap)
def index_key(doc_id: str, chunk_size: int, chunk_overlap: int) -> str:
    return f"{doc_id}_cs{chunk_size}_co{chunk_overlap}"  # build the name


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

# Stopwords = common words ("what", "the") ignored when measuring overlap.
# .split() cuts the text into words, set() = fast lookups.
QUESTION_STOPWORDS = set("""what which who when where how why is are was were do
does did the a an and or of to in on for from with that this these those it its
as by at be been name given used use uses using shown show list three two one
according can could would should may might will there their they you your""".split())


# IN: question, its source page text (lowercase) -> OUT: 0.0 to 1.0
# e.g. "What SQL command is shown for creating the example database?", 4 of 5 content words on page -> 0.8
def question_page_overlap(question: str, page_text_lower: str) -> float:
    """
    Fraction of a question's content words that appear verbatim on its source page.

    High overlap means a keyword matcher can find the page without understanding
    anything, so recall measures the QUESTION SET rather than the retriever. This
    set measured 73% mean overlap, and BM25-only scored 97.7% recall@4 on it --
    the reason a harder, paraphrased set exists.
    """
    import re  # re = regex, text patterns
    # Regex: a letter then 2+ letters/digits/_ -> words of 3+ chars ("sql", "create_db").
    words = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", question.lower())  # unique words
             if w not in QUESTION_STOPWORDS}  # drop stopwords
    if not words:  # only stopwords
        return 1.0  # treat as full overlap (worst case)
    # Substring check: "data" also counts if the page has "database".
    return sum(1 for w in words if w in page_text_lower) / len(words)  # share found on page


_TERM_STOPWORDS = {  # long (6+ letters) but generic words, never "distinctive"
    "document", "according", "information", "provided", "because", "returns",
    "should", "between", "through", "example", "following", "statement",
    "including", "however", "therefore", "without",
}


# IN: reference answer -> OUT: up to `limit` (6) distinctive lowercase terms, in order
# e.g. "The page shows the command `CREATE DATABASE startersql;`." -> ["command", "create", "database", "startersql"]
def answer_key_terms(answer: str, limit: int = 6):
    """Distinctive tokens from a reference answer: long words, numbers, identifiers."""
    import re  # regex, only needed here
    # Regex: words of 6+ chars OR numbers of 2+ digits ("65", "2019"); single digits too common.
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}|\d{2,}", answer)
    seen, out = set(), []  # terms taken / result list
    for t in toks:  # in order of appearance
        t = t.lower()  # compare in lowercase
        if t in _TERM_STOPWORDS or t in seen:  # generic word or repeat
            continue  # skip it
        seen.add(t)  # remember it
        out.append(t)  # keep it
    return out[:limit]  # first 6 only, long answers don't demand too much


# IN: reference answer, every page's lowercase text, source page -> OUT: sorted page indexes holding the answer
# e.g. q001 "CREATE DATABASE startersql;" -> [4, 65] (PyPDFLoader indexes, from 0); 18.6% of answerable Qs have 2+ pages
def acceptable_pages(answer: str, pages_lower, gt_page):
    """
    Pages that genuinely contain the reference answer.

    Requires at least 3 distinctive terms and ALL of them present on the page.
    An earlier draft allowed one missing term, which produced false matches --
    page 14 "contained" an answer on the strength of the generic words 'select'
    and 'salary' alone. Requiring every term removed those.
    """
    terms = answer_key_terms(answer)  # distinctive words of the answer
    pages = {gt_page} if gt_page is not None else set()  # always the source page (None = unanswerable)
    if len(terms) < 3:  # too few terms to judge other pages safely
        return sorted(pages)  # source page only
    for i, text in enumerate(pages_lower):  # (index, text), index from 0
        if all(t in text for t in terms):  # EVERY term on this page
            pages.add(i)  # page counts
    return sorted(pages)  # e.g. [4, 65]
