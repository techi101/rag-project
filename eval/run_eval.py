"""
eval/run_eval.py
─────────────────────────────────────────────────────────────────────────────
The scoring harness. Runs every question through a given configuration and
measures four things that matter independently:

  1. RETRIEVAL RECALL@k  -- did the retriever surface a chunk from the page the
                            answer actually lives on? This is the honest test of
                            the search half of RAG, and it does not care what the
                            LLM said afterwards.

  2. ANSWER ACCURACY     -- graded by a larger model (gpt-oss-120b) against the
                            reference answer: CORRECT / PARTIAL / INCORRECT.

  3. HALLUCINATION RATE  -- on questions whose answers are NOT in the document,
                            how often does the system invent one instead of
                            saying it cannot find it? Lower is better.

  4. TRUNCATION RATE     -- how often generation stops at the token ceiling. This
                            matters because gpt-oss-20b is a reasoning model and
                            burns completion budget on hidden reasoning before it
                            writes anything visible.

Separating 1 from 2 is the point. If accuracy is poor you need to know whether
the retriever handed the model the wrong pages, or the model fumbled the right
ones. Those have opposite fixes.

Usage:
    py -3.12 eval/run_eval.py                  # every configuration
    py -3.12 eval/run_eval.py baseline         # just one
    py -3.12 eval/run_eval.py baseline hybrid  # a subset
"""
# WHAT THIS FILE IS: the "examiner" of the project. It gives every test question to the RAG system (retrieve chunks,
# write an answer with the app's real prompt), then a bigger "teacher" model marks the answer, and the script adds up
# the marks into recall, accuracy, hallucination and truncation numbers.
# Real example: question q001 "What SQL command is shown for creating the example database?" (MySQL handbook).
#   retrieve 4 chunks -> pages like [4, ...] -> gpt-oss-20b answers "CREATE DATABASE startersql;" -> gpt-oss-120b
#   grades it CORRECT -> the row is cached in eval/cache/runs/ -> summary.json gets recall@k, MRR, correct_pct ...
# Real numbers (eval/README.md, from results/summary.json): baseline on the easy set scored 17 of 48 questions,
# recall@4 100.0%, MRR 0.772, accuracy 100.0% (15/15), hallucination 0.0% on 2 probes. These are PARTIAL runs.
# Overall flow: load questions -> for each config -> for each question: cache? reuse : retrieve -> generate -> judge
#   -> save row -> score_row -> summarize -> merge into summary.json -> print a table
#
# Key words used below:
#   RECALL@k = share of answerable questions where at least one of the k retrieved chunks came from the right page.
#   MRR (Mean Reciprocal Rank) = average of 1/rank of the first right chunk (rank 1 -> 1.0, rank 2 -> 0.5, missing -> 0).
#   HALLUCINATION = the model inventing an answer that is not in the document.
#   TRUNCATION = the reply was cut off because it hit the max token limit (finish_reason "length").
#
# json = read the question sets and write cache/results files.
import json
# re = regular expressions (text patterns): used to split text into words and to strip ``` fences from judge output.
import re
# sys = command-line arguments (config names, --hard, --cached-only) and the import path.
import sys
# time = measure answer latency (how long the LLM took) and sleep between retries.
import time
# hashlib = make a short fingerprint (sha256) of each question+config, used as the cache file name.
import hashlib
# pathlib = file paths.
import pathlib
# defaultdict(float) = a dict that starts every new key at 0.0, handy for adding up RRF scores.
from collections import defaultdict

# Make "import config" find eval/config.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

# Read API keys from .env. (config.py already put the project root on the import path, so
# "from documind import ..." works.)
config.load_env()

# Chroma = open the vector stores that build_index.py saved in eval/chroma_eval/.
from langchain_chroma import Chroma
# GoogleGenerativeAIEmbeddings = embed the QUESTION with the same model used for the chunks (needed for the search).
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# ChatGroq = LangChain client for Groq's hosted models (gpt-oss-20b answers, gpt-oss-120b judges).
from langchain_groq import ChatGroq
# ChatPromptTemplate = a fill-in-the-blanks message list; MessagesPlaceholder = a slot for earlier chat messages.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import the LIVE app's prompt and formatter so these numbers describe the real
# system. Importing is read-only; documind/ is never modified.
from documind import SYSTEM_PROMPT, format_docs

# RUN_CACHE = eval/cache/runs/: one JSON file per (config, question). Created if missing.
RUN_CACHE = config.CACHE_DIR / "runs"
RUN_CACHE.mkdir(parents=True, exist_ok=True)

# RRF = Reciprocal Rank Fusion: a way to merge two ranked lists. Each chunk gets 1 / (60 + rank) from every list
# it appears in, and the totals decide the final order. 60 is the usual constant from the original RRF paper; it
# stops the rank-1 item of one list from completely dominating.
RRF_K = 60          # reciprocal-rank-fusion constant, standard value


# ── Retrieval ────────────────────────────────────────────────────────────────

# IN: doc id + one config dict  ->  OUT: an object whose retrieve(question) returns the best k chunks.
# WHY: one class for both search styles, so run_config does not care which config it is testing.
# Example: Retriever("mysql", CONFIGS["baseline"]).retrieve("What SQL command ...") -> 4 dicts {"text", "page"}.
class Retriever:
    """Dense (Chroma) or hybrid (BM25 + dense, fused with RRF)."""

    # IN: doc id + config  ->  OUT: a ready retriever (store opened, chunk texts loaded, BM25 built if hybrid).
    # Example: doc "bi" + baseline -> opens eval/chroma_eval/bi_cs1000_co200 and bi_cs1000_co200_chunks.json.
    def __init__(self, doc_id, cfg):
        # Work out which index this config needs (same naming as build_index.py).
        key = config.index_key(doc_id, cfg["chunk_size"], cfg["chunk_overlap"])
        persist_dir = config.CHROMA_EVAL_DIR / key
        chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")
        # Index not built yet -> raise FileNotFoundError; run_config catches it and skips this document's questions.
        if not persist_dir.exists() or not chunks_path.exists():
            # Raised (not SystemExit) so a partial run can skip these questions and
            # pick them up later: results are cached per question, so a second run
            # after the index finishes only pays for what is still missing.
            raise FileNotFoundError(
                "index %s not built yet -- run eval/build_index.py" % key)

        # k = how many chunks to return; mode = "dense" or "hybrid".
        self.k = cfg["k"]
        self.mode = cfg["retrieval"]
        # Open the saved Chroma store with the same embedding model.
        self.store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL),
        )
        # Load the plain chunk texts (needed for BM25).
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

        # Hybrid only: build a BM25 index over all chunks.
        # BM25 = a classic keyword-search formula (from 1994): a chunk scores high if it contains the question's words,
        # especially rare words, with a bonus that saturates so one word repeated 50 times does not win. No AI, no API.
        # BM25Okapi comes from the rank-bm25 package (imported here so dense-only runs do not need it).
        if self.mode == "hybrid":
            from rank_bm25 import BM25Okapi
            corpus = [self._tokenize(c["text"]) for c in self.chunks]
            self.bm25 = BM25Okapi(corpus)

    # IN: any text  ->  OUT: list of lowercase words.
    # WHY: BM25 needs the chunks and the question split into words the same way.
    # Example: "CREATE DATABASE startersql;" -> ["create", "database", "startersql"].
    # Regex [a-z0-9_]+ = one or more lowercase letters, digits or underscores in a row (punctuation and spaces split words).
    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9_]+", text.lower())

    # IN: a question  ->  OUT: list of k {text, page} dicts, best first.
    # WHY: this is the "R" in RAG that recall@k measures.
    # Example: k=4 dense -> the 4 chunks whose embeddings are closest to the question's embedding.
    def retrieve(self, question):
        """Return a list of {text, page} dicts, best first."""
        # Dense search: ask Chroma for k*2 nearest chunks (8 when k=4). The extra ones are only used by hybrid fusion.
        dense_docs = self.store.similarity_search(question, k=self.k * 2)
        # Turn LangChain Documents into simple dicts {"text", "page"}.
        dense = [{"text": d.page_content, "page": d.metadata.get("page", 0)}
                 for d in dense_docs]

        # Dense mode: return the top k only (same as the live app's retriever).
        if self.mode == "dense":
            return dense[:self.k]

        # Hybrid: fuse dense and BM25 rankings with reciprocal rank fusion.
        # BM25 score for every chunk against the question's words.
        scores = self.bm25.get_scores(self._tokenize(question))
        # Indexes of the k*2 highest-scoring chunks: sort chunk numbers by their score, biggest first, keep the first k*2.
        top_bm25_idx = sorted(range(len(scores)), key=lambda i: scores[i],
                              reverse=True)[:self.k * 2]
        # Turn those indexes into {text, page} dicts.
        bm25 = [{"text": self.chunks[i]["text"], "page": self.chunks[i]["page"]}
                for i in top_bm25_idx]

        # fused = total RRF score per chunk; holder = the chunk dict for each key.
        fused = defaultdict(float)
        holder = {}
        # Add dense ranks. sig (signature) = the first 200 characters of the chunk, used as its ID so the same chunk found by
        # both searches gets both scores. rank starts at 0, so "+ 1" makes the top chunk rank 1: score 1/(60+1).
        for rank, d in enumerate(dense):
            sig = d["text"][:200]
            fused[sig] += 1.0 / (RRF_K + rank + 1)
            holder[sig] = d
        # Add BM25 ranks in exactly the same way.
        for rank, d in enumerate(bm25):
            sig = d["text"][:200]
            fused[sig] += 1.0 / (RRF_K + rank + 1)
            holder[sig] = d

        # Sort chunk keys by total score, highest first, and return the top k chunks.
        ranked = sorted(fused, key=lambda s: fused[s], reverse=True)
        return [holder[s] for s in ranked[:self.k]]


# ── Generation ───────────────────────────────────────────────────────────────

# IN: nothing  ->  OUT: the answer model client: gpt-oss-20b, temperature 0.1, max 1024 output tokens.
# WHY: the same settings as documind/chain.py create_qa_chain(), so the numbers describe the live app.
# TEMPERATURE = how random the model's word choice is; 0.1 = almost always the most likely word (stable answers).
# Example: make_llm().invoke(messages) -> an AIMessage with .content "The command is CREATE DATABASE startersql; ..."
def make_llm():
    return ChatGroq(model=config.GEN_MODEL, temperature=config.GEN_TEMPERATURE,
                    max_tokens=config.GEN_MAX_TOKENS)


# IN: answer model, question, retrieved chunks  ->  OUT: {"answer", "latency_s", "finish_reason", "completion_tokens"}.
# WHY: builds exactly the app's prompt (SYSTEM_PROMPT + format_docs) so we test the real behaviour.
# Example: -> {"answer": "...startersql...", "latency_s": 0.68, "finish_reason": "stop", "completion_tokens": 170}
#   (illustrative values; finish_reason "stop" = finished normally, "length" = cut off at max tokens).
def generate_answer(llm, question, retrieved):
    """Run the app's real prompt over the retrieved context."""
    # _D = a tiny stand-in for a LangChain Document (it only needs .page_content and .metadata) so format_docs() works.
    class _D:                       # format_docs() expects .page_content/.metadata
        def __init__(self, d):
            self.page_content = d["text"]
            self.metadata = {"page": d["page"]}

    # Join the chunks into one context string: "[Page N]\n<text>" blocks separated by "---".
    context = format_docs([_D(d) for d in retrieved])
    # The same message layout as the live app: system rules + (empty) chat history + the question.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    # Fill the blanks. chat_history=[] because every eval question is asked fresh, with no earlier conversation.
    msgs = prompt.format_messages(context=context, chat_history=[], question=question)

    # Stopwatch around the LLM call = latency in seconds.
    t0 = time.time()
    resp = llm.invoke(msgs)
    latency = time.time() - t0
    # response_metadata holds finish_reason and token_usage from Groq; "or {}" guards against None.
    meta = resp.response_metadata or {}
    return {
        "answer": resp.content,
        "latency_s": round(latency, 2),
        # "?" if the provider did not report a finish reason.
        "finish_reason": meta.get("finish_reason", "?"),
        "completion_tokens": (meta.get("token_usage") or {}).get("completion_tokens"),
    }


# ── Judging ──────────────────────────────────────────────────────────────────

# The judge's instructions. {question}, {reference}, {answer} are blanks filled by .format(); the doubled {{ }} make
# literal braces in the output, so the judge sees an example JSON object.
JUDGE_PROMPT = """You are grading a document-QA system. Compare the SYSTEM ANSWER
to the REFERENCE ANSWER for the given question.

Grade CORRECT   if the system answer conveys the same key fact as the reference,
                even if worded differently or with extra detail.
Grade PARTIAL   if it is on the right topic and partly right, but misses or
                garbles the key fact.
Grade INCORRECT if it states something different from, or contradicting, the
                reference.
Grade REFUSED   if the system said it could not find the information.

Judge only factual agreement. Ignore style, length, formatting and citations.

Return STRICT JSON only:
{{"grade": "CORRECT|PARTIAL|INCORRECT|REFUSED", "why": "at most 15 words"}}

QUESTION: {question}

REFERENCE ANSWER: {reference}

SYSTEM ANSWER: {answer}
"""


# IN: judge model, question, reference answer, system answer  ->  OUT: {"grade": ..., "why": ...}.
# WHY: accuracy and hallucination need a grade per answer; a bigger model (gpt-oss-120b) grades to avoid a model
#      marking its own homework. Grades: CORRECT / PARTIAL / INCORRECT / REFUSED.
# Example: answer "I could not find this information in the uploaded document." -> {"grade": "REFUSED", ...}
#   without calling the judge at all (fast path below).
def judge(judge_llm, question, reference, answer):
    # Empty answer -> INCORRECT straight away (the user would have seen a blank reply).
    if not answer or not answer.strip():
        return {"grade": "INCORRECT", "why": "empty answer"}

    # Fast path for the exact refusal the system prompt asks for.
    #
    # Two guards, both added after adversarial testing (see verify_method.py):
    #   - length < 200: the template refusal is 59 chars. A 340-char answer that
    #     happens to contain "could not find this information" mid-sentence is a
    #     real answer, and grading it REFUSED would understate hallucination.
    #   - phrase within the first 100 chars: a refusal leads with it; prose that
    #     merely mentions the phrase does not.
    # Lowercase copy so the phrase search ignores capital letters.
    low = answer.lower()
    refusal_phrases = (
        "could not find this information",
        "could not find that information",
        "cannot find this information",
        "i could not find",
        "i cannot find",
    )
    # Position of the earliest refusal phrase found; -1 if none. ("or [-1]" handles the empty list, since min([]) errors.)
    pos = min([low.find(p) for p in refusal_phrases if p in low] or [-1])
    # Refusal fast path: phrase found, within the first 100 characters, and the whole answer under 200 characters.
    if pos != -1 and pos < 100 and len(answer.strip()) < 200:
        return {"grade": "REFUSED", "why": "explicit refusal"}

    # Otherwise ask the judge model. Temperature 0 (set in main) for repeatable grades. The answer is cut to 2500
    # characters so a runaway answer cannot blow up the prompt.
    raw = judge_llm.invoke(JUDGE_PROMPT.format(
        question=question, reference=reference, answer=answer[:2500])).content
    # Strip Markdown code fences the judge sometimes adds. Regex piece by piece:
    #   ^```(?:json)?  = at the start of a line, three backticks, optionally followed by "json" ((?:...) groups without saving)
    #   |              = OR
    #   ```$           = three backticks at the end of a line.
    # re.MULTILINE makes ^ and $ match at every line, not only at the start/end of the whole text.
    raw = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.MULTILINE)
    # Find the JSON object: from the first "{" to the last "}".
    start, end = raw.find("{"), raw.rfind("}")
    # Both braces found -> try to parse that slice as JSON.
    if start != -1 and end != -1:
        try:
            # Accept the result only if the grade is one of the 4 allowed words.
            parsed = json.loads(raw[start:end + 1])
            if parsed.get("grade") in ("CORRECT", "PARTIAL", "INCORRECT", "REFUSED"):
                return parsed
        # Broken JSON -> fall through to the default below.
        except json.JSONDecodeError:
            pass
    # Anything unreadable counts as INCORRECT (the safe, pessimistic choice).
    return {"grade": "INCORRECT", "why": "unparseable judge output"}


# ── Driver ───────────────────────────────────────────────────────────────────

# IN: one result row + its question  ->  OUT: the same row with retrieval_hit, retrieval_hit_lenient, hit_rank added.
# WHY: scoring is done from cached rows, so changing the recall rule re-scores old results with zero API calls.
# Example: q001 with gt_page 4, acceptable [4, 65], retrieved_pages [65, 12, 4, 30]
#   -> retrieval_hit True, retrieval_hit_lenient True, hit_rank 3.
def score_row(row, q):
    """
    Derive retrieval metrics from a result row. Kept separate from generation so
    metric definitions can change without invalidating cached API calls.

    strict  -- credit only the page the question was written from
    lenient -- credit any page that genuinely contains the answer
    """
    # The page indexes of the retrieved chunks, best first.
    pages = row.get("retrieved_pages", [])
    # Unanswerable question (a hallucination probe) -> there is no right page, so retrieval is not scored (None).
    if not q["answerable"]:
        row["retrieval_hit"] = None
        row["retrieval_hit_lenient"] = None
        row["hit_rank"] = None
        return row

    # Pages that genuinely contain the answer; if the question has none recorded, use only its source page.
    acceptable = q.get("acceptable_pages") or [q["gt_page"]]
    # strict hit = source page retrieved; lenient hit = any acceptable page retrieved;
    # hit_rank = 1-based position of the source page in the list (None if missed). Used for MRR.
    row["retrieval_hit"] = q["gt_page"] in pages
    row["retrieval_hit_lenient"] = any(p in acceptable for p in pages)
    row["hit_rank"] = (pages.index(q["gt_page"]) + 1) if row["retrieval_hit"] else None
    return row


# IN: config name + settings, question list, answer model, judge model, cached_only flag  ->  OUT: list of scored rows.
# WHY: runs one configuration over every question, reusing cached rows and retrying rate limits.
# Example: run_config("baseline", CONFIGS["baseline"], 48 questions, ...) -> 17 rows when only 17 have results
#   (the easy-set baseline coverage reported in eval/README.md).
def run_config(name, cfg, questions, llm, judge_llm, cached_only=False):
    # Print a banner: a line of 74 "=" characters, the config name and settings, another line.
    print("\n%s\n  CONFIG: %s  %s\n%s" % ("=" * 74, name, cfg, "=" * 74))

    # retrievers = one Retriever per document, built on first use. The three lists record WHY questions are missing.
    retrievers = {}
    results = []
    skipped = []        # index genuinely missing
    not_cached = []     # --cached-only, and this question has no cached result
    gave_up = []        # retries exhausted (rate limits)

    # Loop over questions; enumerate(..., 1) counts from 1 for the "[ 3/48]" progress display.
    for i, q in enumerate(questions, 1):
        # Cache key = config name + question id + the config settings (sorted, so key order never changes the text).
        # Its sha256 fingerprint, cut to 16 hex characters, is the cache file name.
        cache_key = "%s::%s::%s" % (name, q["id"], json.dumps(cfg, sort_keys=True))
        cache_file = RUN_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".json")
        # Cached -> reuse the saved row (no API calls), score it again, and move on.
        if cache_file.exists():
            # The cache holds what cost money (retrieved pages, the generated
            # answer, the grade). Metrics are recomputed from it every run, so
            # a change to how recall is scored applies to old results without
            # re-spending a single API call.
            row = json.loads(cache_file.read_text(encoding="utf-8"))
            results.append(score_row(row, q))
            print("  [%2d/%d] %s cached" % (i, len(questions), q["id"]))
            continue

        # --cached-only and no cached row -> note it and skip (no API calls allowed in this mode).
        if cached_only:
            not_cached.append(q["id"])
            continue

        # First question of this document -> build its retriever. Index missing -> store None so we do not try again.
        if q["doc_id"] not in retrievers:
            try:
                retrievers[q["doc_id"]] = Retriever(q["doc_id"], cfg)
            except FileNotFoundError as e:
                retrievers[q["doc_id"]] = None
                print("  SKIPPING all '%s' questions: %s" % (q["doc_id"], e))
        # Look up the retriever; None means this document's index is missing -> skip the question.
        retriever = retrievers[q["doc_id"]]
        if retriever is None:
            skipped.append(q["id"])
            continue

        # Retry with exponential backoff rather than skipping. A rate limit is a
        # "come back shortly", not a verdict on the question -- an earlier version
        # slept 8s and skipped, silently dropping 13 questions from a run and
        # leaving the summary to average over whatever survived.
        # Start empty. delay = 20 seconds before the first retry.
        retrieved = gen = verdict = None
        delay = 20.0
        # Up to 5 attempts at the full retrieve -> generate -> judge step.
        for attempt in range(5):
            # Success: all three steps worked; pause 1.2 s (to be gentle with the free tier) and leave the loop.
            try:
                retrieved = retriever.retrieve(q["question"])
                gen = generate_answer(llm, q["question"], retrieved)
                verdict = judge(judge_llm, q["question"],
                                q["reference_answer"], gen["answer"])
                time.sleep(1.2)                   # free-tier courtesy
                break
            # Failure: find out if it was a rate limit (Google says RESOURCE_EXHAUSTED, Groq gives HTTP 429 = "too many requests").
            except Exception as e:
                msg = str(e)
                rate = "RESOURCE_EXHAUSTED" in msg or "429" in msg
                # Print what happened. The ternaries print "rate limited" for a rate limit, else "ERROR" plus the exception's class name.
                print("  [%2d/%d] %s %s %s -- waiting %.0fs (attempt %d/5)"
                      % (i, len(questions), q["id"],
                         "rate limited" if rate else "ERROR",
                         "" if rate else type(e).__name__, delay, attempt + 1))
                # Wait, then grow the wait by 1.8x: 20, 36, 64.8, 116.6, 209.9 seconds (exponential backoff).
                time.sleep(delay)
                delay *= 1.8
        # All 5 attempts failed -> record it as "gave up" (reported honestly in the coverage lines) and continue.
        if verdict is None:
            print("  [%2d/%d] %s GIVING UP after 5 attempts" % (i, len(questions), q["id"]))
            gave_up.append(q["id"])
            continue
        # The page index of each retrieved chunk, best first.
        pages = [d["page"] for d in retrieved]

        # One result row. Text is cut to sensible sizes (800 chars per chunk, 1200 for the answer) to keep the cache small.
        row = {
            "id": q["id"], "doc_id": q["doc_id"], "answerable": q["answerable"],
            "fact_type": q["fact_type"], "gt_page": q["gt_page"],
            "retrieved_pages": pages,
            # Store the retrieved TEXT, not just page numbers. A page can hold
            # several chunks, so page numbers alone cannot reconstruct the exact
            # context a failure saw -- which makes failures unreproducible.
            "retrieved_chunks": [d["text"][:800] for d in retrieved],
            "grade": verdict["grade"], "judge_why": verdict.get("why", ""),
            "latency_s": gen["latency_s"], "finish_reason": gen["finish_reason"],
            "completion_tokens": gen["completion_tokens"],
            "answer": gen["answer"][:1200],
        }
        # Save the row to the cache BEFORE scoring, so a crash later never loses a paid-for answer.
        cache_file.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        row = score_row(row, q)
        results.append(row)

        # Progress line: HIT (source page found), MISS (not found) or n/a (unanswerable question, hit is None).
        # "%-9s" pads the grade to 9 characters so the columns line up.
        hit = row["retrieval_hit"]
        flag = "HIT " if hit else ("MISS" if hit is False else "n/a ")
        print("  [%2d/%d] %s  retr=%s  grade=%-9s %.1fs"
              % (i, len(questions), q["id"], flag, verdict["grade"], gen["latency_s"]))

    # Report each reason separately. An earlier version labelled every omission
    # "skipped for missing indexes", which reported 32 missing indexes when all
    # nine were present and complete -- the questions simply had no cached result.
    # A coverage figure is only meaningful when the reason for the gap is honest.
    # Coverage = how many questions actually got a scored row, out of all of them.
    done = len(results)
    total = len(questions)
    # Fewer than all -> print the coverage and each separate reason. [:8] shows at most 8 ids, then "...".
    if done < total:
        print("\n  COVERAGE: %d/%d questions scored (%.0f%%)"
              % (done, total, 100.0 * done / total))
        # Reason 1: no cached result and --cached-only was used.
        if not_cached:
            print("    %d not yet run (--cached-only): %s"
                  % (len(not_cached), ", ".join(not_cached[:8])
                     + ("..." if len(not_cached) > 8 else "")))
        # Reason 2: the index for that document is missing.
        if skipped:
            print("    %d skipped, index missing: %s"
                  % (len(skipped), ", ".join(skipped[:8])
                     + ("..." if len(skipped) > 8 else "")))
        # Reason 3: still failing after 5 retries (rate limits).
        if gave_up:
            print("    %d abandoned after retries (rate limits): %s"
                  % (len(gave_up), ", ".join(gave_up)))
        print("    Cached answers are reused, so a re-run only pays for these.")

    return results


# IN: config label, its scored rows, total question count  ->  OUT: one summary dict of percentages.
# WHY: turns many rows into the numbers in summary.json and the README tables.
# Example (from eval/README.md): "baseline" easy -> n_scored 17, n_total 48, coverage_pct 35.4, recall_at_k 100.0,
#   mrr 0.772, correct_pct 100.0, hallucination_pct 0.0, mean_latency_s 1.51.
def summarize(name, results, n_total=None):
    # Split rows into answerable questions and unanswerable ones (hallucination probes).
    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]

    # Each metric = count of matching rows / number of rows; "if ans else 0.0" avoids dividing by zero.
    # recall = strict hit rate; recall_len = lenient hit rate; correct/partial/refused = grade shares on answerable questions.
    recall = sum(1 for r in ans if r["retrieval_hit"]) / len(ans) if ans else 0.0
    recall_len = sum(1 for r in ans if r["retrieval_hit_lenient"]) / len(ans) if ans else 0.0
    correct = sum(1 for r in ans if r["grade"] == "CORRECT") / len(ans) if ans else 0.0
    partial = sum(1 for r in ans if r["grade"] == "PARTIAL") / len(ans) if ans else 0.0
    refused_ans = sum(1 for r in ans if r["grade"] == "REFUSED") / len(ans) if ans else 0.0
    # On unanswerable questions, refusing is the CORRECT behaviour.
    halluc = (sum(1 for r in unans if r["grade"] != "REFUSED") / len(unans)) if unans else 0.0
    # trunc = share of ALL answers that stopped because of the token limit; lat = average seconds per answer.
    trunc = sum(1 for r in results if r["finish_reason"] == "length") / len(results) if results else 0.0
    lat = sum(r["latency_s"] for r in results) / len(results) if results else 0.0

    # MRR: 1/rank for every answerable question that had a hit, summed, divided by ALL answerable questions
    # (so a miss adds 0). Example: ranks 1, 2 and one miss -> (1 + 0.5 + 0) / 3 = 0.5.
    ranks = [r["hit_rank"] for r in ans if r["hit_rank"]]
    mrr = sum(1.0 / r for r in ranks) / len(ans) if ans else 0.0

    # Build the summary. Fractions become percentages rounded to 1 decimal place; MRR to 3 decimals.
    return {
        "config": name, "n_answerable": len(ans), "n_unanswerable": len(unans),
        # Coverage travels WITH the metrics. A 100% recall over 10 of 42
        # questions is not the same claim as 100% over all 42, and the summary
        # must not let those two look alike.
        "n_scored": len(results),
        "n_total": n_total if n_total is not None else len(results),
        "coverage_pct": round(100.0 * len(results) / n_total, 1) if n_total else 100.0,
        "recall_at_k": round(recall * 100, 1),
        "recall_at_k_lenient": round(recall_len * 100, 1),
        "mrr": round(mrr, 3),
        "correct_pct": round(correct * 100, 1), "partial_pct": round(partial * 100, 1),
        "refused_on_answerable_pct": round(refused_ans * 100, 1),
        "hallucination_pct": round(halluc * 100, 1),
        "truncated_pct": round(trunc * 100, 1), "mean_latency_s": round(lat, 2),
    }


# IN: command-line arguments  ->  OUT: nothing; prints a results table and writes results/summary.json + raw_*.json.
# WHY: the entry point. Examples: py -3.12 eval/run_eval.py baseline ; py -3.12 eval/run_eval.py --hard ;
#      py -3.12 eval/run_eval.py --cached-only (re-score from cache, zero API calls).
def main():
    # --hard scores the paraphrased set instead of the original. Results are
    # labelled "<config>@hard" so the two sets never overwrite each other in
    # summary.json -- they are different tests and are not comparable.
    # Pick the question file: qa_set_hard.json with --hard, else qa_set.json.
    hard = "--hard" in sys.argv
    qa_path = config.QA_SET_HARD_PATH if hard else config.QA_SET_PATH
    suffix = "@hard" if hard else ""

    # No question file -> stop with a hint.
    if not qa_path.exists():
        raise SystemExit("Missing %s -- generate it first." % qa_path.name)
    # Load the questions and print how many are answerable vs unanswerable.
    questions = json.loads(qa_path.read_text(encoding="utf-8"))
    print("Loaded %d questions from %s (%d answerable, %d unanswerable)"
          % (len(questions), qa_path.name,
             sum(1 for q in questions if q["answerable"]),
             sum(1 for q in questions if not q["answerable"])))

    # Config names = every argument that does not start with "-"; none given -> all six configs.
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or list(config.CONFIGS)
    # Reject a typo'd config name up front, listing the valid options.
    for w in wanted:
        if w not in config.CONFIGS:
            raise SystemExit("Unknown config %r. Options: %s" % (w, ", ".join(config.CONFIGS)))

    # --cached-only re-scores existing results without making any API call.
    # Metrics are derived from the cache on every run, so a change to how recall
    # is defined can be applied to past results for free.
    cached_only = "--cached-only" in sys.argv
    # Cached-only mode: print a notice.
    if cached_only:
        print("--cached-only: re-scoring from cache, no API calls will be made.")

    # Create the two models, unless cached-only (then no model is needed). The judge uses temperature 0 so the same
    # answer always gets the same grade, and JUDGE_MAX_TOKENS = 512.
    llm = None if cached_only else make_llm()
    judge_llm = None if cached_only else ChatGroq(
        model=config.JUDGE_MODEL, temperature=0, max_tokens=config.JUDGE_MAX_TOKENS)

    # Run each requested config and summarize it.
    summaries = []
    for name in wanted:
        label = name + suffix
        results = run_config(label, config.CONFIGS[name], questions, llm, judge_llm,
                             cached_only=cached_only)
        # Nothing scored for this config -> say so and move on.
        if not results:
            print("  (no cached results for %s)" % label)
            continue
        # Save the raw rows as results/raw_<label>.json ("@" replaced by "_", e.g. raw_baseline_hard.json).
        (config.RESULTS_DIR / ("raw_%s.json" % label.replace("@", "_"))).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        # Summarize and print the headline numbers for this config.
        s = summarize(label, results, n_total=len(questions))
        summaries.append(s)
        print("\n  -> recall@k %.1f%% | correct %.1f%% | halluc %.1f%% | trunc %.1f%%"
              % (s["recall_at_k"], s["correct_pct"], s["hallucination_pct"], s["truncated_pct"]))

    # Merge with the existing summary.json: start from the old entries, then overwrite those configs run just now,
    # so running one config never deletes the others' results.
    out = config.RESULTS_DIR / "summary.json"
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    merged = {s["config"]: s for s in existing}
    merged.update({s["config"]: s for s in summaries})
    out.write_text(json.dumps(list(merged.values()), indent=2), encoding="utf-8")

    # Print the final table: one row per config. "%-16s", "%9s" etc. set column widths so it lines up.
    print("\n%s\n%-16s %9s %8s %8s %6s %8s %8s %7s" % ("=" * 96, "CONFIG",
          "COVERAGE", "RECALL", "RECALL+", "MRR", "CORRECT", "HALLUC", "TRUNC"))
    print("-" * 96)
    # One table row per config; a " *" flag marks a partial run (coverage under 100%).
    for s in merged.values():
        cov = "%d/%d" % (s.get("n_scored", 0), s.get("n_total", 0))
        flag = " *" if s.get("coverage_pct", 100) < 100 else ""
        print("%-16s %9s %7.1f%% %7.1f%% %6.3f %7.1f%% %7.1f%% %6.1f%%%s"
              % (s["config"], cov, s["recall_at_k"], s.get("recall_at_k_lenient", 0),
                 s["mrr"], s["correct_pct"], s["hallucination_pct"],
                 s["truncated_pct"], flag))
    print("=" * 96)
    print("*       = PARTIAL run -- not comparable to a complete one")
    print("RECALL  = strict: only the page the question was written from counts")
    print("RECALL+ = lenient: any page genuinely containing the answer counts")
    print("\nRaw per-question results: %s" % config.RESULTS_DIR)


# Run main() only when started directly (not when imported).
if __name__ == "__main__":
    main()
