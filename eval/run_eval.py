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
# WHAT THIS FILE IS: asks every test question, a bigger model grades each answer, then it scores the configs.
# Example: q001 "What SQL command is shown for creating the example database?" -> retrieve 4 chunks
#   -> gpt-oss-20b answers "CREATE DATABASE startersql;" -> gpt-oss-120b grades CORRECT -> cached in eval/cache/runs/
# Flow: questions -> each config -> each question: cached? reuse : retrieve -> generate -> judge -> save
#   -> score_row -> summarize -> merge into summary.json -> print table
# RECALL@k = share of answerable Qs with a right-page chunk in the top k.
# MRR = average 1/rank of the first right chunk (rank 1 -> 1.0, 2 -> 0.5, miss -> 0).
# Hallucination = inventing an answer not in the doc; truncation = cut off at max tokens ("length").
import json  # question sets, cache + result files
import re  # regex: split words, strip ``` fences
import sys  # CLI args (config names, --hard, --cached-only) + import path
import time  # answer latency + sleep between retries
import hashlib  # sha256 fingerprint = cache file name
import pathlib  # file paths
from collections import defaultdict  # defaultdict(float): new keys start at 0.0, for RRF sums

sys.path.insert(0, str(pathlib.Path(__file__).parent))  # so "import config" finds eval/config.py
import config  # shared eval settings

config.load_env()  # API keys from .env (config also put project root on path)

from langchain_chroma import Chroma  # opens stores saved by build_index.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # embeds the QUESTION, same model as chunks
from langchain_groq import ChatGroq  # Groq client: gpt-oss-20b answers, 120b judges
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # fill-in prompt + chat-history slot

# Import the LIVE app's prompt and formatter so these numbers describe the real
# system. Importing is read-only; documind/ is never modified.
from documind import SYSTEM_PROMPT, format_docs  # the app's real prompt + chunk formatter

RUN_CACHE = config.CACHE_DIR / "runs"  # one JSON per (config, question)
RUN_CACHE.mkdir(parents=True, exist_ok=True)  # create if missing

# RRF = Reciprocal Rank Fusion: each chunk gets 1/(60 + rank) per list, totals decide order.
RRF_K = 60          # reciprocal-rank-fusion constant, standard value


# ── Retrieval ────────────────────────────────────────────────────────────────

# IN: doc id, config -> OUT: object whose retrieve(question) gives best k chunks
# e.g. Retriever("mysql", CONFIGS["baseline"]).retrieve("What SQL command ...") -> 4 {"text", "page"} dicts
class Retriever:
    """Dense (Chroma) or hybrid (BM25 + dense, fused with RRF)."""

    # IN: doc id, config -> OUT: ready retriever (store open, chunks loaded, BM25 if hybrid)
    # e.g. "bi" + baseline -> opens eval/chroma_eval/bi_cs1000_co200 + its _chunks.json
    def __init__(self, doc_id, cfg):
        # ── Step 1: find this config's index ──
        key = config.index_key(doc_id, cfg["chunk_size"], cfg["chunk_overlap"])  # e.g. "bi_cs1000_co200"
        persist_dir = config.CHROMA_EVAL_DIR / key  # vectors folder
        chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")  # chunk texts
        if not persist_dir.exists() or not chunks_path.exists():  # not built yet
            # Raised (not SystemExit) so a partial run can skip these questions and
            # pick them up later: results are cached per question, so a second run
            # after the index finishes only pays for what is still missing.
            raise FileNotFoundError(
                "index %s not built yet -- run eval/build_index.py" % key)  # run_config catches it

        # ── Step 2: open the store + chunk texts ──
        self.k = cfg["k"]  # chunks to return, e.g. 4
        self.mode = cfg["retrieval"]  # "dense" or "hybrid"
        self.store = Chroma(  # saved vector store
            persist_directory=str(persist_dir),  # its folder
            embedding_function=GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL),  # same embedder
        )
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))  # plain texts, for BM25

        # ── Step 3 (hybrid only): BM25 keyword index ──
        # BM25 = classic keyword scoring, rare words count more, repeats saturate. No AI, no API.
        if self.mode == "hybrid":  # dense runs skip this
            from rank_bm25 import BM25Okapi  # rank-bm25 package, only needed here
            corpus = [self._tokenize(c["text"]) for c in self.chunks]  # each chunk as words
            self.bm25 = BM25Okapi(corpus)  # build the index

    # IN: text -> OUT: lowercase words
    # e.g. "CREATE DATABASE startersql;" -> ["create", "database", "startersql"]
    @staticmethod  # no self needed
    def _tokenize(text):
        return re.findall(r"[a-z0-9_]+", text.lower())  # runs of letters/digits/_

    # IN: question -> OUT: k {text, page} dicts, best first (the "R" in RAG)
    # e.g. k=4 dense -> 4 chunks with embeddings closest to the question
    def retrieve(self, question):
        """Return a list of {text, page} dicts, best first."""
        # ── Step 1: dense search ──
        dense_docs = self.store.similarity_search(question, k=self.k * 2)  # 8 when k=4, extras for fusion
        dense = [{"text": d.page_content, "page": d.metadata.get("page", 0)}  # to simple dicts
                 for d in dense_docs]

        if self.mode == "dense":  # dense mode
            return dense[:self.k]  # top k, like the live app

        # ── Step 2: BM25 search ──
        # Hybrid: fuse dense and BM25 rankings with reciprocal rank fusion.
        scores = self.bm25.get_scores(self._tokenize(question))  # score every chunk
        top_bm25_idx = sorted(range(len(scores)), key=lambda i: scores[i],  # chunk numbers by score
                              reverse=True)[:self.k * 2]  # biggest first, keep k*2
        bm25 = [{"text": self.chunks[i]["text"], "page": self.chunks[i]["page"]}  # to dicts
                for i in top_bm25_idx]

        # ── Step 3: RRF fusion ──
        fused = defaultdict(float)  # total score per chunk
        holder = {}  # chunk dict per key
        # sig = first 200 chars, the chunk's ID so both lists add to one entry.
        for rank, d in enumerate(dense):  # rank from 0
            sig = d["text"][:200]  # chunk ID
            fused[sig] += 1.0 / (RRF_K + rank + 1)  # top chunk -> 1/61
            holder[sig] = d  # remember the chunk
        for rank, d in enumerate(bm25):  # same for BM25
            sig = d["text"][:200]
            fused[sig] += 1.0 / (RRF_K + rank + 1)
            holder[sig] = d

        ranked = sorted(fused, key=lambda s: fused[s], reverse=True)  # highest total first
        return [holder[s] for s in ranked[:self.k]]  # top k chunks


# ── Generation ───────────────────────────────────────────────────────────────

# IN: nothing -> OUT: answer model gpt-oss-20b, temperature 0.1, max 1024 tokens (same as documind/chain.py)
# Temperature = randomness; 0.1 = stable answers.
def make_llm():
    return ChatGroq(model=config.GEN_MODEL, temperature=config.GEN_TEMPERATURE,
                    max_tokens=config.GEN_MAX_TOKENS)  # app's own settings


# IN: model, question, retrieved chunks -> OUT: {"answer", "latency_s", "finish_reason", "completion_tokens"}
# e.g. {"answer": "...startersql...", "latency_s": 0.68, "finish_reason": "stop", ...} ("length" = cut off)
def generate_answer(llm, question, retrieved):
    """Run the app's real prompt over the retrieved context."""
    class _D:                       # format_docs() expects .page_content/.metadata
        def __init__(self, d):  # wrap one chunk dict
            self.page_content = d["text"]  # chunk text
            self.metadata = {"page": d["page"]}  # its page

    # ── Step 1: build the app's exact prompt ──
    context = format_docs([_D(d) for d in retrieved])  # "[Page N]\n<text>" blocks joined by "---"
    prompt = ChatPromptTemplate.from_messages([  # same layout as the live app
        ("system", SYSTEM_PROMPT),  # app rules + context
        MessagesPlaceholder(variable_name="chat_history"),  # earlier chat slot
        ("human", "{question}"),  # the question
    ])
    msgs = prompt.format_messages(context=context, chat_history=[], question=question)  # fresh, no history

    # ── Step 2: call the model, time it ──
    t0 = time.time()  # start stopwatch
    resp = llm.invoke(msgs)  # ask the model
    latency = time.time() - t0  # seconds taken
    meta = resp.response_metadata or {}  # finish_reason + token_usage; {} if None
    return {
        "answer": resp.content,  # answer text
        "latency_s": round(latency, 2),  # e.g. 0.68
        "finish_reason": meta.get("finish_reason", "?"),  # "?" if not reported
        "completion_tokens": (meta.get("token_usage") or {}).get("completion_tokens"),  # tokens written
    }


# ── Judging ──────────────────────────────────────────────────────────────────

# {question}, {reference}, {answer} filled by .format(); {{ }} = literal braces.
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


# IN: judge model, question, reference, answer -> OUT: {"grade", "why"}
# e.g. "I could not find this information in the uploaded document." -> REFUSED, no judge call
def judge(judge_llm, question, reference, answer):
    if not answer or not answer.strip():  # blank answer
        return {"grade": "INCORRECT", "why": "empty answer"}

    # ── Step 1: fast path for the exact refusal ──
    # Fast path for the exact refusal the system prompt asks for.
    #
    # Two guards, both added after adversarial testing (see verify_method.py):
    #   - length < 200: the template refusal is 59 chars. A 340-char answer that
    #     happens to contain "could not find this information" mid-sentence is a
    #     real answer, and grading it REFUSED would understate hallucination.
    #   - phrase within the first 100 chars: a refusal leads with it; prose that
    #     merely mentions the phrase does not.
    low = answer.lower()  # ignore capitals
    refusal_phrases = (  # phrases that mean "not found"
        "could not find this information",
        "could not find that information",
        "cannot find this information",
        "i could not find",
        "i cannot find",
    )
    pos = min([low.find(p) for p in refusal_phrases if p in low] or [-1])  # earliest phrase, -1 if none
    if pos != -1 and pos < 100 and len(answer.strip()) < 200:  # found early in a short answer
        return {"grade": "REFUSED", "why": "explicit refusal"}

    # ── Step 2: ask the judge model ──
    raw = judge_llm.invoke(JUDGE_PROMPT.format(  # temperature 0, set in main
        question=question, reference=reference, answer=answer[:2500])).content  # cap answer at 2500 chars
    # Regex strips ```json / ``` fences; MULTILINE = ^ and $ match every line.
    raw = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.MULTILINE)
    start, end = raw.find("{"), raw.rfind("}")  # first "{" to last "}"
    if start != -1 and end != -1:  # both braces found
        try:
            parsed = json.loads(raw[start:end + 1])  # parse the slice
            if parsed.get("grade") in ("CORRECT", "PARTIAL", "INCORRECT", "REFUSED"):  # allowed grade?
                return parsed
        except json.JSONDecodeError:  # broken JSON -> default below
            pass
    return {"grade": "INCORRECT", "why": "unparseable judge output"}  # unreadable = safe pessimistic grade


# ── Driver ───────────────────────────────────────────────────────────────────

# IN: result row, its question -> OUT: row + retrieval_hit, retrieval_hit_lenient, hit_rank
# e.g. gt_page 4, acceptable [4, 65], retrieved [65, 12, 4, 30] -> True, True, hit_rank 3
def score_row(row, q):
    """
    Derive retrieval metrics from a result row. Kept separate from generation so
    metric definitions can change without invalidating cached API calls.

    strict  -- credit only the page the question was written from
    lenient -- credit any page that genuinely contains the answer
    """
    pages = row.get("retrieved_pages", [])  # retrieved page indexes, best first
    if not q["answerable"]:  # hallucination probe: no right page
        row["retrieval_hit"] = None  # not scored
        row["retrieval_hit_lenient"] = None
        row["hit_rank"] = None
        return row

    acceptable = q.get("acceptable_pages") or [q["gt_page"]]  # pages holding the answer, else source page
    row["retrieval_hit"] = q["gt_page"] in pages  # strict: source page found
    row["retrieval_hit_lenient"] = any(p in acceptable for p in pages)  # lenient: any acceptable page
    row["hit_rank"] = (pages.index(q["gt_page"]) + 1) if row["retrieval_hit"] else None  # 1-based, for MRR
    return row


# IN: config name + settings, questions, models, cached_only -> OUT: scored rows
# e.g. run_config("baseline", CONFIGS["baseline"], 48 questions, ...) -> 17 rows when only 17 have results
def run_config(name, cfg, questions, llm, judge_llm, cached_only=False):
    print("\n%s\n  CONFIG: %s  %s\n%s" % ("=" * 74, name, cfg, "=" * 74))  # banner

    retrievers = {}  # one Retriever per doc, built on first use
    results = []  # scored rows
    skipped = []        # index genuinely missing
    not_cached = []     # --cached-only, and this question has no cached result
    gave_up = []        # retries exhausted (rate limits)

    for i, q in enumerate(questions, 1):  # count from 1 for "[ 3/48]"
        # ── Step 1: cached? reuse it ──
        cache_key = "%s::%s::%s" % (name, q["id"], json.dumps(cfg, sort_keys=True))  # config + question + settings
        cache_file = RUN_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".json")  # 16-hex name
        if cache_file.exists():  # already paid for
            # The cache holds what cost money (retrieved pages, the generated
            # answer, the grade). Metrics are recomputed from it every run, so
            # a change to how recall is scored applies to old results without
            # re-spending a single API call.
            row = json.loads(cache_file.read_text(encoding="utf-8"))  # saved row
            results.append(score_row(row, q))  # re-score, no API
            print("  [%2d/%d] %s cached" % (i, len(questions), q["id"]))
            continue

        if cached_only:  # no API calls allowed
            not_cached.append(q["id"])  # note and skip
            continue

        # ── Step 2: get this document's retriever ──
        if q["doc_id"] not in retrievers:  # first question of this doc
            try:
                retrievers[q["doc_id"]] = Retriever(q["doc_id"], cfg)  # build it
            except FileNotFoundError as e:  # index missing
                retrievers[q["doc_id"]] = None  # don't try again
                print("  SKIPPING all '%s' questions: %s" % (q["doc_id"], e))
        retriever = retrievers[q["doc_id"]]  # look it up
        if retriever is None:  # no index
            skipped.append(q["id"])  # skip question
            continue

        # ── Step 3: retrieve -> generate -> judge, with retries ──
        # Retry with exponential backoff rather than skipping. A rate limit is a
        # "come back shortly", not a verdict on the question -- an earlier version
        # slept 8s and skipped, silently dropping 13 questions from a run and
        # leaving the summary to average over whatever survived.
        retrieved = gen = verdict = None  # start empty
        delay = 20.0  # 20 s before first retry
        for attempt in range(5):  # up to 5 tries
            try:
                retrieved = retriever.retrieve(q["question"])  # top k chunks
                gen = generate_answer(llm, q["question"], retrieved)  # model's answer
                verdict = judge(judge_llm, q["question"],
                                q["reference_answer"], gen["answer"])  # grade it
                time.sleep(1.2)                   # free-tier courtesy
                break  # success
            except Exception as e:  # something failed
                msg = str(e)  # error text
                rate = "RESOURCE_EXHAUSTED" in msg or "429" in msg  # Google / Groq rate limit
                print("  [%2d/%d] %s %s %s -- waiting %.0fs (attempt %d/5)"
                      % (i, len(questions), q["id"],
                         "rate limited" if rate else "ERROR",  # which kind
                         "" if rate else type(e).__name__, delay, attempt + 1))  # error class name
                time.sleep(delay)  # wait
                delay *= 1.8  # 20, 36, 64.8, 116.6, 209.9 s
        if verdict is None:  # all 5 failed
            print("  [%2d/%d] %s GIVING UP after 5 attempts" % (i, len(questions), q["id"]))
            gave_up.append(q["id"])  # reported in coverage lines
            continue
        pages = [d["page"] for d in retrieved]  # page of each chunk, best first

        # ── Step 4: build, cache and score the row ──
        row = {
            "id": q["id"], "doc_id": q["doc_id"], "answerable": q["answerable"],  # question info
            "fact_type": q["fact_type"], "gt_page": q["gt_page"],  # gt_page = source page
            "retrieved_pages": pages,  # e.g. [4, 12, 65, 30]
            # Store the retrieved TEXT, not just page numbers. A page can hold
            # several chunks, so page numbers alone cannot reconstruct the exact
            # context a failure saw -- which makes failures unreproducible.
            "retrieved_chunks": [d["text"][:800] for d in retrieved],  # 800 chars each
            "grade": verdict["grade"], "judge_why": verdict.get("why", ""),  # judge result
            "latency_s": gen["latency_s"], "finish_reason": gen["finish_reason"],  # timing + stop reason
            "completion_tokens": gen["completion_tokens"],  # tokens written
            "answer": gen["answer"][:1200],  # first 1200 chars
        }
        cache_file.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")  # save BEFORE scoring
        row = score_row(row, q)  # add hit metrics
        results.append(row)  # keep it

        hit = row["retrieval_hit"]  # True / False / None
        flag = "HIT " if hit else ("MISS" if hit is False else "n/a ")  # None = unanswerable
        print("  [%2d/%d] %s  retr=%s  grade=%-9s %.1fs"
              % (i, len(questions), q["id"], flag, verdict["grade"], gen["latency_s"]))  # progress line

    # ── Step 5: honest coverage report ──
    # Report each reason separately. An earlier version labelled every omission
    # "skipped for missing indexes", which reported 32 missing indexes when all
    # nine were present and complete -- the questions simply had no cached result.
    # A coverage figure is only meaningful when the reason for the gap is honest.
    done = len(results)  # scored questions
    total = len(questions)  # all questions
    if done < total:  # some missing
        print("\n  COVERAGE: %d/%d questions scored (%.0f%%)"
              % (done, total, 100.0 * done / total))
        if not_cached:  # reason 1: --cached-only, no cache
            print("    %d not yet run (--cached-only): %s"
                  % (len(not_cached), ", ".join(not_cached[:8])  # max 8 ids
                     + ("..." if len(not_cached) > 8 else "")))
        if skipped:  # reason 2: index missing
            print("    %d skipped, index missing: %s"
                  % (len(skipped), ", ".join(skipped[:8])
                     + ("..." if len(skipped) > 8 else "")))
        if gave_up:  # reason 3: retries exhausted
            print("    %d abandoned after retries (rate limits): %s"
                  % (len(gave_up), ", ".join(gave_up)))
        print("    Cached answers are reused, so a re-run only pays for these.")

    return results


# IN: config label, scored rows, total questions -> OUT: summary dict of percentages
# e.g. "baseline" easy -> n_scored 17/48, recall_at_k 100.0, mrr 0.772, correct_pct 100.0, hallucination_pct 0.0
def summarize(name, results, n_total=None):
    ans = [r for r in results if r["answerable"]]  # answerable questions
    unans = [r for r in results if not r["answerable"]]  # hallucination probes

    # ── Step 1: rates (count / rows; "if ans else 0.0" avoids divide by zero) ──
    recall = sum(1 for r in ans if r["retrieval_hit"]) / len(ans) if ans else 0.0  # strict hit rate
    recall_len = sum(1 for r in ans if r["retrieval_hit_lenient"]) / len(ans) if ans else 0.0  # lenient hit rate
    correct = sum(1 for r in ans if r["grade"] == "CORRECT") / len(ans) if ans else 0.0  # share CORRECT
    partial = sum(1 for r in ans if r["grade"] == "PARTIAL") / len(ans) if ans else 0.0  # share PARTIAL
    refused_ans = sum(1 for r in ans if r["grade"] == "REFUSED") / len(ans) if ans else 0.0  # refused a real Q
    halluc = (sum(1 for r in unans if r["grade"] != "REFUSED") / len(unans)) if unans else 0.0  # probe not refused
    trunc = sum(1 for r in results if r["finish_reason"] == "length") / len(results) if results else 0.0  # hit token cap
    lat = sum(r["latency_s"] for r in results) / len(results) if results else 0.0  # mean seconds

    # ── Step 2: MRR, misses add 0 ──
    ranks = [r["hit_rank"] for r in ans if r["hit_rank"]]  # ranks of hits only
    mrr = sum(1.0 / r for r in ranks) / len(ans) if ans else 0.0  # e.g. ranks 1, 2, miss -> 0.5

    # ── Step 3: summary dict (percent to 1 decimal, MRR to 3) ──
    return {
        "config": name, "n_answerable": len(ans), "n_unanswerable": len(unans),  # counts
        # Coverage travels WITH the metrics. A 100% recall over 10 of 42
        # questions is not the same claim as 100% over all 42, and the summary
        # must not let those two look alike.
        "n_scored": len(results),  # rows scored
        "n_total": n_total if n_total is not None else len(results),  # all questions
        "coverage_pct": round(100.0 * len(results) / n_total, 1) if n_total else 100.0,  # e.g. 35.4
        "recall_at_k": round(recall * 100, 1),  # strict recall %
        "recall_at_k_lenient": round(recall_len * 100, 1),  # lenient recall %
        "mrr": round(mrr, 3),  # e.g. 0.772
        "correct_pct": round(correct * 100, 1), "partial_pct": round(partial * 100, 1),  # grade shares
        "refused_on_answerable_pct": round(refused_ans * 100, 1),  # wrong refusals
        "hallucination_pct": round(halluc * 100, 1),  # invented answers
        "truncated_pct": round(trunc * 100, 1), "mean_latency_s": round(lat, 2),  # cut-offs + speed
    }


# IN: CLI args -> OUT: nothing, prints table, writes results/summary.json + raw_*.json
# e.g. py -3.12 eval/run_eval.py baseline | --hard | --cached-only (re-score, zero API calls)
def main():
    # ── Step 1: pick the question set ──
    # --hard scores the paraphrased set instead of the original. Results are
    # labelled "<config>@hard" so the two sets never overwrite each other in
    # summary.json -- they are different tests and are not comparable.
    hard = "--hard" in sys.argv  # hard set wanted?
    qa_path = config.QA_SET_HARD_PATH if hard else config.QA_SET_PATH  # which file
    suffix = "@hard" if hard else ""  # label suffix

    if not qa_path.exists():  # file missing
        raise SystemExit("Missing %s -- generate it first." % qa_path.name)
    questions = json.loads(qa_path.read_text(encoding="utf-8"))  # load questions
    print("Loaded %d questions from %s (%d answerable, %d unanswerable)"
          % (len(questions), qa_path.name,
             sum(1 for q in questions if q["answerable"]),  # answerable count
             sum(1 for q in questions if not q["answerable"])))  # probe count

    # ── Step 2: pick configs ──
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or list(config.CONFIGS)  # none given -> all 6
    for w in wanted:  # check each name
        if w not in config.CONFIGS:  # typo -> stop, list options
            raise SystemExit("Unknown config %r. Options: %s" % (w, ", ".join(config.CONFIGS)))

    # --cached-only re-scores existing results without making any API call.
    # Metrics are derived from the cache on every run, so a change to how recall
    # is defined can be applied to past results for free.
    cached_only = "--cached-only" in sys.argv  # no-API mode?
    if cached_only:  # tell the user
        print("--cached-only: re-scoring from cache, no API calls will be made.")

    # ── Step 3: models (none in cached-only) ──
    llm = None if cached_only else make_llm()  # answer model
    judge_llm = None if cached_only else ChatGroq(  # judge model
        model=config.JUDGE_MODEL, temperature=0, max_tokens=config.JUDGE_MAX_TOKENS)  # temp 0 = same grade every time

    # ── Step 4: run + summarize each config ──
    summaries = []  # one dict per config
    for name in wanted:  # each config
        label = name + suffix  # e.g. "baseline@hard"
        results = run_config(label, config.CONFIGS[name], questions, llm, judge_llm,
                             cached_only=cached_only)  # scored rows
        if not results:  # nothing scored
            print("  (no cached results for %s)" % label)
            continue
        # Raw rows -> results/raw_<label>.json, "@" -> "_" (e.g. raw_baseline_hard.json).
        (config.RESULTS_DIR / ("raw_%s.json" % label.replace("@", "_"))).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        s = summarize(label, results, n_total=len(questions))  # headline numbers
        summaries.append(s)  # keep it
        print("\n  -> recall@k %.1f%% | correct %.1f%% | halluc %.1f%% | trunc %.1f%%"
              % (s["recall_at_k"], s["correct_pct"], s["hallucination_pct"], s["truncated_pct"]))

    # ── Step 5: merge into summary.json (other configs kept) ──
    out = config.RESULTS_DIR / "summary.json"  # results file
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []  # old entries
    merged = {s["config"]: s for s in existing}  # keyed by config label
    merged.update({s["config"]: s for s in summaries})  # overwrite ones run now
    out.write_text(json.dumps(list(merged.values()), indent=2), encoding="utf-8")  # save

    # ── Step 6: print the table (%-16s, %9s = column widths) ──
    print("\n%s\n%-16s %9s %8s %8s %6s %8s %8s %7s" % ("=" * 96, "CONFIG",
          "COVERAGE", "RECALL", "RECALL+", "MRR", "CORRECT", "HALLUC", "TRUNC"))  # header
    print("-" * 96)  # divider
    for s in merged.values():  # one row per config
        cov = "%d/%d" % (s.get("n_scored", 0), s.get("n_total", 0))  # e.g. "17/48"
        flag = " *" if s.get("coverage_pct", 100) < 100 else ""  # "*" = partial run
        print("%-16s %9s %7.1f%% %7.1f%% %6.3f %7.1f%% %7.1f%% %6.1f%%%s"
              % (s["config"], cov, s["recall_at_k"], s.get("recall_at_k_lenient", 0),
                 s["mrr"], s["correct_pct"], s["hallucination_pct"],
                 s["truncated_pct"], flag))
    print("=" * 96)  # footer + legend
    print("*       = PARTIAL run -- not comparable to a complete one")
    print("RECALL  = strict: only the page the question was written from counts")
    print("RECALL+ = lenient: any page genuinely containing the answer counts")
    print("\nRaw per-question results: %s" % config.RESULTS_DIR)


if __name__ == "__main__":  # only when run directly
    main()
