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
import json
import re
import sys
import time
import hashlib
import pathlib
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

config.load_env()
sys.path.insert(0, str(config.PROJECT_DIR))

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import the LIVE app's prompt and formatter so these numbers describe the real
# system. Importing is read-only; rag_engine.py is never modified.
from rag_engine import SYSTEM_PROMPT, format_docs

RUN_CACHE = config.CACHE_DIR / "runs"
RUN_CACHE.mkdir(parents=True, exist_ok=True)

RRF_K = 60          # reciprocal-rank-fusion constant, standard value


# ── Retrieval ────────────────────────────────────────────────────────────────

class Retriever:
    """Dense (Chroma) or hybrid (BM25 + dense, fused with RRF)."""

    def __init__(self, doc_id, cfg):
        key = config.index_key(doc_id, cfg["chunk_size"], cfg["chunk_overlap"])
        persist_dir = config.CHROMA_EVAL_DIR / key
        chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")
        if not persist_dir.exists() or not chunks_path.exists():
            # Raised (not SystemExit) so a partial run can skip these questions and
            # pick them up later: results are cached per question, so a second run
            # after the index finishes only pays for what is still missing.
            raise FileNotFoundError(
                "index %s not built yet -- run eval/build_index.py" % key)

        self.k = cfg["k"]
        self.mode = cfg["retrieval"]
        self.store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL),
        )
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

        if self.mode == "hybrid":
            from rank_bm25 import BM25Okapi
            corpus = [self._tokenize(c["text"]) for c in self.chunks]
            self.bm25 = BM25Okapi(corpus)

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9_]+", text.lower())

    def retrieve(self, question):
        """Return a list of {text, page} dicts, best first."""
        dense_docs = self.store.similarity_search(question, k=self.k * 2)
        dense = [{"text": d.page_content, "page": d.metadata.get("page", 0)}
                 for d in dense_docs]

        if self.mode == "dense":
            return dense[:self.k]

        # Hybrid: fuse dense and BM25 rankings with reciprocal rank fusion.
        scores = self.bm25.get_scores(self._tokenize(question))
        top_bm25_idx = sorted(range(len(scores)), key=lambda i: scores[i],
                              reverse=True)[:self.k * 2]
        bm25 = [{"text": self.chunks[i]["text"], "page": self.chunks[i]["page"]}
                for i in top_bm25_idx]

        fused = defaultdict(float)
        holder = {}
        for rank, d in enumerate(dense):
            sig = d["text"][:200]
            fused[sig] += 1.0 / (RRF_K + rank + 1)
            holder[sig] = d
        for rank, d in enumerate(bm25):
            sig = d["text"][:200]
            fused[sig] += 1.0 / (RRF_K + rank + 1)
            holder[sig] = d

        ranked = sorted(fused, key=lambda s: fused[s], reverse=True)
        return [holder[s] for s in ranked[:self.k]]


# ── Generation ───────────────────────────────────────────────────────────────

def make_llm():
    return ChatGroq(model=config.GEN_MODEL, temperature=0.1,
                    max_tokens=config.GEN_MAX_TOKENS)


def generate_answer(llm, question, retrieved):
    """Run the app's real prompt over the retrieved context."""
    class _D:                       # format_docs() expects .page_content/.metadata
        def __init__(self, d):
            self.page_content = d["text"]
            self.metadata = {"page": d["page"]}

    context = format_docs([_D(d) for d in retrieved])
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    msgs = prompt.format_messages(context=context, chat_history=[], question=question)

    t0 = time.time()
    resp = llm.invoke(msgs)
    latency = time.time() - t0
    meta = resp.response_metadata or {}
    return {
        "answer": resp.content,
        "latency_s": round(latency, 2),
        "finish_reason": meta.get("finish_reason", "?"),
        "completion_tokens": (meta.get("token_usage") or {}).get("completion_tokens"),
    }


# ── Judging ──────────────────────────────────────────────────────────────────

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


def judge(judge_llm, question, reference, answer):
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
    low = answer.lower()
    refusal_phrases = (
        "could not find this information",
        "could not find that information",
        "cannot find this information",
        "i could not find",
        "i cannot find",
    )
    pos = min([low.find(p) for p in refusal_phrases if p in low] or [-1])
    if pos != -1 and pos < 100 and len(answer.strip()) < 200:
        return {"grade": "REFUSED", "why": "explicit refusal"}

    raw = judge_llm.invoke(JUDGE_PROMPT.format(
        question=question, reference=reference, answer=answer[:2500])).content
    raw = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.MULTILINE)
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        try:
            parsed = json.loads(raw[start:end + 1])
            if parsed.get("grade") in ("CORRECT", "PARTIAL", "INCORRECT", "REFUSED"):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"grade": "INCORRECT", "why": "unparseable judge output"}


# ── Driver ───────────────────────────────────────────────────────────────────

def score_row(row, q):
    """
    Derive retrieval metrics from a result row. Kept separate from generation so
    metric definitions can change without invalidating cached API calls.

    strict  -- credit only the page the question was written from
    lenient -- credit any page that genuinely contains the answer
    """
    pages = row.get("retrieved_pages", [])
    if not q["answerable"]:
        row["retrieval_hit"] = None
        row["retrieval_hit_lenient"] = None
        row["hit_rank"] = None
        return row

    acceptable = q.get("acceptable_pages") or [q["gt_page"]]
    row["retrieval_hit"] = q["gt_page"] in pages
    row["retrieval_hit_lenient"] = any(p in acceptable for p in pages)
    row["hit_rank"] = (pages.index(q["gt_page"]) + 1) if row["retrieval_hit"] else None
    return row


def run_config(name, cfg, questions, llm, judge_llm, cached_only=False):
    print("\n%s\n  CONFIG: %s  %s\n%s" % ("=" * 74, name, cfg, "=" * 74))

    retrievers = {}
    results = []
    skipped = []        # index genuinely missing
    not_cached = []     # --cached-only, and this question has no cached result
    gave_up = []        # retries exhausted (rate limits)

    for i, q in enumerate(questions, 1):
        cache_key = "%s::%s::%s" % (name, q["id"], json.dumps(cfg, sort_keys=True))
        cache_file = RUN_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".json")
        if cache_file.exists():
            # The cache holds what cost money (retrieved pages, the generated
            # answer, the grade). Metrics are recomputed from it every run, so
            # a change to how recall is scored applies to old results without
            # re-spending a single API call.
            row = json.loads(cache_file.read_text(encoding="utf-8"))
            results.append(score_row(row, q))
            print("  [%2d/%d] %s cached" % (i, len(questions), q["id"]))
            continue

        if cached_only:
            not_cached.append(q["id"])
            continue

        if q["doc_id"] not in retrievers:
            try:
                retrievers[q["doc_id"]] = Retriever(q["doc_id"], cfg)
            except FileNotFoundError as e:
                retrievers[q["doc_id"]] = None
                print("  SKIPPING all '%s' questions: %s" % (q["doc_id"], e))
        retriever = retrievers[q["doc_id"]]
        if retriever is None:
            skipped.append(q["id"])
            continue

        # Retry with exponential backoff rather than skipping. A rate limit is a
        # "come back shortly", not a verdict on the question -- an earlier version
        # slept 8s and skipped, silently dropping 13 questions from a run and
        # leaving the summary to average over whatever survived.
        retrieved = gen = verdict = None
        delay = 20.0
        for attempt in range(5):
            try:
                retrieved = retriever.retrieve(q["question"])
                gen = generate_answer(llm, q["question"], retrieved)
                verdict = judge(judge_llm, q["question"],
                                q["reference_answer"], gen["answer"])
                time.sleep(1.2)                   # free-tier courtesy
                break
            except Exception as e:
                msg = str(e)
                rate = "RESOURCE_EXHAUSTED" in msg or "429" in msg
                print("  [%2d/%d] %s %s %s -- waiting %.0fs (attempt %d/5)"
                      % (i, len(questions), q["id"],
                         "rate limited" if rate else "ERROR",
                         "" if rate else type(e).__name__, delay, attempt + 1))
                time.sleep(delay)
                delay *= 1.8
        if verdict is None:
            print("  [%2d/%d] %s GIVING UP after 5 attempts" % (i, len(questions), q["id"]))
            gave_up.append(q["id"])
            continue
        pages = [d["page"] for d in retrieved]

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
        cache_file.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        row = score_row(row, q)
        results.append(row)

        hit = row["retrieval_hit"]
        flag = "HIT " if hit else ("MISS" if hit is False else "n/a ")
        print("  [%2d/%d] %s  retr=%s  grade=%-9s %.1fs"
              % (i, len(questions), q["id"], flag, verdict["grade"], gen["latency_s"]))

    # Report each reason separately. An earlier version labelled every omission
    # "skipped for missing indexes", which reported 32 missing indexes when all
    # nine were present and complete -- the questions simply had no cached result.
    # A coverage figure is only meaningful when the reason for the gap is honest.
    done = len(results)
    total = len(questions)
    if done < total:
        print("\n  COVERAGE: %d/%d questions scored (%.0f%%)"
              % (done, total, 100.0 * done / total))
        if not_cached:
            print("    %d not yet run (--cached-only): %s"
                  % (len(not_cached), ", ".join(not_cached[:8])
                     + ("..." if len(not_cached) > 8 else "")))
        if skipped:
            print("    %d skipped, index missing: %s"
                  % (len(skipped), ", ".join(skipped[:8])
                     + ("..." if len(skipped) > 8 else "")))
        if gave_up:
            print("    %d abandoned after retries (rate limits): %s"
                  % (len(gave_up), ", ".join(gave_up)))
        print("    Cached answers are reused, so a re-run only pays for these.")

    return results


def summarize(name, results, n_total=None):
    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]

    recall = sum(1 for r in ans if r["retrieval_hit"]) / len(ans) if ans else 0.0
    recall_len = sum(1 for r in ans if r["retrieval_hit_lenient"]) / len(ans) if ans else 0.0
    correct = sum(1 for r in ans if r["grade"] == "CORRECT") / len(ans) if ans else 0.0
    partial = sum(1 for r in ans if r["grade"] == "PARTIAL") / len(ans) if ans else 0.0
    refused_ans = sum(1 for r in ans if r["grade"] == "REFUSED") / len(ans) if ans else 0.0
    # On unanswerable questions, refusing is the CORRECT behaviour.
    halluc = (sum(1 for r in unans if r["grade"] != "REFUSED") / len(unans)) if unans else 0.0
    trunc = sum(1 for r in results if r["finish_reason"] == "length") / len(results) if results else 0.0
    lat = sum(r["latency_s"] for r in results) / len(results) if results else 0.0

    ranks = [r["hit_rank"] for r in ans if r["hit_rank"]]
    mrr = sum(1.0 / r for r in ranks) / len(ans) if ans else 0.0

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


def main():
    # --hard scores the paraphrased set instead of the original. Results are
    # labelled "<config>@hard" so the two sets never overwrite each other in
    # summary.json -- they are different tests and are not comparable.
    hard = "--hard" in sys.argv
    qa_path = (config.EVAL_DIR / "qa_set_hard.json") if hard else config.QA_SET_PATH
    suffix = "@hard" if hard else ""

    if not qa_path.exists():
        raise SystemExit("Missing %s -- generate it first." % qa_path.name)
    questions = json.loads(qa_path.read_text(encoding="utf-8"))
    print("Loaded %d questions from %s (%d answerable, %d unanswerable)"
          % (len(questions), qa_path.name,
             sum(1 for q in questions if q["answerable"]),
             sum(1 for q in questions if not q["answerable"])))

    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or list(config.CONFIGS)
    for w in wanted:
        if w not in config.CONFIGS:
            raise SystemExit("Unknown config %r. Options: %s" % (w, ", ".join(config.CONFIGS)))

    # --cached-only re-scores existing results without making any API call.
    # Metrics are derived from the cache on every run, so a change to how recall
    # is defined can be applied to past results for free.
    cached_only = "--cached-only" in sys.argv
    if cached_only:
        print("--cached-only: re-scoring from cache, no API calls will be made.")

    llm = None if cached_only else make_llm()
    judge_llm = None if cached_only else ChatGroq(
        model=config.JUDGE_MODEL, temperature=0, max_tokens=config.JUDGE_MAX_TOKENS)

    summaries = []
    for name in wanted:
        label = name + suffix
        results = run_config(label, config.CONFIGS[name], questions, llm, judge_llm,
                             cached_only=cached_only)
        if not results:
            print("  (no cached results for %s)" % label)
            continue
        (config.RESULTS_DIR / ("raw_%s.json" % label.replace("@", "_"))).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        s = summarize(label, results, n_total=len(questions))
        summaries.append(s)
        print("\n  -> recall@k %.1f%% | correct %.1f%% | halluc %.1f%% | trunc %.1f%%"
              % (s["recall_at_k"], s["correct_pct"], s["hallucination_pct"], s["truncated_pct"]))

    out = config.RESULTS_DIR / "summary.json"
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    merged = {s["config"]: s for s in existing}
    merged.update({s["config"]: s for s in summaries})
    out.write_text(json.dumps(list(merged.values()), indent=2), encoding="utf-8")

    print("\n%s\n%-16s %9s %8s %8s %6s %8s %8s %7s" % ("=" * 96, "CONFIG",
          "COVERAGE", "RECALL", "RECALL+", "MRR", "CORRECT", "HALLUC", "TRUNC"))
    print("-" * 96)
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


if __name__ == "__main__":
    main()
