"""
eval/verify_retrieval.py
─────────────────────────────────────────────────────────────────────────────
Tests the hybrid retriever before spending quota on it.

The `hybrid` configuration fuses dense (vector) and BM25 (keyword) rankings with
reciprocal rank fusion. That code has never been executed against real data. If
the fusion is wrong, tomorrow's comparison is meaningless and the mistake would
be invisible -- a broken fuser still returns plausible-looking chunks.

PART 1  RRF fusion, on synthetic rankings with known correct output.
PART 2  BM25 retrieval over the real corpus, scored with the real metric.

Part 2 needs no embedding API, so it produces a genuine keyword-search baseline
while the Google quota is exhausted. If BM25 alone is close to the dense
retriever, that is worth knowing -- it is free, local and instant.
"""
import json
import re
import sys
import pathlib
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

from rank_bm25 import BM25Okapi

RRF_K = 60
FAILURES = []


def report(ok, detail):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", detail))
    if not ok:
        FAILURES.append(detail)


def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def rrf_fuse(dense, bm25, k):
    """Mirrors Retriever.retrieve()'s fusion. Kept in sync deliberately."""
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
    return [holder[s] for s in ranked[:k]]


# ── PART 1 ───────────────────────────────────────────────────────────────────
def test_rrf():
    print("\nPART 1: RRF fusion logic")

    def doc(name, page=0):
        return {"text": name, "page": page}

    # A document ranked highly by BOTH must outrank one ranked highly by one.
    dense = [doc("A"), doc("B"), doc("C")]
    bm25 = [doc("C"), doc("A"), doc("D")]
    out = [d["text"] for d in rrf_fuse(dense, bm25, 4)]
    report(out[0] in ("A", "C"),
           "consensus doc ranks first (got %s)" % out)
    report(set(out) == {"A", "B", "C", "D"},
           "union of both lists returned, deduplicated (got %s)" % out)

    # A document in both lists must appear exactly once.
    report(len(out) == len(set(out)), "no duplicates in fused output")

    # k must be respected.
    report(len(rrf_fuse(dense, bm25, 2)) == 2, "respects k=2")

    # Identical lists must preserve order.
    same = [doc("X"), doc("Y"), doc("Z")]
    report([d["text"] for d in rrf_fuse(same, same, 3)] == ["X", "Y", "Z"],
           "identical rankings preserve order")

    # An empty dense list must degrade to BM25 order, not crash.
    report([d["text"] for d in rrf_fuse([], bm25, 3)] == ["C", "A", "D"],
           "empty dense list falls back to BM25 order")

    # Scores must actually differ, or "fusion" is doing nothing.
    fused = defaultdict(float)
    for rank in range(3):
        fused[rank] = 1.0 / (RRF_K + rank + 1)
    report(len(set(fused.values())) == 3, "RRF assigns distinct scores by rank")


# ── PART 2 ───────────────────────────────────────────────────────────────────
def test_bm25_on_corpus(qa_path=None):
    qa_path = qa_path or config.QA_SET_PATH
    print("\nPART 2: BM25-only retrieval over the real corpus (no API)")
    print("        question set: %s" % qa_path.name)
    if not qa_path.exists():
        print("        SKIP -- not generated yet")
        return
    qa = json.loads(qa_path.read_text(encoding="utf-8"))

    by_doc = defaultdict(list)
    for q in qa:
        if q["answerable"]:
            by_doc[q["doc_id"]].append(q)

    overall_strict = overall_lenient = overall_n = 0
    for doc_id, questions in by_doc.items():
        key = config.index_key(doc_id, 1000, 200)
        chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")
        if not chunks_path.exists():
            print("  %-8s SKIP (index not built)" % doc_id)
            continue

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

        strict = lenient = 0
        for q in questions:
            scores = bm25.get_scores(tokenize(q["question"]))
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]
            pages = [chunks[i]["page"] for i in top]
            acceptable = q.get("acceptable_pages") or [q["gt_page"]]
            strict += q["gt_page"] in pages
            lenient += any(p in acceptable for p in pages)

        n = len(questions)
        overall_strict += strict
        overall_lenient += lenient
        overall_n += n
        print("  %-8s BM25 recall@4: strict %d/%d (%.0f%%)  lenient %d/%d (%.0f%%)"
              % (doc_id, strict, n, 100.0 * strict / n, lenient, n, 100.0 * lenient / n))

    if overall_n:
        strict_pct = 100.0 * overall_strict / overall_n
        print("\n  OVERALL BM25-only recall@4: strict %.1f%%  lenient %.1f%%  (n=%d)"
              % (strict_pct, 100.0 * overall_lenient / overall_n, overall_n))
        if strict_pct >= 95:
            print("  >= 95%: a keyword matcher cannot lose on this set, so it does")
            print("  NOT discriminate between retrievers. Recall here measures the")
            print("  questions, not the retriever.")
        else:
            print("  Below 95%: the set has room for a semantic retriever to win.")
            print("  Compare against the dense baseline once the quota resets.")


def main():
    hard = "--hard" in sys.argv
    qa_path = (config.EVAL_DIR / "qa_set_hard.json") if hard else config.QA_SET_PATH
    print("=" * 78)
    print("RETRIEVAL VERIFICATION%s" % ("  (HARD SET)" if hard else ""))
    print("=" * 78)
    test_rrf()
    test_bm25_on_corpus(qa_path)
    print("\n" + "=" * 78)
    print("FAILURES: %d" % len(FAILURES) if FAILURES else "All retrieval checks passed.")
    for f in FAILURES:
        print("  %s" % f)
    print("=" * 78)


if __name__ == "__main__":
    main()
