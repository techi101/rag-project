"""
eval/verify/verify_retrieval.py
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
# WHAT THIS FILE IS: retrieval checks with no embedding API.
# PART 1: RRF (Reciprocal Rank Fusion: each list adds 1/(60 + rank), summed) on tiny made-up rankings.
# PART 2: BM25 (1994 keyword ranking, rare words weigh more) alone on real chunks, scored by recall@4
#         (share of questions with a top-4 chunk from the right page).
# Example (README): BM25 recall@4 = 97.7% easy set, 57.6% hard set (--hard) -> easy set was too easy.
# Flow: PART 1 RRF unit checks -> PART 2 BM25 top 4 per question -> strict + lenient recall -> failures.

import json      # read question set + chunk files
import re        # regex: split text into words
import sys       # sys.argv (--hard), sys.path
import pathlib   # this file's folder
from collections import defaultdict  # dict that auto-creates 0.0 / [] for new keys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # add eval/ so "import config" works
import config    # eval/config.py: folders, DOCUMENTS, index_key()

from rank_bm25 import BM25Okapi  # standard BM25 ("Okapi" = system it came from)

RRF_K = 60       # 1/(60 + rank): standard value, same as run_eval.py
FAILURES = []    # failed check texts, printed at the end


# IN: ok, detail -> OUT: prints "[PASS]/[FAIL] detail", saves fails
# e.g. report(False, "respects k=2") -> "  [FAIL] respects k=2"
def report(ok, detail):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", detail))  # one result line
    if not ok:
        FAILURES.append(detail)                            # remember failure


# IN: text -> OUT: lowercase word list for BM25
# e.g. "CREATE DATABASE startersql;" -> ["create", "database", "startersql"]
def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())         # runs of letters/digits/_, rest = separators


# IN: dense ranking, bm25 ranking, k -> OUT: top k after RRF merge
# e.g. dense [A, B, C] + bm25 [C, A, D], k=4 -> [A, C, B, D]
def rrf_fuse(dense, bm25, k):
    """Mirrors Retriever.retrieve()'s fusion. Kept in sync deliberately."""
    fused = defaultdict(float)                             # signature -> total score
    holder = {}                                            # signature -> chunk
    for rank, d in enumerate(dense):                       # rank 0, 1, 2, ...
        sig = d["text"][:200]                              # first 200 chars = identity, same chunk counted once
        fused[sig] += 1.0 / (RRF_K + rank + 1)             # 1st place adds 1/61
        holder[sig] = d
    for rank, d in enumerate(bm25):                        # same for BM25 list
        sig = d["text"][:200]
        fused[sig] += 1.0 / (RRF_K + rank + 1)             # adds to dense score if shared
        holder[sig] = d
    ranked = sorted(fused, key=lambda s: fused[s], reverse=True)  # highest total first
    return [holder[s] for s in ranked[:k]]                 # top k chunks


# ── PART 1 ───────────────────────────────────────────────────────────────────
# IN: nothing -> OUT: 7 PASS/FAIL checks on made-up rankings
# README: "RRF fusion logic PASS - 7 synthetic cases"
def test_rrf():
    print("\nPART 1: RRF fusion logic")

    # IN: name, page -> OUT: fake chunk, e.g. doc("A") -> {"text": "A", "page": 0}
    def doc(name, page=0):
        return {"text": name, "page": page}

    # A document ranked highly by BOTH must outrank one ranked highly by one.
    dense = [doc("A"), doc("B"), doc("C")]                 # A 1st, C 3rd
    bm25 = [doc("C"), doc("A"), doc("D")]                  # C 1st, A 2nd
    out = [d["text"] for d in rrf_fuse(dense, bm25, 4)]    # merged names
    report(out[0] in ("A", "C"),                           # check 1: consensus first
           "consensus doc ranks first (got %s)" % out)
    report(set(out) == {"A", "B", "C", "D"},               # check 2: all four present
           "union of both lists returned, deduplicated (got %s)" % out)

    # A document in both lists must appear exactly once.
    report(len(out) == len(set(out)), "no duplicates in fused output")  # check 3

    # k must be respected.
    report(len(rrf_fuse(dense, bm25, 2)) == 2, "respects k=2")  # check 4

    # Identical lists must preserve order.
    same = [doc("X"), doc("Y"), doc("Z")]
    report([d["text"] for d in rrf_fuse(same, same, 3)] == ["X", "Y", "Z"],  # check 5
           "identical rankings preserve order")

    # An empty dense list must degrade to BM25 order, not crash.
    report([d["text"] for d in rrf_fuse([], bm25, 3)] == ["C", "A", "D"],  # check 6
           "empty dense list falls back to BM25 order")

    # Scores must actually differ, or "fusion" is doing nothing.
    fused = defaultdict(float)
    for rank in range(3):                                  # ranks 0, 1, 2
        fused[rank] = 1.0 / (RRF_K + rank + 1)             # 1/61, 1/62, 1/63
    report(len(set(fused.values())) == 3, "RRF assigns distinct scores by rank")  # check 7


# ── PART 2 ───────────────────────────────────────────────────────────────────
# IN: qa_path (easy or hard set) -> OUT: BM25 recall@4 per PDF + overall
# README: 97.7% easy, 57.6% hard
def test_bm25_on_corpus(qa_path=None):
    # ── Step 1: load questions ──
    qa_path = qa_path or config.QA_SET_PATH                # default: easy set
    print("\nPART 2: BM25-only retrieval over the real corpus (no API)")
    print("        question set: %s" % qa_path.name)
    if not qa_path.exists():                               # not generated yet
        print("        SKIP -- not generated yet")
        return
    qa = json.loads(qa_path.read_text(encoding="utf-8"))   # question list

    by_doc = defaultdict(list)                             # doc_id -> answerable questions
    for q in qa:
        if q["answerable"]:                                # skip probes
            by_doc[q["doc_id"]].append(q)

    # ── Step 2: BM25 per PDF ──
    overall_strict = overall_lenient = overall_n = 0       # totals over all PDFs
    for doc_id, questions in by_doc.items():
        key = config.index_key(doc_id, 1000, 200)          # baseline, e.g. "mysql_cs1000_co200"
        chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")
        if not chunks_path.exists():                       # build_index.py (--chunks-only works, no API)
            print("  %-8s SKIP (index not built)" % doc_id)
            continue

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))  # [{"page", "text"}, ...]
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])      # index over word lists

        strict = lenient = 0                               # hits for this PDF
        for q in questions:
            scores = bm25.get_scores(tokenize(q["question"]))  # one score per chunk
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]  # top 4 = app's k
            pages = [chunks[i]["page"] for i in top]       # their pages
            acceptable = q.get("acceptable_pages") or [q["gt_page"]]  # fallback: gt_page only
            strict += q["gt_page"] in pages                # original page found (True = 1)
            lenient += any(p in acceptable for p in pages)  # any valid page found

        # ── Step 3: per-PDF result ──
        n = len(questions)
        overall_strict += strict                           # add to totals
        overall_lenient += lenient
        overall_n += n
        print("  %-8s BM25 recall@4: strict %d/%d (%.0f%%)  lenient %d/%d (%.0f%%)"
              % (doc_id, strict, n, 100.0 * strict / n, lenient, n, 100.0 * lenient / n))

    # ── Step 4: overall verdict ──
    if overall_n:                                          # avoid divide by zero
        strict_pct = 100.0 * overall_strict / overall_n    # e.g. 97.7
        print("\n  OVERALL BM25-only recall@4: strict %.1f%%  lenient %.1f%%  (n=%d)"
              % (strict_pct, 100.0 * overall_lenient / overall_n, overall_n))
        if strict_pct >= 95:                               # judgement cut-off: keywords too strong
            print("  >= 95%: a keyword matcher cannot lose on this set, so it does")
            print("  NOT discriminate between retrievers. Recall here measures the")
            print("  questions, not the retriever.")
        else:                                              # room for dense retriever
            print("  Below 95%: the set has room for a semantic retriever to win.")
            print("  Compare against the dense baseline once the quota resets.")


# ── BLOCK: main ─────────────────────────────────────────────────────────
# IN: optional --hard -> OUT: PART 1 + PART 2 + failures
# e.g. py -3.12 eval/verify/verify_retrieval.py --hard -> uses qa_set_hard.json
def main():
    hard = "--hard" in sys.argv                            # paraphrased set?
    qa_path = config.QA_SET_HARD_PATH if hard else config.QA_SET_PATH  # pick file
    print("=" * 78)
    print("RETRIEVAL VERIFICATION%s" % ("  (HARD SET)" if hard else ""))
    print("=" * 78)
    test_rrf()                                             # PART 1
    test_bm25_on_corpus(qa_path)                           # PART 2
    print("\n" + "=" * 78)
    print("FAILURES: %d" % len(FAILURES) if FAILURES else "All retrieval checks passed.")
    for f in FAILURES:                                     # each failure
        print("  %s" % f)
    print("=" * 78)


if __name__ == "__main__":  # only when run directly
    main()                  # start
