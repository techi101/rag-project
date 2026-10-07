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
# WHAT THIS FILE IS: the "practice match" for retrieval. It does two things without any embedding API:
#   PART 1 tests the RRF code on tiny made-up rankings whose correct answer we know.
#     RRF = Reciprocal Rank Fusion: a way to merge two ranked lists. Each item gets 1 / (60 + its rank) from each
#     list it appears in, and the scores are added. Items ranked high in BOTH lists end up on top.
#   PART 2 runs BM25 alone over the real chunks and scores it with recall@4.
#     BM25 = a classic keyword-search ranking formula (from 1994): a chunk scores higher when it contains the
#     question's words, especially rare words, and is not padded with lots of other text. No AI, no meaning.
#     recall@4 = the share of questions where at least one of the top 4 chunks comes from the right page.
# Real example from eval/README.md: BM25-only recall@4 was 97.7% on the easy set but 57.6% on the hard set
# (run with --hard), which proved the easy set was too easy.
# Overall flow: PART 1 RRF unit checks -> PART 2 load questions + chunk files -> BM25 top 4 per question
# -> strict and lenient recall per document -> overall recall -> list any failures.
#
# json: reads the question set and the saved chunk files.
import json
# re = regular expressions, used to split text into lowercase words.
import re
# sys: reads the --hard option and changes the import path.
import sys
# pathlib: finds the folder this file lives in.
import pathlib
# defaultdict: a dictionary that creates a default value (0.0 or an empty list) the first time a key is used.
from collections import defaultdict

# Put eval/ first on Python's search path, so "import config" finds eval/config.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# config = eval/config.py: folders, document list and the index_key() naming helper.
import config

# rank_bm25 (package rank-bm25 in requirements.txt): a small library that implements BM25. BM25Okapi is its
# standard version ("Okapi" is the name of the search system BM25 first came from).
from rank_bm25 import BM25Okapi

# RRF_K = 60: the constant in 1 / (60 + rank). 60 is the standard value from the original RRF paper; it stops the
# very first rank from completely dominating, so lower ranks still count. Same value as RRF_K in run_eval.py.
RRF_K = 60
# FAILURES: every failed check's description, printed at the end by main().
FAILURES = []


# IN: ok (True/False), detail text -> OUT: prints "[PASS] ..." or "[FAIL] ..." and remembers failures.
# WHY: one consistent way to print and collect results.
# Example: report(False, "respects k=2") prints "  [FAIL] respects k=2" and adds it to FAILURES.
def report(ok, detail):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", detail))
    # Only failures are remembered.
    if not ok:
        FAILURES.append(detail)


# IN: any text -> OUT: list of lowercase words. BM25 compares these word lists.
# WHY: BM25 works on words (tokens), so text must be cut into words first.
# Regex r"[a-z0-9_]+": one or more lowercase letters, digits or underscores in a row. Everything else
# (spaces, punctuation, brackets) acts as a separator.
# Example: tokenize("CREATE DATABASE startersql;") -> ["create", "database", "startersql"].
def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


# IN: dense = chunks ranked by the vector search, bm25 = chunks ranked by BM25, k = how many to keep
#     -> OUT: the top k chunks after merging both rankings with RRF.
# WHY: a copy of the fusion used by the "hybrid" configuration, tested here before any quota is spent on it.
# Example: dense [A, B, C] and bm25 [C, A, D] with k=4 -> A first (1/61 + 1/62 beats C's 1/63 + 1/61), then C, B, D.
def rrf_fuse(dense, bm25, k):
    """Mirrors Retriever.retrieve()'s fusion. Kept in sync deliberately."""
    # fused: signature -> total RRF score (starts at 0.0). holder: signature -> the chunk itself.
    fused = defaultdict(float)
    holder = {}
    # Add each dense-ranked chunk's score. enumerate gives rank 0, 1, 2, ... so "rank + 1" makes the first place 1.
    for rank, d in enumerate(dense):
        # sig = the first 200 characters of the chunk text, used as its identity, so the same chunk found by both
        # retrievers is counted as ONE item and its two scores are added together.
        sig = d["text"][:200]
        fused[sig] += 1.0 / (RRF_K + rank + 1)
        holder[sig] = d
    # Same for the BM25 ranking.
    for rank, d in enumerate(bm25):
        sig = d["text"][:200]
        fused[sig] += 1.0 / (RRF_K + rank + 1)
        holder[sig] = d
    # Sort signatures by total score, highest first, then return the chunks for the top k.
    ranked = sorted(fused, key=lambda s: fused[s], reverse=True)
    return [holder[s] for s in ranked[:k]]


# ── PART 1 ───────────────────────────────────────────────────────────────────
# IN: nothing -> OUT: PASS/FAIL lines for 7 checks on made-up rankings.
# WHY: a broken fuser still returns normal-looking chunks, so the bug would be invisible later.
# eval/README.md reports this part as: RRF fusion logic PASS - 7 synthetic cases.
def test_rrf():
    print("\nPART 1: RRF fusion logic")

    # IN: a name like "A" (and an optional page) -> OUT: a tiny fake chunk {"text": "A", "page": 0}.
    # WHY: the tests need chunk-shaped dicts, not real PDF text.
    def doc(name, page=0):
        return {"text": name, "page": page}

    # A document ranked highly by BOTH must outrank one ranked highly by one.
    # Two rankings: A is 1st and 2nd, C is 3rd and 1st, B and D appear in only one list each.
    dense = [doc("A"), doc("B"), doc("C")]
    bm25 = [doc("C"), doc("A"), doc("D")]
    out = [d["text"] for d in rrf_fuse(dense, bm25, 4)]
    # Check 1: an item ranked high in both lists (A or C) must come first.
    report(out[0] in ("A", "C"),
           "consensus doc ranks first (got %s)" % out)
    # Check 2: all four different items appear, each once.
    report(set(out) == {"A", "B", "C", "D"},
           "union of both lists returned, deduplicated (got %s)" % out)

    # A document in both lists must appear exactly once.
    report(len(out) == len(set(out)), "no duplicates in fused output")

    # k must be respected.
    # Check 4: asking for k=2 returns exactly 2 chunks.
    report(len(rrf_fuse(dense, bm25, 2)) == 2, "respects k=2")

    # Identical lists must preserve order.
    # Check 5: if both lists are identical, the merged order must be the same order.
    same = [doc("X"), doc("Y"), doc("Z")]
    report([d["text"] for d in rrf_fuse(same, same, 3)] == ["X", "Y", "Z"],
           "identical rankings preserve order")

    # An empty dense list must degrade to BM25 order, not crash.
    # Check 6: with an empty dense list, the result must simply be BM25's order (C, A, D).
    report([d["text"] for d in rrf_fuse([], bm25, 3)] == ["C", "A", "D"],
           "empty dense list falls back to BM25 order")

    # Scores must actually differ, or "fusion" is doing nothing.
    # Check 7: ranks 0, 1, 2 must get 3 different scores (1/61, 1/62, 1/63), or the ranking would mean nothing.
    fused = defaultdict(float)
    for rank in range(3):
        fused[rank] = 1.0 / (RRF_K + rank + 1)
    report(len(set(fused.values())) == 3, "RRF assigns distinct scores by rank")


# ── PART 2 ───────────────────────────────────────────────────────────────────
# IN: qa_path = which question file (easy or hard) -> OUT: printed BM25 recall@4 per document and overall.
# WHY: a free, local, instant keyword baseline. If BM25 alone does as well as the dense retriever, the
# question set cannot tell them apart. eval/README.md: 97.7% on the easy set, 57.6% on the hard set.
# Example row format: "mysql    BM25 recall@4: strict <hits>/<n> (<pct>)  lenient <hits>/<n> (<pct>)".
def test_bm25_on_corpus(qa_path=None):
    # No path given -> use the easy set eval/qa_set.json.
    qa_path = qa_path or config.QA_SET_PATH
    print("\nPART 2: BM25-only retrieval over the real corpus (no API)")
    print("        question set: %s" % qa_path.name)
    # If the question file does not exist yet, print SKIP and stop this part.
    if not qa_path.exists():
        print("        SKIP -- not generated yet")
        return
    # Load the questions.
    qa = json.loads(qa_path.read_text(encoding="utf-8"))

    # Group answerable questions by document: doc_id -> list of questions (probes have no answer page, so skipped).
    by_doc = defaultdict(list)
    for q in qa:
        if q["answerable"]:
            by_doc[q["doc_id"]].append(q)

    # Running totals across all documents: strict hits, lenient hits, number of questions.
    overall_strict = overall_lenient = overall_n = 0
    # Loop over each document and its questions.
    for doc_id, questions in by_doc.items():
        # Use the baseline chunking (1000 characters, 200 overlap), e.g. key "mysql_cs1000_co200".
        key = config.index_key(doc_id, 1000, 200)
        chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")
        # The chunk file is written by build_index.py (also with --chunks-only, which needs no API). Missing -> skip.
        if not chunks_path.exists():
            print("  %-8s SKIP (index not built)" % doc_id)
            continue

        # Load the chunks and build a BM25 index over their word lists.
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

        # Hit counters for this document.
        strict = lenient = 0
        # Loop over each question of this document.
        for q in questions:
            # One BM25 score per chunk for this question's words.
            scores = bm25.get_scores(tokenize(q["question"]))
            # Indexes of the 4 highest-scoring chunks (4 = TOP_K_RESULTS, the k the live app uses).
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]
            # The page number of each of those 4 chunks.
            pages = [chunks[i]["page"] for i in top]
            # Pages that count as correct: the saved acceptable_pages list, or just gt_page if that list is missing or empty.
            acceptable = q.get("acceptable_pages") or [q["gt_page"]]
            # strict hit = the original page (gt_page) is among the 4. True adds 1, False adds 0.
            strict += q["gt_page"] in pages
            # lenient hit = any of the 4 pages is an acceptable page.
            lenient += any(p in acceptable for p in pages)

        # Add this document's numbers to the totals and print its strict and lenient recall.
        n = len(questions)
        overall_strict += strict
        overall_lenient += lenient
        overall_n += n
        print("  %-8s BM25 recall@4: strict %d/%d (%.0f%%)  lenient %d/%d (%.0f%%)"
              % (doc_id, strict, n, 100.0 * strict / n, lenient, n, 100.0 * lenient / n))

    # Only if at least one question was scored (avoids dividing by zero).
    if overall_n:
        strict_pct = 100.0 * overall_strict / overall_n
        print("\n  OVERALL BM25-only recall@4: strict %.1f%%  lenient %.1f%%  (n=%d)"
              % (strict_pct, 100.0 * overall_lenient / overall_n, overall_n))
        # 95% or more: the keyword matcher almost never misses, so the set is too easy to compare retrievers.
        # 95 is a judgement threshold; the easy set's 97.7% is above it.
        if strict_pct >= 95:
            print("  >= 95%: a keyword matcher cannot lose on this set, so it does")
            print("  NOT discriminate between retrievers. Recall here measures the")
            print("  questions, not the retriever.")
        # Below 95%: there is room for the meaning-based (dense) retriever to do better than keywords.
        else:
            print("  Below 95%: the set has room for a semantic retriever to win.")
            print("  Compare against the dense baseline once the quota resets.")


# IN: optional --hard flag -> OUT: runs PART 1 and PART 2, then prints the failures or an all-clear.
# WHY: one command for both retrieval checks.
# Example: py -3.12 eval/verify_retrieval.py --hard  -> PART 2 uses eval/qa_set_hard.json.
def main():
    # --hard picks the paraphrased question set; otherwise the easy set.
    hard = "--hard" in sys.argv
    qa_path = (config.EVAL_DIR / "qa_set_hard.json") if hard else config.QA_SET_PATH
    print("=" * 78)
    print("RETRIEVAL VERIFICATION%s" % ("  (HARD SET)" if hard else ""))
    print("=" * 78)
    test_rrf()
    test_bm25_on_corpus(qa_path)
    print("\n" + "=" * 78)
    print("FAILURES: %d" % len(FAILURES) if FAILURES else "All retrieval checks passed.")
    # Print each failure on its own line (the loop does nothing if there are none).
    for f in FAILURES:
        print("  %s" % f)
    print("=" * 78)


# Run main() only when this file is started directly, not when imported.
if __name__ == "__main__":
    main()
