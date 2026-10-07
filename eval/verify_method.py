"""
eval/verify_method.py
─────────────────────────────────────────────────────────────────────────────
Adversarial checks on the harness's own assumptions.

Every claim this project makes rests on something that could be wrong. This
script tries to break each one. It uses NO embedding API, so it runs even when
the Google free tier is exhausted.

CHECK 1  Chunks never span pages
         The claim "chunk_size=2000 changes nothing on sparse documents"
         depends on RecursiveCharacterTextSplitter never merging across pages.
         Asserted from reading the code; here it is tested.

CHECK 2  Ground-truth pages are unique
         Recall@k scores a MISS whenever no retrieved chunk comes from gt_page.
         If the same answer also appears on other pages, that MISS is a FALSE
         NEGATIVE -- the retriever found a correct page and we punished it.
         This measures how often that can happen.

CHECK 3  Refusal detection cannot misfire
         The fast-path grades an answer REFUSED on phrase match. If it fires on
         a real answer, hallucination rate is understated.

CHECK 4  Chunk-count sensitivity per document
         Reports chunks per scheme so "chunk size did nothing" is a measured
         statement, not an impression.
"""
# WHAT THIS FILE IS: the "self-inspection" of the evaluation harness. Like a teacher checking that the answer key
# itself has no mistakes before marking any exam, it tries to break the harness's own assumptions.
# It makes NO API calls (no embeddings, no LLM), so it runs even when Google's daily quota is used up.
# Real example from eval/README.md: CHECK 2 found that "CREATE DATABASE startersql;" sits on both page 4 and
# page 65 of the MySQL handbook, so crediting only one page was unfair to the retriever (18.6% of answerable
# questions have more than one valid page). CHECK 5 measured 73% word overlap on the easy set.
# Overall flow: CHECK 1 chunks never span pages -> CHECK 2 ground-truth pages -> CHECK 3 refusal detector
# -> CHECK 4 chunk counts per size -> CHECK 5 question/page word overlap -> print every FAIL at the end.
#
# json: reads the question set eval/qa_set.json.
import json
# re = regular expressions: small patterns that find text, e.g. our page markers or the words in a question.
import re
# sys: changes the import path so "import config" finds eval/config.py.
import sys
# pathlib: finds the folder this file lives in.
import pathlib

# Put eval/ first on Python's search path, so "import config" works from any starting folder.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# config = eval/config.py: the document list, the configurations, and the shared scoring helpers.
import config

# PyPDFLoader: LangChain's PDF reader (built on pypdf). It returns one Document per page, with the page number
# in metadata. Same loader the live app uses, so the checks test the real behaviour.
from langchain_community.document_loaders import PyPDFLoader
# RecursiveCharacterTextSplitter: the same chunker the app uses. It cuts text at paragraph breaks first, then
# lines, then sentences, then spaces, and only cuts mid-word as a last resort.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAILURES: a list that collects every failed check as (check name, detail), printed in main() at the end.
FAILURES = []


# IN: check name, ok (True/False), detail text -> OUT: prints "[PASS] ..." or "[FAIL] ..."; failures are also saved.
# WHY: one place that prints results in the same format and remembers what failed.
# Example: report("refusal", False, "silent on real answer: ...") prints "  [FAIL] silent on real answer: ..."
# and adds ("refusal", "silent on real answer: ...") to FAILURES.
def report(check, ok, detail):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", detail))
    # Only failed checks are remembered for the final summary.
    if not ok:
        FAILURES.append((check, detail))


# ── CHECK 1 ──────────────────────────────────────────────────────────────────
# IN: nothing (reads the PDFs listed in config.DOCUMENTS) -> OUT: one PASS/FAIL line per document.
# WHY: the claim "chunk_size=2000 changes nothing on sparse documents" is only true if the splitter never joins
# two pages into one chunk. eval/README.md reports the result: 0 of 792 chunks span two pages.
# Example: MySQL has 71 chunks at size 2000 (eval/README.md), so a passing line reads "mysql    71 chunks, 0 span >1 page".
def check_chunks_never_span_pages():
    print("\nCHECK 1: do chunks ever span two pages?")
    # Loop over each document in the corpus (doc_id like "mysql", meta holds its file path).
    for doc_id, meta in config.DOCUMENTS.items():
        # Read the PDF: one Document object per page.
        pages = PyPDFLoader(meta["path"]).load()
        # Give every page a unique sentinel, then see if any chunk holds two.
        # Loop over pages with their position i (0, 1, 2, ...).
        for i, p in enumerate(pages):
            # Put a marker like "ZQPAGE0004ZQ " at the start of page 4. "%04d" = the number padded to 4 digits.
            # "ZQ" is a letter pair that never appears in real text, so the marker cannot be confused with real content.
            p.page_content = "ZQPAGE%04dZQ " % i + p.page_content
        # Use the LARGEST chunk size tested (2000, overlap 400), because big chunks are the most likely to merge pages.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=400,
            separators=["\n\n", "\n", ". ", " ", ""])
        # Split all pages into chunks.
        chunks = splitter.split_documents(pages)
        # Regex r"ZQPAGE(\d{4})ZQ": the literal text "ZQPAGE", then (\d{4}) = exactly 4 digits captured as a group
        # (the page number), then the literal "ZQ". findall returns every page number found in a chunk.
        # set(...) removes repeats; more than 1 distinct page number means the chunk mixes two pages.
        spanning = [c for c in chunks
                    if len(set(re.findall(r"ZQPAGE(\d{4})ZQ", c.page_content))) > 1]
        # PASS if the list of spanning chunks is empty (an empty list counts as False, so "not spanning" is True).
        report("chunk-span", not spanning,
               "%-8s %d chunks, %d span >1 page" % (doc_id, len(chunks), len(spanning)))


# ── CHECK 2 ──────────────────────────────────────────────────────────────────
# IN: nothing (reads qa_set.json and the PDFs) -> OUT: printed list of questions with extra valid pages + 2 checks.
# WHY: recall@k ("did any of the top k retrieved chunks come from the right page?") would wrongly score a MISS
# when the answer is also on another page. This counts how often that can happen.
# Example: "q001 gt_page=4 also answerable from [65]".
def check_ground_truth_unique():
    """
    Uses config.acceptable_pages -- the SAME function run_eval.py scores with --
    so this tests the shipped logic rather than a copy that can drift.
    """
    print("\nCHECK 2: could a recall MISS be a false negative?")
    # Load every question in the easy set.
    qa = json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))
    # page_text maps doc_id -> list of page texts in lowercase (lowercase so matching ignores capital letters).
    page_text = {}
    # Loop over each document and read all of its pages.
    for doc_id, meta in config.DOCUMENTS.items():
        page_text[doc_id] = [p.page_content.lower()
                             for p in PyPDFLoader(meta["path"]).load()]

    # ambiguous = questions with more than one valid page; checked = answerable questions looked at;
    # unannotated = questions whose JSON entry has no "acceptable_pages" list yet.
    ambiguous, checked, unannotated = [], 0, 0
    # Loop over every question.
    for q in qa:
        # Skip probes (questions the document cannot answer); they have no answer page.
        if not q["answerable"]:
            continue
        # Count this answerable question.
        checked += 1
        # acceptable_pages (eval/config.py) = every page that holds ALL of the answer's distinctive words (at least 3 of
        # them needed), plus the original gt_page.
        acc = config.acceptable_pages(q["reference_answer"],
                                      page_text[q["doc_id"]], q["gt_page"])
        # If the question file has no saved "acceptable_pages" field, annotate_pages.py has not been run for it.
        if "acceptable_pages" not in q:
            unannotated += 1
        # Pages other than the one the question was written from.
        others = [p for p in acc if p != q["gt_page"]]
        # If there are any, record the question id, its page and at most the first 4 other pages.
        if others:
            ambiguous.append((q["id"], q["gt_page"], others[:4]))

    # Percentage of answerable questions with several valid pages. "if checked else 0.0" avoids dividing by zero.
    pct = 100.0 * len(ambiguous) / checked if checked else 0.0
    # Print each ambiguous question.
    for qid, gt, others in ambiguous:
        print("       %s gt_page=%s also answerable from %s" % (qid, gt, others))
    print("       -> these are why recall is reported strict AND lenient")
    # This one always passes (True): it only reports the number, it is not a pass/fail rule.
    report("gt-unique", True,
           "%d/%d answerable questions (%.1f%%) have multiple valid pages"
           % (len(ambiguous), checked, pct))
    # This one passes only if every question already has "acceptable_pages" saved.
    report("gt-annotated", unannotated == 0,
           "%d question(s) missing acceptable_pages -- run annotate_pages.py"
           % unannotated)


# ── CHECK 3 ──────────────────────────────────────────────────────────────────
# The phrases that mark a refusal (the model saying "it is not in the document").
# They must match the list inside judge() in run_eval.py exactly.
REFUSAL_PHRASES = (
    "could not find this information",
    "could not find that information",
    "cannot find this information",
    "i could not find",
    "i cannot find",
)


# IN: an answer string -> OUT: True if it looks like a refusal.
# WHY: a copy of judge()'s quick refusal test, so it can be attacked here without any API call.
# Rule: a refusal phrase must appear in the first 100 characters AND the whole answer must be under 200 characters.
# The app's template refusal is 59 characters, so 200 leaves room; real answers that mention the phrase are longer.
# Example: "I could not find this information in the uploaded document." -> True;
# a 340-character answer that says "could not find this information" in the middle -> False.
def looks_like_refusal(answer):
    """Must stay identical to judge()'s fast path in run_eval.py."""
    low = answer.lower()
    # For every phrase found, low.find gives its position; min picks the earliest one. If none is found, the
    # "or [-1]" turns the empty list into [-1], so pos = -1 means "no phrase".
    pos = min([low.find(p) for p in REFUSAL_PHRASES if p in low] or [-1])
    return pos != -1 and pos < 100 and len(answer.strip()) < 200


# IN: nothing -> OUT: PASS/FAIL lines for 2 answers that must be detected and 4 that must not.
# WHY: if the detector fires on a real answer, that answer is graded REFUSED and the hallucination rate looks
# better than it is. eval/README.md (Defect 2): the first version matched "not in the" and misfired.
# Example: "The buffer pool is not in the query cache; ..." must NOT be called a refusal.
def check_refusal_detector():
    print("\nCHECK 3: can the refusal detector misfire?")
    # Two real refusals: the detector must fire on these.
    should_fire = [
        "I could not find this information in the uploaded document.",
        "I cannot find this information in the provided context.",
    ]
    # Four real answers: the detector must stay silent on these.
    should_not_fire = [
        "The buffer pool is not in the query cache; it is a separate structure "
        "that InnoDB uses to cache table and index data in memory.",
        "LEFT JOIN returns all rows from the left table. Rows with no match are "
        "returned with NULL values in the right table's columns.",
        "Internal data, external data and personal data are the three categories.",
        # Adversarial: contains the phrase but is a genuine long answer.
        "Earlier versions could not find this information quickly, so MySQL 8.0 "
        "added a histogram-based optimizer that samples column values to estimate "
        "selectivity, which improves join ordering decisions substantially when "
        "indexes are absent or when data distribution is heavily skewed. This "
        "changes plan selection for large analytical queries in practice.",
    ]
    # Each refusal must be detected. a[:46] = first 46 characters, just to keep the printed line short.
    for a in should_fire:
        report("refusal", looks_like_refusal(a), "fires on refusal: %r" % a[:46])
    # Each real answer must NOT be detected as a refusal.
    for a in should_not_fire:
        report("refusal", not looks_like_refusal(a),
               "silent on real answer: %r" % a[:46])


# ── CHECK 4 ──────────────────────────────────────────────────────────────────
# IN: nothing (reads the PDFs) -> OUT: a table of chunk counts per document and chunking scheme.
# WHY: turns "chunk size did nothing" into a measured number. eval/README.md: MySQL gives 115 / 71 / 71 chunks at
# 500 / 1000 / 2000 characters, so going from 1000 to 2000 changes nothing on that document.
# Example row: "mysql  115  71  71  <- identical counts".
def check_chunk_sensitivity():
    print("\nCHECK 4: chunks produced per scheme (is chunk_size a no-op?)")
    # The unique (chunk_size, chunk_overlap) pairs from config.CONFIGS: (500, 100), (1000, 200), (2000, 400).
    schemes = sorted({(c["chunk_size"], c["chunk_overlap"])
                      for c in config.CONFIGS.values()})
    # Header row, e.g. "doc  500/100  1000/200  2000/400".
    print("       %-10s %s" % ("doc", "  ".join("%d/%d" % s for s in schemes)))
    # Loop over each document.
    for doc_id, meta in config.DOCUMENTS.items():
        pages = PyPDFLoader(meta["path"]).load()
        # counts will hold one chunk count per scheme.
        counts = []
        # Split the same pages with each scheme and count the chunks.
        for cs, co in schemes:
            sp = RecursiveCharacterTextSplitter(
                chunk_size=cs, chunk_overlap=co,
                separators=["\n\n", "\n", ". ", " ", ""])
            counts.append(len(sp.split_documents(pages)))
        # flat = True if any two schemes gave the same count (a set drops duplicates, so it gets shorter).
        flat = len(set(counts)) < len(counts)
        # Print the row, marking documents where chunk size made no difference.
        print("       %-10s %s   %s" % (doc_id, "   ".join("%6d" % c for c in counts),
                                        "<- identical counts" if flat else ""))
        # Always PASS: this is a measurement, not a rule.
        report("chunk-sensitivity", True,
               "%-8s counts %s" % (doc_id, counts))


# ── CHECK 5 ──────────────────────────────────────────────────────────────────
# _STOP = stop words: very common words ("what", "the", "of", ...) that carry no topic meaning.
# They are removed before measuring overlap, so overlap reflects the topic words only.
_STOP = set("""what which who when where how why is are was were do does did the a an
and or of to in on for from with that this these those it its as by at be been
name given used use uses using shown show list three two one according""".split())


# IN: nothing (reads qa_set.json and the PDFs) -> OUT: mean overlap, share of very leaky and hard questions, PASS/FAIL.
# WHY: if most question words appear on the answer page, a plain keyword search finds it without understanding.
# Example from eval/README.md: the easy set measured 73% mean overlap, and BM25 (a classic keyword-ranking
# method, no AI) scored 97.7% recall@4 on it - so the set could not tell a smart retriever from a dumb one.
def check_lexical_overlap():
    """
    Questions generated FROM a page tend to reuse that page's wording, so a plain
    keyword matcher finds it without any semantic understanding. When that
    happens, high recall measures the question set, not the retriever.

    BM25-only scored 97.7% recall@4 on this set -- the symptom that prompted this
    check. Here we quantify the cause.
    """
    print("\nCHECK 5: are questions lexically 'leaky' toward their source page?")
    # Load the question set.
    qa = json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))
    # Dictionary comprehension: doc_id -> list of lowercase page texts, for every document.
    page_text = {d: [p.page_content.lower() for p in PyPDFLoader(m["path"]).load()]
                 for d, m in config.DOCUMENTS.items()}

    # overlaps will hold (overlap fraction, question id, number of content words) per question.
    overlaps = []
    # Loop over every question.
    for q in qa:
        # Skip probes: they have no source page to compare with.
        if not q["answerable"]:
            continue
        # Regex r"[a-z][a-z0-9_]{2,}": a word that starts with a letter, followed by 2 or more letters, digits or
        # underscores (so words of 3+ characters, like "select" or "user_id"). Stop words are then removed.
        words = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", q["question"].lower())
                 if w not in _STOP}
        # A question made only of stop words has nothing to measure; skip it.
        if not words:
            continue
        # The text of the question's own source page (gt_page is used as a list position; PyPDFLoader counts pages from 0).
        gt = page_text[q["doc_id"]][q["gt_page"]]
        # Count content words found on that page. "w in gt" is a plain substring test, not a whole-word test.
        on_page = sum(1 for w in words if w in gt)
        overlaps.append((on_page / len(words), q["id"], len(words)))

    # Sort from lowest to highest overlap, so the hardest questions come first.
    overlaps.sort()
    # Average overlap across all questions.
    mean = sum(o[0] for o in overlaps) / len(overlaps)
    # high = questions with 80% or more of their words on the page (very leaky).
    # low = questions with under 50% (the genuinely hard ones).
    high = [o for o in overlaps if o[0] >= 0.8]
    low = [o for o in overlaps if o[0] < 0.5]

    # Print the summary numbers. "%%" prints a literal "%" sign inside a % format string.
    print("       mean overlap with source page : %.0f%%" % (100 * mean))
    print("       questions >=80%% overlap       : %d/%d (%.0f%%)"
          % (len(high), len(overlaps), 100.0 * len(high) / len(overlaps)))
    print("       questions <50%% overlap        : %d/%d  <- the genuinely hard ones"
          % (len(low), len(overlaps)))
    # If there are hard questions, show up to 6 of them with their overlap.
    if low:
        print("       hardest: %s" % ", ".join("%s(%.0f%%)" % (o[1], 100 * o[0])
                                               for o in low[:6]))
    # Threshold set at 0.60 from EVIDENCE, not taste: at a measured 73% overlap,
    # BM25-only retrieval scored 97.7% recall@4 (verify_retrieval.py). A set a
    # keyword matcher cannot lose on does not discriminate between retrievers,
    # so anything at or above ~0.60 should be treated as failing.
    # PASS only if the mean overlap is below 60% (the threshold explained in the comment just above).
    report("lexical-overlap", mean < 0.60,
           "mean question/page word overlap %.0f%% (lower is a harder test)"
           % (100 * mean))
    # If the set is too leaky, print advice on what to do about it.
    if mean >= 0.60:
        print("       -> Recall on this set is inflated. A keyword matcher can win")
        print("          without understanding anything. Report recall stratified")
        print("          by overlap, or regenerate with paraphrased questions.")


# IN: nothing -> OUT: runs all 5 checks and prints a final list of failures (or an all-clear).
# WHY: one command (py -3.12 eval/verify_method.py) runs every assumption check.
# Example: on the easy set CHECK 5 FAILS (73% overlap is above the 60% limit); eval/README.md lists it as FAILED.
def main():
    print("=" * 74)
    print("ADVERSARIAL VERIFICATION OF HARNESS ASSUMPTIONS")
    print("=" * 74)
    # Run the five checks in order; each prints its own section.
    check_chunks_never_span_pages()
    check_ground_truth_unique()
    check_refusal_detector()
    check_chunk_sensitivity()
    check_lexical_overlap()

    print("\n" + "=" * 74)
    # If any check failed, list each failure with its check name.
    if FAILURES:
        print("FAILURES: %d" % len(FAILURES))
        # Print each saved failure.
        for c, d in FAILURES:
            print("  [%s] %s" % (c, d))
    # No failures at all.
    else:
        print("All assumption checks passed.")
    print("=" * 74)


# Run main() only when this file is started directly, not when imported.
if __name__ == "__main__":
    main()
