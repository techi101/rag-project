"""
eval/verify/verify_method.py
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
# WHAT THIS FILE IS: tries to break the harness's own assumptions (5 checks). No API calls, runs with quota dead.
# Example (README): CHECK 2 found "CREATE DATABASE startersql;" on pages 4 AND 65 (18.6% of questions have 2+ pages);
#          CHECK 5 measured 73% word overlap on the easy set.
# Flow: CHECK 1 chunks vs pages -> 2 ground-truth pages -> 3 refusal detector -> 4 chunk counts -> 5 overlap -> FAILs.

import json      # read eval/data/qa_set.json
import re        # regex: page markers, question words
import sys       # sys.path for imports
import pathlib   # this file's folder

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # add eval/ so "import config" works
import config    # eval/config.py: DOCUMENTS, CONFIGS, scoring helpers

from langchain_community.document_loaders import PyPDFLoader        # PDF -> one Document per page (same as app)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # app's chunker: paragraph, line, sentence, space

FAILURES = []  # (check name, detail) of every FAIL, printed at the end


# IN: check name, ok, detail -> OUT: prints "[PASS]/[FAIL] detail", saves fails
# e.g. report("refusal", False, "silent on ...") -> "  [FAIL] silent on ..."
def report(check, ok, detail):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", detail))  # one result line
    if not ok:                                             # failed?
        FAILURES.append((check, detail))                   # remember for summary


# ── CHECK 1 ──────────────────────────────────────────────────────────────────
# IN: PDFs -> OUT: PASS/FAIL per PDF. README: 0 of 792 chunks span two pages.
# e.g. "mysql    71 chunks, 0 span >1 page"
def check_chunks_never_span_pages():
    print("\nCHECK 1: do chunks ever span two pages?")
    for doc_id, meta in config.DOCUMENTS.items():          # each PDF
        pages = PyPDFLoader(meta["path"]).load()           # one Document per page
        # Give every page a unique sentinel, then see if any chunk holds two.
        for i, p in enumerate(pages):                      # i = page index
            p.page_content = "ZQPAGE%04dZQ " % i + p.page_content  # e.g. "ZQPAGE0004ZQ ", "ZQ" never in real text
        splitter = RecursiveCharacterTextSplitter(         # biggest size = most likely to merge pages
            chunk_size=2000, chunk_overlap=400,
            separators=["\n\n", "\n", ". ", " ", ""])      # paragraph, line, sentence, space, char
        chunks = splitter.split_documents(pages)           # chunk all pages
        spanning = [c for c in chunks
                    if len(set(re.findall(r"ZQPAGE(\d{4})ZQ", c.page_content))) > 1]  # 2+ distinct markers
        report("chunk-span", not spanning,                 # pass if none span
               "%-8s %d chunks, %d span >1 page" % (doc_id, len(chunks), len(spanning)))


# ── CHECK 2 ──────────────────────────────────────────────────────────────────
# IN: qa_set.json + PDFs -> OUT: questions with extra valid pages + 2 checks
# recall@k = any of top k chunks from the right page? e.g. "q001 gt_page=4 also answerable from [65]"
def check_ground_truth_unique():
    """
    Uses config.acceptable_pages -- the SAME function run_eval.py scores with --
    so this tests the shipped logic rather than a copy that can drift.
    """
    print("\nCHECK 2: could a recall MISS be a false negative?")
    qa = json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))  # easy set
    page_text = {}                                         # doc_id -> lowercase page texts
    for doc_id, meta in config.DOCUMENTS.items():
        page_text[doc_id] = [p.page_content.lower()
                             for p in PyPDFLoader(meta["path"]).load()]  # lowercase = case-blind

    ambiguous, checked, unannotated = [], 0, 0             # multi-page Qs, count, missing field
    for q in qa:
        if not q["answerable"]:                            # skip probes
            continue
        checked += 1                                       # count answerable
        acc = config.acceptable_pages(q["reference_answer"],
                                      page_text[q["doc_id"]], q["gt_page"])  # pages with all answer words (3+)
        if "acceptable_pages" not in q:                    # annotate_pages.py not run
            unannotated += 1
        others = [p for p in acc if p != q["gt_page"]]     # extra valid pages
        if others:
            ambiguous.append((q["id"], q["gt_page"], others[:4]))  # keep max 4

    pct = 100.0 * len(ambiguous) / checked if checked else 0.0  # % multi-page, guard zero
    for qid, gt, others in ambiguous:                      # list them
        print("       %s gt_page=%s also answerable from %s" % (qid, gt, others))
    print("       -> these are why recall is reported strict AND lenient")
    report("gt-unique", True,                              # always pass: just a number
           "%d/%d answerable questions (%.1f%%) have multiple valid pages"
           % (len(ambiguous), checked, pct))
    report("gt-annotated", unannotated == 0,               # pass only if all annotated
           "%d question(s) missing acceptable_pages -- run annotate_pages.py"
           % unannotated)


# ── CHECK 3 ──────────────────────────────────────────────────────────────────
# Refusal phrases; must match judge() in run_eval.py exactly.
REFUSAL_PHRASES = (
    "could not find this information",
    "could not find that information",
    "cannot find this information",
    "i could not find",
    "i cannot find",
)


# IN: answer -> OUT: True if it looks like a refusal (copy of judge()'s fast path)
# rule: phrase in first 100 chars AND answer under 200 chars (template refusal is 59)
# e.g. "I could not find this information in the uploaded document." -> True
def looks_like_refusal(answer):
    """Must stay identical to judge()'s fast path in run_eval.py."""
    low = answer.lower()                                   # case-blind
    pos = min([low.find(p) for p in REFUSAL_PHRASES if p in low] or [-1])  # earliest phrase, -1 = none
    return pos != -1 and pos < 100 and len(answer.strip()) < 200  # early + short = refusal


# IN: nothing -> OUT: PASS/FAIL for 2 refusals + 4 real answers
# README Defect 2: first version matched "not in the" and misfired
def check_refusal_detector():
    print("\nCHECK 3: can the refusal detector misfire?")
    should_fire = [                                        # real refusals
        "I could not find this information in the uploaded document.",
        "I cannot find this information in the provided context.",
    ]
    should_not_fire = [                                    # real answers
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
    for a in should_fire:                                  # must detect
        report("refusal", looks_like_refusal(a), "fires on refusal: %r" % a[:46])  # 46 chars shown
    for a in should_not_fire:                              # must stay silent
        report("refusal", not looks_like_refusal(a),
               "silent on real answer: %r" % a[:46])


# ── CHECK 4 ──────────────────────────────────────────────────────────────────
# IN: PDFs -> OUT: chunk counts per PDF x scheme
# README: MySQL 115 / 71 / 71 at 500 / 1000 / 2000 -> "<- identical counts"
def check_chunk_sensitivity():
    print("\nCHECK 4: chunks produced per scheme (is chunk_size a no-op?)")
    schemes = sorted({(c["chunk_size"], c["chunk_overlap"])
                      for c in config.CONFIGS.values()})   # (500,100) (1000,200) (2000,400)
    print("       %-10s %s" % ("doc", "  ".join("%d/%d" % s for s in schemes)))  # header row
    for doc_id, meta in config.DOCUMENTS.items():          # each PDF
        pages = PyPDFLoader(meta["path"]).load()
        counts = []                                        # one count per scheme
        for cs, co in schemes:
            sp = RecursiveCharacterTextSplitter(
                chunk_size=cs, chunk_overlap=co,
                separators=["\n\n", "\n", ". ", " ", ""])  # same as the app
            counts.append(len(sp.split_documents(pages)))  # e.g. 115
        flat = len(set(counts)) < len(counts)              # any two counts equal?
        print("       %-10s %s   %s" % (doc_id, "   ".join("%6d" % c for c in counts),
                                        "<- identical counts" if flat else ""))  # one row
        report("chunk-sensitivity", True,                  # always pass: a measurement
               "%-8s counts %s" % (doc_id, counts))


# ── CHECK 5 ──────────────────────────────────────────────────────────────────
# stop words = common words with no topic meaning, removed before measuring overlap
_STOP = set("""what which who when where how why is are was were do does did the a an
and or of to in on for from with that this these those it its as by at be been
name given used use uses using shown show list three two one according""".split())


# IN: qa_set.json + PDFs -> OUT: mean overlap, leaky/hard counts, PASS/FAIL
# README: easy set 73% overlap, BM25 (keyword ranking, no AI) recall@4 97.7%
def check_lexical_overlap():
    """
    Questions generated FROM a page tend to reuse that page's wording, so a plain
    keyword matcher finds it without any semantic understanding. When that
    happens, high recall measures the question set, not the retriever.

    BM25-only scored 97.7% recall@4 on this set -- the symptom that prompted this
    check. Here we quantify the cause.
    """
    print("\nCHECK 5: are questions lexically 'leaky' toward their source page?")
    qa = json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))  # easy set
    page_text = {d: [p.page_content.lower() for p in PyPDFLoader(m["path"]).load()]
                 for d, m in config.DOCUMENTS.items()}     # doc_id -> lowercase pages

    # ── Step 1: overlap per question ──
    overlaps = []                                          # (fraction, id, word count)
    for q in qa:
        if not q["answerable"]:                            # probes have no page
            continue
        words = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", q["question"].lower())  # 3+ char words
                 if w not in _STOP}                        # minus stop words
        if not words:                                      # nothing to measure
            continue
        gt = page_text[q["doc_id"]][q["gt_page"]]          # source page (0-based)
        on_page = sum(1 for w in words if w in gt)         # substring test, not whole word
        overlaps.append((on_page / len(words), q["id"], len(words)))  # e.g. (0.75, "q001", 4)

    # ── Step 2: summarise ──
    overlaps.sort()                                        # hardest first
    mean = sum(o[0] for o in overlaps) / len(overlaps)     # e.g. 0.73
    high = [o for o in overlaps if o[0] >= 0.8]            # very leaky
    low = [o for o in overlaps if o[0] < 0.5]              # genuinely hard

    print("       mean overlap with source page : %.0f%%" % (100 * mean))  # "%%" = literal %
    print("       questions >=80%% overlap       : %d/%d (%.0f%%)"
          % (len(high), len(overlaps), 100.0 * len(high) / len(overlaps)))
    print("       questions <50%% overlap        : %d/%d  <- the genuinely hard ones"
          % (len(low), len(overlaps)))
    if low:                                                # show up to 6 hard ones
        print("       hardest: %s" % ", ".join("%s(%.0f%%)" % (o[1], 100 * o[0])
                                               for o in low[:6]))
    # Threshold set at 0.60 from EVIDENCE, not taste: at a measured 73% overlap,
    # BM25-only retrieval scored 97.7% recall@4 (verify_retrieval.py). A set a
    # keyword matcher cannot lose on does not discriminate between retrievers,
    # so anything at or above ~0.60 should be treated as failing.
    report("lexical-overlap", mean < 0.60,                 # pass below 60%
           "mean question/page word overlap %.0f%% (lower is a harder test)"
           % (100 * mean))
    if mean >= 0.60:                                       # too leaky: advise
        print("       -> Recall on this set is inflated. A keyword matcher can win")
        print("          without understanding anything. Report recall stratified")
        print("          by overlap, or regenerate with paraphrased questions.")


# ── BLOCK: main ─────────────────────────────────────────────────────────
# IN: nothing -> OUT: all 5 checks + failure list
# e.g. easy set: CHECK 5 FAILS (73% > 60%)
def main():
    print("=" * 74)
    print("ADVERSARIAL VERIFICATION OF HARNESS ASSUMPTIONS")
    print("=" * 74)
    check_chunks_never_span_pages()                        # CHECK 1
    check_ground_truth_unique()                            # CHECK 2
    check_refusal_detector()                               # CHECK 3
    check_chunk_sensitivity()                              # CHECK 4
    check_lexical_overlap()                                # CHECK 5

    print("\n" + "=" * 74)
    if FAILURES:                                           # any FAIL?
        print("FAILURES: %d" % len(FAILURES))
        for c, d in FAILURES:                              # list each
            print("  [%s] %s" % (c, d))
    else:
        print("All assumption checks passed.")
    print("=" * 74)


if __name__ == "__main__":  # only when run directly
    main()                  # start
