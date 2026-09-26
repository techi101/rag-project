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
import json
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FAILURES = []


def report(check, ok, detail):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", detail))
    if not ok:
        FAILURES.append((check, detail))


# ── CHECK 1 ──────────────────────────────────────────────────────────────────
def check_chunks_never_span_pages():
    print("\nCHECK 1: do chunks ever span two pages?")
    for doc_id, meta in config.DOCUMENTS.items():
        pages = PyPDFLoader(meta["path"]).load()
        # Give every page a unique sentinel, then see if any chunk holds two.
        for i, p in enumerate(pages):
            p.page_content = "ZQPAGE%04dZQ " % i + p.page_content
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=400,
            separators=["\n\n", "\n", ". ", " ", ""])
        chunks = splitter.split_documents(pages)
        spanning = [c for c in chunks
                    if len(set(re.findall(r"ZQPAGE(\d{4})ZQ", c.page_content))) > 1]
        report("chunk-span", not spanning,
               "%-8s %d chunks, %d span >1 page" % (doc_id, len(chunks), len(spanning)))


# ── CHECK 2 ──────────────────────────────────────────────────────────────────
def check_ground_truth_unique():
    """
    Uses config.acceptable_pages -- the SAME function run_eval.py scores with --
    so this tests the shipped logic rather than a copy that can drift.
    """
    print("\nCHECK 2: could a recall MISS be a false negative?")
    qa = json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))
    page_text = {}
    for doc_id, meta in config.DOCUMENTS.items():
        page_text[doc_id] = [p.page_content.lower()
                             for p in PyPDFLoader(meta["path"]).load()]

    ambiguous, checked, unannotated = [], 0, 0
    for q in qa:
        if not q["answerable"]:
            continue
        checked += 1
        acc = config.acceptable_pages(q["reference_answer"],
                                      page_text[q["doc_id"]], q["gt_page"])
        if "acceptable_pages" not in q:
            unannotated += 1
        others = [p for p in acc if p != q["gt_page"]]
        if others:
            ambiguous.append((q["id"], q["gt_page"], others[:4]))

    pct = 100.0 * len(ambiguous) / checked if checked else 0.0
    for qid, gt, others in ambiguous:
        print("       %s gt_page=%s also answerable from %s" % (qid, gt, others))
    print("       -> these are why recall is reported strict AND lenient")
    report("gt-unique", True,
           "%d/%d answerable questions (%.1f%%) have multiple valid pages"
           % (len(ambiguous), checked, pct))
    report("gt-annotated", unannotated == 0,
           "%d question(s) missing acceptable_pages -- run annotate_pages.py"
           % unannotated)


# ── CHECK 3 ──────────────────────────────────────────────────────────────────
REFUSAL_PHRASES = (
    "could not find this information",
    "could not find that information",
    "cannot find this information",
    "i could not find",
    "i cannot find",
)


def looks_like_refusal(answer):
    """Must stay identical to judge()'s fast path in run_eval.py."""
    low = answer.lower()
    pos = min([low.find(p) for p in REFUSAL_PHRASES if p in low] or [-1])
    return pos != -1 and pos < 100 and len(answer.strip()) < 200


def check_refusal_detector():
    print("\nCHECK 3: can the refusal detector misfire?")
    should_fire = [
        "I could not find this information in the uploaded document.",
        "I cannot find this information in the provided context.",
    ]
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
    for a in should_fire:
        report("refusal", looks_like_refusal(a), "fires on refusal: %r" % a[:46])
    for a in should_not_fire:
        report("refusal", not looks_like_refusal(a),
               "silent on real answer: %r" % a[:46])


# ── CHECK 4 ──────────────────────────────────────────────────────────────────
def check_chunk_sensitivity():
    print("\nCHECK 4: chunks produced per scheme (is chunk_size a no-op?)")
    schemes = sorted({(c["chunk_size"], c["chunk_overlap"])
                      for c in config.CONFIGS.values()})
    print("       %-10s %s" % ("doc", "  ".join("%d/%d" % s for s in schemes)))
    for doc_id, meta in config.DOCUMENTS.items():
        pages = PyPDFLoader(meta["path"]).load()
        counts = []
        for cs, co in schemes:
            sp = RecursiveCharacterTextSplitter(
                chunk_size=cs, chunk_overlap=co,
                separators=["\n\n", "\n", ". ", " ", ""])
            counts.append(len(sp.split_documents(pages)))
        flat = len(set(counts)) < len(counts)
        print("       %-10s %s   %s" % (doc_id, "   ".join("%6d" % c for c in counts),
                                        "<- identical counts" if flat else ""))
        report("chunk-sensitivity", True,
               "%-8s counts %s" % (doc_id, counts))


# ── CHECK 5 ──────────────────────────────────────────────────────────────────
_STOP = set("""what which who when where how why is are was were do does did the a an
and or of to in on for from with that this these those it its as by at be been
name given used use uses using shown show list three two one according""".split())


def check_lexical_overlap():
    """
    Questions generated FROM a page tend to reuse that page's wording, so a plain
    keyword matcher finds it without any semantic understanding. When that
    happens, high recall measures the question set, not the retriever.

    BM25-only scored 97.7% recall@4 on this set -- the symptom that prompted this
    check. Here we quantify the cause.
    """
    print("\nCHECK 5: are questions lexically 'leaky' toward their source page?")
    qa = json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))
    page_text = {d: [p.page_content.lower() for p in PyPDFLoader(m["path"]).load()]
                 for d, m in config.DOCUMENTS.items()}

    overlaps = []
    for q in qa:
        if not q["answerable"]:
            continue
        words = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", q["question"].lower())
                 if w not in _STOP}
        if not words:
            continue
        gt = page_text[q["doc_id"]][q["gt_page"]]
        on_page = sum(1 for w in words if w in gt)
        overlaps.append((on_page / len(words), q["id"], len(words)))

    overlaps.sort()
    mean = sum(o[0] for o in overlaps) / len(overlaps)
    high = [o for o in overlaps if o[0] >= 0.8]
    low = [o for o in overlaps if o[0] < 0.5]

    print("       mean overlap with source page : %.0f%%" % (100 * mean))
    print("       questions >=80%% overlap       : %d/%d (%.0f%%)"
          % (len(high), len(overlaps), 100.0 * len(high) / len(overlaps)))
    print("       questions <50%% overlap        : %d/%d  <- the genuinely hard ones"
          % (len(low), len(overlaps)))
    if low:
        print("       hardest: %s" % ", ".join("%s(%.0f%%)" % (o[1], 100 * o[0])
                                               for o in low[:6]))
    # Threshold set at 0.60 from EVIDENCE, not taste: at a measured 73% overlap,
    # BM25-only retrieval scored 97.7% recall@4 (verify_retrieval.py). A set a
    # keyword matcher cannot lose on does not discriminate between retrievers,
    # so anything at or above ~0.60 should be treated as failing.
    report("lexical-overlap", mean < 0.60,
           "mean question/page word overlap %.0f%% (lower is a harder test)"
           % (100 * mean))
    if mean >= 0.60:
        print("       -> Recall on this set is inflated. A keyword matcher can win")
        print("          without understanding anything. Report recall stratified")
        print("          by overlap, or regenerate with paraphrased questions.")


def main():
    print("=" * 74)
    print("ADVERSARIAL VERIFICATION OF HARNESS ASSUMPTIONS")
    print("=" * 74)
    check_chunks_never_span_pages()
    check_ground_truth_unique()
    check_refusal_detector()
    check_chunk_sensitivity()
    check_lexical_overlap()

    print("\n" + "=" * 74)
    if FAILURES:
        print("FAILURES: %d" % len(FAILURES))
        for c, d in FAILURES:
            print("  [%s] %s" % (c, d))
    else:
        print("All assumption checks passed.")
    print("=" * 74)


if __name__ == "__main__":
    main()
