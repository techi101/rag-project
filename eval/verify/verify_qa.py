"""
eval/verify/verify_qa.py
─────────────────────────────────────────────────────────────────────────────
Quality control on the generated question set. Run this BEFORE trusting any
score, because a test built on bad questions produces confident nonsense.

It catches three failure modes found in the first generated batch:

  1. FALSE UNANSWERABLES -- generate_qa.py shows the model only a SAMPLE of the
     document when asking for an "unanswerable" question, so it can claim a
     topic is absent when it appears later in the full text. Every unanswerable
     probe is re-checked against the COMPLETE document.

  2. DUPLICATES -- the same prompt run twice can produce near-identical
     questions, which silently double-weights one fact.

  3. LEAKED GROUND TRUTH -- a question that quotes its own answer, or that
     names a page, is not testing retrieval.

Usage:
    py -3.12 eval/verify/verify_qa.py           # report only
    py -3.12 eval/verify/verify_qa.py --fix     # drop duplicates, write back
"""
# WHAT THIS FILE IS: proof-reads the LLM-written questions in eval/data/qa_set.json for 3 mistakes:
# probe (should-be-unanswerable) whose topic IS in the PDF, near-duplicate questions, questions saying "page 12".
# Example: probe "slow query log" is SUSPECT if "slow_query_log" appears anywhere in the MySQL handbook.
# Flow: load questions + full PDF text -> check probes -> duplicates -> leaks -> summary -> --fix drops duplicates.

import json      # read/write the question set
import re        # regex: spot "page 12", "according to the text"
import sys       # sys.argv (--fix), sys.path
import pathlib   # this file's folder
from difflib import SequenceMatcher  # 0-1 string similarity, 1.0 = identical

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # add eval/ so "import config" works
import config    # eval/config.py: DOCUMENTS, QA_SET_PATH

from langchain_community.document_loaders import PyPDFLoader  # PDF -> pages, for full text

# Distinctive terms per unanswerable probe. If ANY appears in the full document,
# the probe is suspect and must be reviewed by hand.
PROBE_TERMS = {                                            # probe topic -> words proving it IS in the PDF
    "slow query log": ["slow query log", "slow_query_log"],
    "storage engine": ["storage engine", "myisam", "engine="],
    "slowly changing": ["slowly changing", "scd type", "type 2 dimension"],
    "maintenance budget": ["maintenance budget", "annual budget"],
    "coco map": ["test-dev", "map on coco"],
    "optimizer": ["optimizer", "learning rate", "adam"],
}

DUPLICATE_THRESHOLD = 0.82  # hand-picked: skips shared "What is the ...", catches rewordings


# IN: doc_id e.g. "mysql" -> OUT: whole PDF as one lowercase string
# checks probes vs the COMPLETE text, not the generator's sample
def full_text(doc_id):
    path = config.DOCUMENTS[doc_id]["path"]                # PDF path
    return " ".join(p.page_content for p in PyPDFLoader(path).load()).lower()  # all pages joined


# IN: optional --fix -> OUT: report; with --fix, qa_set.json rewritten without duplicates
# e.g. "--fix: dropped 1 duplicate(s), ..."
def main():
    fix = "--fix" in sys.argv                              # True if --fix typed
    qa = json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))  # all questions
    texts = {d: full_text(d) for d in config.DOCUMENTS}    # doc_id -> full lowercase text
    problems = []                                          # (id, reason) for summary

    # ---- Check 1: probes against the full document text ----
    print("=" * 74)
    print("1. UNANSWERABLE PROBES vs FULL DOCUMENT TEXT")
    print("=" * 74)
    for q in qa:
        if q["answerable"]:                                # probes only
            continue
        body = q["question"].lower()                       # lowercase question
        matched = [term for key, terms in PROBE_TERMS.items()
                   for term in terms if term in body.replace("?", "") or key in body]  # topic terms for this probe
        hits = [t for t in set(matched) if t in texts[q["doc_id"]]]  # terms really in the PDF
        status = "SUSPECT" if hits else "ok"               # any hit = maybe answerable
        print("  %-6s [%-6s] %-7s %s" % (q["id"], q["doc_id"], status, q["question"][:60]))  # one line per probe
        if hits:
            print("         terms present in document: %s" % hits)  # which terms
            problems.append((q["id"], "probe topic present in document"))

    # ---- Check 2: near-duplicate questions ----
    print("\n" + "=" * 74)
    print("2. NEAR-DUPLICATE QUESTIONS")
    print("=" * 74)
    dupes = set()                                          # ids to drop (later of each pair)
    for i in range(len(qa)):                               # every pair once
        for j in range(i + 1, len(qa)):                    # j always after i
            if qa[i]["doc_id"] != qa[j]["doc_id"]:         # same PDF only
                continue
            ratio = SequenceMatcher(None, qa[i]["question"].lower(),
                                    qa[j]["question"].lower()).ratio()  # 0-1 similarity
            if ratio >= DUPLICATE_THRESHOLD:               # 0.82+ = duplicate
                print("  %s ~ %s  (similarity %.2f)" % (qa[i]["id"], qa[j]["id"], ratio))
                print("     %s" % qa[i]["question"][:66])  # first question
                print("     %s" % qa[j]["question"][:66])  # second question
                dupes.add(qa[j]["id"])            # keep the first, drop the later
                problems.append((qa[j]["id"], "near-duplicate of %s" % qa[i]["id"]))
    if not dupes:
        print("  none found")

    # ---- Check 3: questions that leak where the answer is ----
    print("\n" + "=" * 74)
    print("3. LEAKED GROUND TRUTH")
    print("=" * 74)
    leaked = 0                                             # leaky question count
    for q in qa:
        body = q["question"].lower()
        # regex: "page 12" OR "according to the text/passage/document" OR "this page"
        if re.search(r"\bpage \d+|according to the (text|passage|document)|this page", body):
            print("  %s  %s" % (q["id"], q["question"][:66]))
            problems.append((q["id"], "references the source rather than asking"))
            leaked += 1
    if not leaked:
        print("  none found")

    # ---- Summary ----
    print("\n" + "=" * 74)
    print("SUMMARY: %d questions, %d problems" % (len(qa), len(problems)))
    for qid, why in problems:                              # each problem
        print("  %-6s %s" % (qid, why))
    print("=" * 74)

    # ── --fix: drop duplicates and save ──
    if fix and dupes:
        kept = [q for q in qa if q["id"] not in dupes]     # non-duplicates
        config.QA_SET_PATH.write_text(
            json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")  # pretty JSON, keep non-English
        print("\n--fix: dropped %d duplicate(s), %d questions remain"
              % (len(dupes), len(kept)))
    elif dupes:                                            # found, but no --fix
        print("\nRe-run with --fix to drop duplicates.")


if __name__ == "__main__":  # only when run directly
    main()                  # start
