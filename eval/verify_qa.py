"""
eval/verify_qa.py
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
    py -3.12 eval/verify_qa.py           # report only
    py -3.12 eval/verify_qa.py --fix     # drop duplicates, write back
"""
import json
import re
import sys
import pathlib
from difflib import SequenceMatcher

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

from langchain_community.document_loaders import PyPDFLoader

# Distinctive terms per unanswerable probe. If ANY appears in the full document,
# the probe is suspect and must be reviewed by hand.
PROBE_TERMS = {
    "slow query log": ["slow query log", "slow_query_log"],
    "storage engine": ["storage engine", "myisam", "engine="],
    "slowly changing": ["slowly changing", "scd type", "type 2 dimension"],
    "maintenance budget": ["maintenance budget", "annual budget"],
    "coco map": ["test-dev", "map on coco"],
    "optimizer": ["optimizer", "learning rate", "adam"],
}

DUPLICATE_THRESHOLD = 0.82


def full_text(doc_id):
    path = config.DOCUMENTS[doc_id]["path"]
    return " ".join(p.page_content for p in PyPDFLoader(path).load()).lower()


def main():
    fix = "--fix" in sys.argv
    qa = json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))
    texts = {d: full_text(d) for d in config.DOCUMENTS}
    problems = []

    print("=" * 74)
    print("1. UNANSWERABLE PROBES vs FULL DOCUMENT TEXT")
    print("=" * 74)
    for q in qa:
        if q["answerable"]:
            continue
        body = q["question"].lower()
        matched = [term for key, terms in PROBE_TERMS.items()
                   for term in terms if term in body.replace("?", "") or key in body]
        hits = [t for t in set(matched) if t in texts[q["doc_id"]]]
        status = "SUSPECT" if hits else "ok"
        print("  %-6s [%-6s] %-7s %s" % (q["id"], q["doc_id"], status, q["question"][:60]))
        if hits:
            print("         terms present in document: %s" % hits)
            problems.append((q["id"], "probe topic present in document"))

    print("\n" + "=" * 74)
    print("2. NEAR-DUPLICATE QUESTIONS")
    print("=" * 74)
    dupes = set()
    for i in range(len(qa)):
        for j in range(i + 1, len(qa)):
            if qa[i]["doc_id"] != qa[j]["doc_id"]:
                continue
            ratio = SequenceMatcher(None, qa[i]["question"].lower(),
                                    qa[j]["question"].lower()).ratio()
            if ratio >= DUPLICATE_THRESHOLD:
                print("  %s ~ %s  (similarity %.2f)" % (qa[i]["id"], qa[j]["id"], ratio))
                print("     %s" % qa[i]["question"][:66])
                print("     %s" % qa[j]["question"][:66])
                dupes.add(qa[j]["id"])            # keep the first, drop the later
                problems.append((qa[j]["id"], "near-duplicate of %s" % qa[i]["id"]))
    if not dupes:
        print("  none found")

    print("\n" + "=" * 74)
    print("3. LEAKED GROUND TRUTH")
    print("=" * 74)
    leaked = 0
    for q in qa:
        body = q["question"].lower()
        if re.search(r"\bpage \d+|according to the (text|passage|document)|this page", body):
            print("  %s  %s" % (q["id"], q["question"][:66]))
            problems.append((q["id"], "references the source rather than asking"))
            leaked += 1
    if not leaked:
        print("  none found")

    print("\n" + "=" * 74)
    print("SUMMARY: %d questions, %d problems" % (len(qa), len(problems)))
    for qid, why in problems:
        print("  %-6s %s" % (qid, why))
    print("=" * 74)

    if fix and dupes:
        kept = [q for q in qa if q["id"] not in dupes]
        config.QA_SET_PATH.write_text(
            json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n--fix: dropped %d duplicate(s), %d questions remain"
              % (len(dupes), len(kept)))
    elif dupes:
        print("\nRe-run with --fix to drop duplicates.")


if __name__ == "__main__":
    main()
