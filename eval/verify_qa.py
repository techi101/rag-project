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
# WHAT THIS FILE IS: the "proof-reader" of the exam paper. The questions in eval/qa_set.json were written by an
# LLM, so before any score is trusted this script checks them for three kinds of mistakes:
#   1. a "probe" (a question the document should NOT be able to answer) whose topic is actually in the document,
#   2. two questions that are almost the same (one fact counted twice),
#   3. a question that gives the game away by saying "page 12" or "according to the text".
# Real example: the probe topic "slow query log" is flagged SUSPECT if "slow_query_log" appears anywhere
# in the MySQL handbook's full text.
# Overall flow: load questions + full text of each PDF -> check probes -> check near-duplicates -> check leaks
# -> print a summary -> with --fix, remove duplicates and save the file again.
#
# json: read and write the question set file.
import json
# re = regular expressions (text patterns), used to spot "page 12" or "according to the document".
import re
# sys: read command-line options (--fix) and change the import path.
import sys
# pathlib: finds the folder this file lives in.
import pathlib
# SequenceMatcher (Python's built-in difflib): gives a 0-to-1 similarity score between two strings.
# 1.0 = identical, 0.0 = nothing in common. Used to find near-duplicate questions.
from difflib import SequenceMatcher

# Put eval/ first on Python's search path, so "import config" finds eval/config.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# config = eval/config.py: document paths and the path of qa_set.json.
import config

# PyPDFLoader: LangChain's PDF reader, one Document per page; used to get the FULL text of each document.
from langchain_community.document_loaders import PyPDFLoader

# Distinctive terms per unanswerable probe. If ANY appears in the full document,
# the probe is suspect and must be reviewed by hand.
# Each key is a probe topic; the list holds words that would show the topic IS in the document.
# Example: for "storage engine", finding "myisam" or "engine=" in the MySQL handbook makes the probe suspect.
PROBE_TERMS = {
    "slow query log": ["slow query log", "slow_query_log"],
    "storage engine": ["storage engine", "myisam", "engine="],
    "slowly changing": ["slowly changing", "scd type", "type 2 dimension"],
    "maintenance budget": ["maintenance budget", "annual budget"],
    "coco map": ["test-dev", "map on coco"],
    "optimizer": ["optimizer", "learning rate", "adam"],
}

# Two questions with similarity 0.82 or more count as duplicates. 0.82 is a hand-picked cut-off: high enough
# that questions sharing only a common opening ("What is the ...") are not flagged, low enough to catch rewordings.
DUPLICATE_THRESHOLD = 0.82


# IN: doc_id like "mysql" -> OUT: the whole document as one lowercase string (all pages joined by spaces).
# WHY: a probe must be checked against the COMPLETE document, not the sample the generator saw.
# Example: full_text("mysql") -> "mysql handbook ... create database startersql; ..." (lowercased).
def full_text(doc_id):
    path = config.DOCUMENTS[doc_id]["path"]
    return " ".join(p.page_content for p in PyPDFLoader(path).load()).lower()


# IN: command-line flag --fix (optional) -> OUT: printed report; with --fix, qa_set.json is rewritten without duplicates.
# WHY: a test built on bad questions gives confident but wrong scores.
# Example: py -3.12 eval/verify_qa.py --fix  -> "--fix: dropped 1 duplicate(s), ..." if a duplicate was found.
def main():
    # fix = True only if "--fix" was typed on the command line.
    fix = "--fix" in sys.argv
    # Load all questions, and the full lowercase text of every document (doc_id -> text).
    qa = json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))
    texts = {d: full_text(d) for d in config.DOCUMENTS}
    # problems collects (question id, reason) for the final summary.
    problems = []

    # ---- Check 1: probes against the full document text ----
    print("=" * 74)
    print("1. UNANSWERABLE PROBES vs FULL DOCUMENT TEXT")
    print("=" * 74)
    # Loop over every question.
    for q in qa:
        # Only probes (answerable = False) are checked here; skip normal questions.
        if q["answerable"]:
            continue
        # Lowercase question text.
        body = q["question"].lower()
        # Collect the topic words linked to this probe: a term is taken if it appears in the question (with "?" removed),
        # or if the topic key itself appears in the question (then every term of that key is taken).
        matched = [term for key, terms in PROBE_TERMS.items()
                   for term in terms if term in body.replace("?", "") or key in body]
        # hits = those terms that really appear in the document's full text. Any hit means the probe may be answerable.
        hits = [t for t in set(matched) if t in texts[q["doc_id"]]]
        # Print one line per probe: id, document, status, first 60 characters of the question.
        status = "SUSPECT" if hits else "ok"
        print("  %-6s [%-6s] %-7s %s" % (q["id"], q["doc_id"], status, q["question"][:60]))
        # If any term was found, show which ones and record the problem.
        if hits:
            print("         terms present in document: %s" % hits)
            problems.append((q["id"], "probe topic present in document"))

    # ---- Check 2: near-duplicate questions ----
    print("\n" + "=" * 74)
    print("2. NEAR-DUPLICATE QUESTIONS")
    print("=" * 74)
    # dupes = ids of the LATER question of each duplicate pair (the ones to drop).
    dupes = set()
    # Compare every pair of questions once: i from the start, j always after i.
    for i in range(len(qa)):
        for j in range(i + 1, len(qa)):
            # Only compare questions from the same document; skip pairs from different documents.
            if qa[i]["doc_id"] != qa[j]["doc_id"]:
                continue
            # Similarity of the two lowercase questions, 0 to 1. None = do not treat any characters as junk.
            ratio = SequenceMatcher(None, qa[i]["question"].lower(),
                                    qa[j]["question"].lower()).ratio()
            # Similar enough -> print both questions (first 66 characters) and mark the later one as a duplicate.
            if ratio >= DUPLICATE_THRESHOLD:
                print("  %s ~ %s  (similarity %.2f)" % (qa[i]["id"], qa[j]["id"], ratio))
                print("     %s" % qa[i]["question"][:66])
                print("     %s" % qa[j]["question"][:66])
                dupes.add(qa[j]["id"])            # keep the first, drop the later
                problems.append((qa[j]["id"], "near-duplicate of %s" % qa[i]["id"]))
    # No pair crossed the threshold.
    if not dupes:
        print("  none found")

    # ---- Check 3: questions that leak where the answer is ----
    print("\n" + "=" * 74)
    print("3. LEAKED GROUND TRUTH")
    print("=" * 74)
    # Count of leaky questions.
    leaked = 0
    # Loop over every question.
    for q in qa:
        body = q["question"].lower()
        # Regex, three alternatives separated by "|":
        #   \bpage \d+  = the word "page", a space, then one or more digits ("page 12"); \b = start of a word.
        #   according to the (text|passage|document)  = that phrase ending in any one of the three words.
        #   this page  = the literal words "this page".
        # Any of these means the question points at its source instead of asking about the fact.
        if re.search(r"\bpage \d+|according to the (text|passage|document)|this page", body):
            print("  %s  %s" % (q["id"], q["question"][:66]))
            problems.append((q["id"], "references the source rather than asking"))
            leaked += 1
    # No leaky questions.
    if not leaked:
        print("  none found")

    # ---- Summary: total questions, total problems, and each problem on its own line ----
    print("\n" + "=" * 74)
    print("SUMMARY: %d questions, %d problems" % (len(qa), len(problems)))
    # Print each recorded problem.
    for qid, why in problems:
        print("  %-6s %s" % (qid, why))
    print("=" * 74)

    # With --fix and at least one duplicate: keep every question that is not a duplicate and save the file.
    # indent=2 makes the JSON readable; ensure_ascii=False keeps any non-English characters as they are.
    if fix and dupes:
        kept = [q for q in qa if q["id"] not in dupes]
        config.QA_SET_PATH.write_text(
            json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n--fix: dropped %d duplicate(s), %d questions remain"
              % (len(dupes), len(kept)))
    # Duplicates found but --fix not given: just tell the user how to remove them.
    elif dupes:
        print("\nRe-run with --fix to drop duplicates.")


# Run main() only when this file is started directly, not when imported.
if __name__ == "__main__":
    main()
