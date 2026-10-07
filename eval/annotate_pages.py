"""
eval/annotate_pages.py
─────────────────────────────────────────────────────────────────────────────
Writes an `acceptable_pages` list into every answerable question in qa_set.json.

Why: recall@k originally credited only the page a question was generated from.
Some answers genuinely appear on several pages, so that scored real successes as
misses. This records every page that truly contains the answer, letting
run_eval.py report recall strictly (originating page only) and leniently (any
page containing the answer).

Uses no API -- pure text matching -- so it runs with the embedding quota dead.
"""
# WHAT THIS FILE IS: the "answer-key checker" of the evaluation. A question was written from ONE page, but the
# same fact can be printed on other pages too. This script finds ALL pages that really hold the answer.
# Real example: q001 "What SQL command is shown for creating the example database?" has the answer
# `CREATE DATABASE startersql;`, which sits on page index 4 AND page index 65 of the MySQL handbook
# (0-based, the way PyPDFLoader counts). So q001 gets "acceptable_pages": [4, 65] in qa_set.json.
# Why it matters: "recall@k" = did any of the top k retrieved chunks come from a correct page? If only page 4
# counted, a retriever that returned page 65 would be punished for a correct find.
# Overall flow: read qa_set.json -> load every PDF page as lowercase text -> for each answerable question,
# config.acceptable_pages() lists pages containing all its key answer words -> write the list back into the JSON.
#
# json = read and write JSON. JSON is a plain-text format for lists and dicts ({"id": "q001", ...}); qa_set.json uses it.
import json
# sys = Python's system module: sys.argv (command-line words) and sys.path (where Python looks for imports).
import sys
# pathlib = easy file paths (Path objects: .name, .read_text(), .write_text()).
import pathlib

# Add this eval/ folder to the import search path, so "import config" finds eval/config.py
# even when the script is started from the project root (py -3.12 eval/annotate_pages.py).
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# config = eval/config.py: the list of PDFs (DOCUMENTS), the default question file path and acceptable_pages().
import config

# PyPDFLoader (from LangChain) reads a PDF and returns one Document per page; .page_content is the page text.
# It is the same loader the live app uses, so page numbering here matches the app's page numbering.
from langchain_community.document_loaders import PyPDFLoader


# IN: optional path to a question file on the command line -> OUT: the same file, rewritten with "acceptable_pages"
# added to every question, plus a printed count of questions that have more than one valid page.
# WHY: lets run_eval.py report recall two ways: strict (only the original page) and lenient (any acceptable page).
# Example: py -3.12 eval/annotate_pages.py eval/qa_set_hard.json -> annotates the hard set instead of qa_set.json.
def main():
    # Optional path argument so the hard set can be annotated too:
    #   py -3.12 eval/annotate_pages.py eval/qa_set_hard.json
    # Keep only command-line words that are not flags (flags start with "-"). The first one, if any, is the file path.
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    # No path given -> use the default easy set eval/qa_set.json (config.QA_SET_PATH).
    qa_path = pathlib.Path(args[0]) if args else config.QA_SET_PATH
    print("annotating %s" % qa_path.name)
    # Load the list of question dicts from the JSON file.
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    # Build {doc_id: [page 0 text, page 1 text, ...]} with every page in lowercase, so word matching ignores case.
    # This is a "dict comprehension": a one-line loop that builds a dictionary.
    pages_lower = {
        doc_id: [p.page_content.lower() for p in PyPDFLoader(meta["path"]).load()]
        for doc_id, meta in config.DOCUMENTS.items()
    }

    # Counter: how many answerable questions have more than one acceptable page.
    multi = 0
    # Go through every question in the set.
    for q in qa:
        # Unanswerable question (a "probe": a trap question whose answer is NOT in the document) -> no page can be
        # correct, so store an empty list and skip to the next question.
        if not q["answerable"]:
            q["acceptable_pages"] = []
            continue
        # Pages whose text contains ALL the distinctive words of the reference answer (at least 3 such words are needed,
        # otherwise only the original page is kept). The original page gt_page is always included.
        acc = config.acceptable_pages(
            q["reference_answer"], pages_lower[q["doc_id"]], q["gt_page"])
        q["acceptable_pages"] = acc
        # More than one valid page -> count it and print the extra pages, so a human can see them.
        if len(acc) > 1:
            multi += 1
            print("  %s  gt=%-3s also answerable from %s"
                  % (q["id"], q["gt_page"], [p for p in acc if p != q["gt_page"]]))

    # Save the whole set back to the same file. indent=2 = pretty, readable JSON. ensure_ascii=False keeps
    # non-English characters as they are instead of \u escape codes.
    qa_path.write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print the share of answerable questions with several valid pages. eval/README.md reports 18.6% for the easy set.
    answerable = sum(1 for q in qa if q["answerable"])
    print("\n%d/%d answerable questions (%.1f%%) have more than one valid page."
          % (multi, answerable, 100.0 * multi / answerable))
    print("Strict recall counts only the originating page; lenient counts any of these.")


# Run main() only when this file is started directly, not when another file imports it.
if __name__ == "__main__":
    main()
