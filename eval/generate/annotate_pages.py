"""
eval/generate/annotate_pages.py
─────────────────────────────────────────────────────────────────────────────
Writes an `acceptable_pages` list into every answerable question in qa_set.json.

Why: recall@k originally credited only the page a question was generated from.
Some answers genuinely appear on several pages, so that scored real successes as
misses. This records every page that truly contains the answer, letting
run_eval.py report recall strictly (originating page only) and leniently (any
page containing the answer).

Uses no API -- pure text matching -- so it runs with the embedding quota dead.
"""
# WHAT THIS FILE IS: finds EVERY page that holds a question's answer, not just the page it was written from.
# Example: q001 answer `CREATE DATABASE startersql;` is on pages 4 AND 65 (0-based) -> "acceptable_pages": [4, 65].
# recall@k = did any top k chunk come from a correct page? Without this, finding page 65 counts as a miss.
# Flow: read qa_set.json -> load PDF pages as lowercase text -> config.acceptable_pages() per question -> write back.

import json      # read/write the question file (JSON text)
import sys       # sys.argv = command-line words, sys.path = import folders
import pathlib   # file paths (Path objects)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # add eval/ so "import config" works
import config    # eval/config.py: DOCUMENTS, QA_SET_PATH, acceptable_pages()

from langchain_community.document_loaders import PyPDFLoader  # PDF -> one Document per page (same as the app)


# IN: optional question-file path on the command line -> OUT: that file rewritten with "acceptable_pages"
# e.g. py -3.12 eval/generate/annotate_pages.py eval/data/qa_set_hard.json
def main():
    # ── Step 1: pick the file and load it ──
    # Optional path argument so the hard set can be annotated too:
    #   py -3.12 eval/generate/annotate_pages.py eval/data/qa_set_hard.json
    args = [a for a in sys.argv[1:] if not a.startswith("-")]  # command-line words that are not flags
    qa_path = pathlib.Path(args[0]) if args else config.QA_SET_PATH  # default: eval/data/qa_set.json
    print("annotating %s" % qa_path.name)                       # show which file
    qa = json.loads(qa_path.read_text(encoding="utf-8"))        # list of question dicts

    # ── Step 2: every PDF page as lowercase text ──
    pages_lower = {                                             # {doc_id: [page 0 text, page 1 text, ...]}
        doc_id: [p.page_content.lower() for p in PyPDFLoader(meta["path"]).load()]  # lowercase = case-blind match
        for doc_id, meta in config.DOCUMENTS.items()            # one entry per PDF
    }

    # ── Step 3: find all valid pages per question ──
    multi = 0                                                   # questions with more than one valid page
    for q in qa:                                                # each question
        if not q["answerable"]:                                 # probe = trap question, answer not in the PDF
            q["acceptable_pages"] = []                          # no page can be correct
            continue                                            # next question
        # pages containing ALL distinctive answer words (needs at least 3, else only gt_page); gt_page always in
        acc = config.acceptable_pages(
            q["reference_answer"], pages_lower[q["doc_id"]], q["gt_page"])  # answer, page texts, original page
        q["acceptable_pages"] = acc                             # e.g. [4, 65]
        if len(acc) > 1:                                        # answer on several pages?
            multi += 1                                          # count it
            print("  %s  gt=%-3s also answerable from %s"
                  % (q["id"], q["gt_page"], [p for p in acc if p != q["gt_page"]]))  # show the extra pages

    # ── Step 4: save and report ──
    qa_path.write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")  # pretty JSON, keep non-English chars

    answerable = sum(1 for q in qa if q["answerable"])          # count of real questions
    print("\n%d/%d answerable questions (%.1f%%) have more than one valid page."
          % (multi, answerable, 100.0 * multi / answerable))    # eval/README.md: 18.6% for the easy set
    print("Strict recall counts only the originating page; lenient counts any of these.")


if __name__ == "__main__":  # only when run directly, not on import
    main()                  # start
