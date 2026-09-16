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
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

from langchain_community.document_loaders import PyPDFLoader


def main():
    # Optional path argument so the hard set can be annotated too:
    #   py -3.12 eval/annotate_pages.py eval/qa_set_hard.json
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    qa_path = pathlib.Path(args[0]) if args else config.QA_SET_PATH
    print("annotating %s" % qa_path.name)
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    pages_lower = {
        doc_id: [p.page_content.lower() for p in PyPDFLoader(meta["path"]).load()]
        for doc_id, meta in config.DOCUMENTS.items()
    }

    multi = 0
    for q in qa:
        if not q["answerable"]:
            q["acceptable_pages"] = []
            continue
        acc = config.acceptable_pages(
            q["reference_answer"], pages_lower[q["doc_id"]], q["gt_page"])
        q["acceptable_pages"] = acc
        if len(acc) > 1:
            multi += 1
            print("  %s  gt=%-3s also answerable from %s"
                  % (q["id"], q["gt_page"], [p for p in acc if p != q["gt_page"]]))

    qa_path.write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    answerable = sum(1 for q in qa if q["answerable"])
    print("\n%d/%d answerable questions (%.1f%%) have more than one valid page."
          % (multi, answerable, 100.0 * multi / answerable))
    print("Strict recall counts only the originating page; lenient counts any of these.")


if __name__ == "__main__":
    main()
