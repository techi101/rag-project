"""
eval/verify_judge.py
─────────────────────────────────────────────────────────────────────────────
Validates the grader before trusting any accuracy number.

Every "answer accuracy" figure this project reports comes from gpt-oss-120b
grading gpt-oss-20b. If the grader is wrong or unstable, the accuracy number is
decoration. Two things get measured:

  ACCURACY    -- cases with a known-correct grade, built from the real question
                 set. A correct answer must grade CORRECT; an answer taken from
                 a DIFFERENT question must grade INCORRECT; a refusal must grade
                 REFUSED; a hedged half-answer must not grade CORRECT.

  CONSISTENCY -- every case graded twice. Disagreement between two runs of the
                 same input is noise, and it bounds how small a difference
                 between configurations can mean anything.

Uses no embedding API, so it runs with the Google quota exhausted.
"""
import json
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

config.load_env()

from langchain_groq import ChatGroq
import run_eval

TRIALS = 2


def build_cases():
    """Cases with known-correct grades, drawn from the real question set."""
    qa = [q for q in json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))
          if q["answerable"]]
    cases = []

    # 1. The reference answer itself must grade CORRECT.
    for q in qa[:4]:
        cases.append({
            "name": "exact/%s" % q["id"], "expect": "CORRECT",
            "question": q["question"], "reference": q["reference_answer"],
            "answer": q["reference_answer"],
        })

    # 2. A reference answer reworded must still grade CORRECT.
    cases.append({
        "name": "reworded", "expect": "CORRECT",
        "question": "What are the three categories of data that feed a data warehouse?",
        "reference": "Internal data, external data, and personal data.",
        "answer": "The document lists three sources: data held internally, data "
                  "sourced externally, and personal data belonging to individuals.",
    })

    # 3. Another question's answer must grade INCORRECT.
    for q, other in [(qa[0], qa[5]), (qa[1], qa[7])]:
        cases.append({
            "name": "mismatched/%s" % q["id"], "expect": "INCORRECT",
            "question": q["question"], "reference": q["reference_answer"],
            "answer": other["reference_answer"],
        })

    # 4. A flatly contradictory answer must grade INCORRECT.
    cases.append({
        "name": "contradiction", "expect": "INCORRECT",
        "question": "What are the three categories of data that feed a data warehouse?",
        "reference": "Internal data, external data, and personal data.",
        "answer": "The three categories are structured data, unstructured data "
                  "and streaming data.",
    })

    # 5. A refusal must grade REFUSED.
    cases.append({
        "name": "refusal", "expect": "REFUSED",
        "question": qa[0]["question"], "reference": qa[0]["reference_answer"],
        "answer": "I could not find this information in the uploaded document.",
    })

    # 6. A half-answer must NOT grade CORRECT (PARTIAL or INCORRECT both fine).
    cases.append({
        "name": "half-answer", "expect": "NOT_CORRECT",
        "question": "What are the three categories of data that feed a data warehouse?",
        "reference": "Internal data, external data, and personal data.",
        "answer": "One of the categories mentioned is internal data.",
    })

    return cases


def main():
    judge_llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0,
                         max_tokens=config.JUDGE_MAX_TOKENS)
    cases = build_cases()
    print("=" * 78)
    print("JUDGE VALIDATION  (%d cases x %d trials, model=%s)"
          % (len(cases), TRIALS, config.JUDGE_MODEL))
    print("=" * 78)

    correct = 0
    consistent = 0
    for c in cases:
        grades = []
        for _ in range(TRIALS):
            g = run_eval.judge(judge_llm, c["question"], c["reference"], c["answer"])
            grades.append(g["grade"])
            time.sleep(1.0)

        if c["expect"] == "NOT_CORRECT":
            ok = all(g != "CORRECT" for g in grades)
        else:
            ok = all(g == c["expect"] for g in grades)
        same = len(set(grades)) == 1

        correct += bool(ok)
        consistent += bool(same)
        print("  [%s]%s %-22s expected %-11s got %s"
              % ("PASS" if ok else "FAIL", "" if same else " (UNSTABLE)",
                 c["name"], c["expect"], grades))

    n = len(cases)
    print("\n" + "=" * 78)
    print("  accuracy    : %d/%d (%.0f%%)  -- graded as expected" % (correct, n, 100.0 * correct / n))
    print("  consistency : %d/%d (%.0f%%)  -- same grade on both trials" % (consistent, n, 100.0 * consistent / n))
    if correct < n or consistent < n:
        print("\n  Accuracy figures inherit this error. Differences between")
        print("  configurations smaller than the judge's own noise mean nothing.")
    print("=" * 78)


if __name__ == "__main__":
    main()
