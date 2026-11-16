"""
eval/verify/verify_judge.py
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
# WHAT THIS FILE IS: tests the judge model (gpt-oss-120b, grades answers CORRECT/PARTIAL/INCORRECT/REFUSED)
# on 10 answers whose right grade we already know, each graded twice.
# Example: "three categories of data that feed a data warehouse?" answered "structured, unstructured, streaming"
#          must be INCORRECT. README: judge accuracy 10/10, consistency 10/10.
# Flow: build 10 cases from qa_set.json -> grade each 2x -> count accuracy + consistency -> print report.

import json      # read eval/data/qa_set.json
import sys       # sys.path for imports
import time      # pause between judge calls (rate limit)
import pathlib   # this file's folder

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # add eval/: finds config.py + run_eval.py
import config    # eval/config.py: JUDGE_MODEL, token limits, QA_SET_PATH

config.load_env()  # load API keys from .env (stops if missing)

from langchain_groq import ChatGroq  # Groq client (runs gpt-oss models)
import run_eval                      # reuse its judge(): test the EXACT real grader

TRIALS = 2  # grade each case twice (minimum to spot self-disagreement)


# ── BLOCK: test cases ───────────────────────────────────────────────────
# IN: eval/data/qa_set.json -> OUT: 10 case dicts (name, expect, question, reference, answer)
# e.g. {"name": "exact/q001", "expect": "CORRECT", "answer": <same as reference>}
def build_cases():
    """Cases with known-correct grades, drawn from the real question set."""
    qa = [q for q in json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))
          if q["answerable"]]                              # skip probes (no answer in PDF)
    cases = []                                             # filled with 6 kinds below

    # 1. The reference answer itself must grade CORRECT.
    for q in qa[:4]:                                       # first 4 questions
        cases.append({
            "name": "exact/%s" % q["id"], "expect": "CORRECT",       # e.g. "exact/q001"
            "question": q["question"], "reference": q["reference_answer"],
            "answer": q["reference_answer"],               # answer = reference
        })

    # 2. A reference answer reworded must still grade CORRECT.
    cases.append({                                         # tests meaning, not exact words
        "name": "reworded", "expect": "CORRECT",
        "question": "What are the three categories of data that feed a data warehouse?",
        "reference": "Internal data, external data, and personal data.",
        "answer": "The document lists three sources: data held internally, data "
                  "sourced externally, and personal data belonging to individuals.",  # same meaning
    })

    # 3. Another question's answer must grade INCORRECT.
    for q, other in [(qa[0], qa[5]), (qa[1], qa[7])]:      # q0 gets q5's answer, q1 gets q7's
        cases.append({
            "name": "mismatched/%s" % q["id"], "expect": "INCORRECT",
            "question": q["question"], "reference": q["reference_answer"],
            "answer": other["reference_answer"],           # wrong question's answer
        })

    # 4. A flatly contradictory answer must grade INCORRECT.
    cases.append({
        "name": "contradiction", "expect": "INCORRECT",
        "question": "What are the three categories of data that feed a data warehouse?",
        "reference": "Internal data, external data, and personal data.",
        "answer": "The three categories are structured data, unstructured data "
                  "and streaming data.",                   # wrong three categories
    })

    # 5. A refusal must grade REFUSED.
    cases.append({
        "name": "refusal", "expect": "REFUSED",
        "question": qa[0]["question"], "reference": qa[0]["reference_answer"],
        "answer": "I could not find this information in the uploaded document.",  # app's exact refusal line
    })

    # 6. A half-answer must NOT grade CORRECT (PARTIAL or INCORRECT both fine).
    cases.append({
        "name": "half-answer", "expect": "NOT_CORRECT",    # special: anything but CORRECT passes
        "question": "What are the three categories of data that feed a data warehouse?",
        "reference": "Internal data, external data, and personal data.",
        "answer": "One of the categories mentioned is internal data.",  # 1 of 3 categories
    })

    return cases                                           # 4+1+2+1+1+1 = 10 cases


# ── BLOCK: main ─────────────────────────────────────────────────────────
# IN: nothing -> OUT: PASS/FAIL per case + accuracy + consistency
# e.g. "[PASS] refusal  expected REFUSED  got ['REFUSED', 'REFUSED']"
def main():
    # temperature 0 = most repeatable; max 512 tokens (token = ~3-4 chars of text)
    judge_llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0,
                         max_tokens=config.JUDGE_MAX_TOKENS)
    cases = build_cases()                                  # the 10 cases
    print("=" * 78)
    print("JUDGE VALIDATION  (%d cases x %d trials, model=%s)"
          % (len(cases), TRIALS, config.JUDGE_MODEL))      # header line
    print("=" * 78)

    correct = 0                                            # graded as expected
    consistent = 0                                         # same grade both trials
    for c in cases:                                        # each case
        # ── Step 1: grade it twice ──
        grades = []                                        # one grade per trial
        for _ in range(TRIALS):                            # 2 trials
            g = run_eval.judge(judge_llm, c["question"], c["reference"], c["answer"])  # {"grade", "why"}
            grades.append(g["grade"])                      # e.g. "CORRECT"
            time.sleep(1.0)                                # free-tier rate limit

        # ── Step 2: pass? stable? ──
        if c["expect"] == "NOT_CORRECT":                   # half-answer case
            ok = all(g != "CORRECT" for g in grades)       # never CORRECT
        else:
            ok = all(g == c["expect"] for g in grades)     # every trial matches
        same = len(set(grades)) == 1                       # both trials agree

        correct += bool(ok)                                # True adds 1
        consistent += bool(same)
        print("  [%s]%s %-22s expected %-11s got %s"
              % ("PASS" if ok else "FAIL", "" if same else " (UNSTABLE)",  # UNSTABLE = trials disagreed
                 c["name"], c["expect"], grades))

    # ── Step 3: totals ──
    n = len(cases)                                         # 10
    print("\n" + "=" * 78)
    print("  accuracy    : %d/%d (%.0f%%)  -- graded as expected" % (correct, n, 100.0 * correct / n))
    print("  consistency : %d/%d (%.0f%%)  -- same grade on both trials" % (consistent, n, 100.0 * consistent / n))
    if correct < n or consistent < n:                      # any fail = warn about noise
        print("\n  Accuracy figures inherit this error. Differences between")
        print("  configurations smaller than the judge's own noise mean nothing.")
    print("=" * 78)


if __name__ == "__main__":  # only when run directly
    main()                  # start
