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
# WHAT THIS FILE IS: the "exam for the examiner". Every accuracy number in this project is given by a JUDGE MODEL:
# a second, bigger LLM (gpt-oss-120b) that reads the app's answer and grades it CORRECT / PARTIAL / INCORRECT / REFUSED
# against a reference answer. Before trusting those grades, we test the judge on answers whose right grade we
# already know, and we grade each one twice to see if the judge gives the same grade both times.
# Real example from this file: question "What are the three categories of data that feed a data warehouse?",
# reference "Internal data, external data, and personal data."; the answer "...structured data, unstructured data
# and streaming data." must be graded INCORRECT. eval/README.md reports: judge accuracy 10/10, consistency 10/10.
# Overall flow: build 10 test cases from qa_set.json -> grade each case 2 times with the judge -> count how many
# matched the expected grade (accuracy) and how many got the same grade twice (consistency) -> print a report.
#
# json: reads the question set file eval/qa_set.json.
import json
# sys: changes the import path so "import config" and "import run_eval" work.
import sys
# time: time.sleep() pauses between judge calls to stay under Groq's free-tier rate limit.
import time
# pathlib: finds the folder this file lives in.
import pathlib

# Put the eval/ folder first on Python's search path, so the imports below find eval/config.py and eval/run_eval.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# config = eval/config.py: model names (JUDGE_MODEL), token limits, the path of qa_set.json.
import config

# Load GOOGLE_API_KEY and GROQ_API_KEY from .env (config.load_env stops with an error if either is missing).
config.load_env()

# ChatGroq: LangChain's client for Groq, the cloud service that runs the open gpt-oss models fast.
from langchain_groq import ChatGroq
# run_eval = eval/run_eval.py. We reuse its judge() function, so we test the EXACT grader the real evaluation uses.
import run_eval

# TRIALS = 2: each case is graded twice. Two runs are the minimum needed to see if the judge disagrees with itself.
TRIALS = 2


# IN: nothing (reads eval/qa_set.json) -> OUT: a list of 10 case dicts, each with name, expected grade,
#     question, reference answer and the answer to be graded.
# WHY: a grader can only be checked on answers whose correct grade we already know.
# Example: {"name": "exact/q001", "expect": "CORRECT", "question": "What SQL command is shown for creating the
#     example database?", "reference": "...CREATE DATABASE startersql;...", "answer": <the same reference>}.
def build_cases():
    """Cases with known-correct grades, drawn from the real question set."""
    # Keep only answerable questions (the set also has "probe" questions with no answer in the document).
    qa = [q for q in json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))
          if q["answerable"]]
    # The list we fill below with 6 kinds of cases.
    cases = []

    # 1. The reference answer itself must grade CORRECT.
    # Loop over the first 4 answerable questions: grading an answer against itself must give CORRECT.
    for q in qa[:4]:
        cases.append({
            "name": "exact/%s" % q["id"], "expect": "CORRECT",
            "question": q["question"], "reference": q["reference_answer"],
            "answer": q["reference_answer"],
        })

    # 2. A reference answer reworded must still grade CORRECT.
    # One hand-written case: same meaning, different words. Tests that the judge grades meaning, not exact wording.
    cases.append({
        "name": "reworded", "expect": "CORRECT",
        "question": "What are the three categories of data that feed a data warehouse?",
        "reference": "Internal data, external data, and personal data.",
        "answer": "The document lists three sources: data held internally, data "
                  "sourced externally, and personal data belonging to individuals.",
    })

    # 3. Another question's answer must grade INCORRECT.
    # Two pairs: question 0 gets question 5's answer, question 1 gets question 7's answer.
    # A real answer to a DIFFERENT question must be graded INCORRECT.
    for q, other in [(qa[0], qa[5]), (qa[1], qa[7])]:
        cases.append({
            "name": "mismatched/%s" % q["id"], "expect": "INCORRECT",
            "question": q["question"], "reference": q["reference_answer"],
            "answer": other["reference_answer"],
        })

    # 4. A flatly contradictory answer must grade INCORRECT.
    # Hand-written wrong answer for the data-warehouse question (wrong three categories).
    cases.append({
        "name": "contradiction", "expect": "INCORRECT",
        "question": "What are the three categories of data that feed a data warehouse?",
        "reference": "Internal data, external data, and personal data.",
        "answer": "The three categories are structured data, unstructured data "
                  "and streaming data.",
    })

    # 5. A refusal must grade REFUSED.
    # The exact refusal sentence the app's SYSTEM_PROMPT tells the model to use. Expected grade: REFUSED.
    cases.append({
        "name": "refusal", "expect": "REFUSED",
        "question": qa[0]["question"], "reference": qa[0]["reference_answer"],
        "answer": "I could not find this information in the uploaded document.",
    })

    # 6. A half-answer must NOT grade CORRECT (PARTIAL or INCORRECT both fine).
    # "NOT_CORRECT" is a special expectation used only here: PARTIAL or INCORRECT both pass, only CORRECT fails.
    # The answer names 1 of the 3 categories.
    cases.append({
        "name": "half-answer", "expect": "NOT_CORRECT",
        "question": "What are the three categories of data that feed a data warehouse?",
        "reference": "Internal data, external data, and personal data.",
        "answer": "One of the categories mentioned is internal data.",
    })

    # Total: 4 + 1 + 2 + 1 + 1 + 1 = 10 cases.
    return cases


# IN: nothing -> OUT: printed report (per-case PASS/FAIL, then accuracy and consistency totals).
# WHY: if the judge is wrong or unstable, every accuracy number in the project inherits that error.
# Example: a row "[PASS] refusal  expected REFUSED  got ['REFUSED', 'REFUSED']".
def main():
    # The judge model: gpt-oss-120b on Groq. temperature=0 = always pick the most likely next word, so grading is as
    # repeatable as possible. max_tokens = JUDGE_MAX_TOKENS (512) = the most tokens the judge may write.
    # A TOKEN is a small piece of text (about 3-4 characters of English) - models count input and output in tokens.
    judge_llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0,
                         max_tokens=config.JUDGE_MAX_TOKENS)
    cases = build_cases()
    # Header, e.g. "JUDGE VALIDATION  (10 cases x 2 trials, model=openai/gpt-oss-120b)".
    print("=" * 78)
    print("JUDGE VALIDATION  (%d cases x %d trials, model=%s)"
          % (len(cases), TRIALS, config.JUDGE_MODEL))
    print("=" * 78)

    # Counters: how many cases graded as expected, and how many got the same grade on both trials.
    correct = 0
    consistent = 0
    # Loop over every test case.
    for c in cases:
        # Collect the grade from each trial.
        grades = []
        # Grade the same case TRIALS (2) times. "_" means we do not need the loop counter.
        for _ in range(TRIALS):
            # run_eval.judge returns a dict like {"grade": "CORRECT", "why": "..."}; we keep only the grade.
            g = run_eval.judge(judge_llm, c["question"], c["reference"], c["answer"])
            grades.append(g["grade"])
            # Wait 1 second between calls so the free Groq tier does not reject us for asking too fast (rate limit).
            time.sleep(1.0)

        # Did the case pass? For "NOT_CORRECT", every trial must be anything except CORRECT.
        if c["expect"] == "NOT_CORRECT":
            ok = all(g != "CORRECT" for g in grades)
        # For all other cases, every trial must equal the expected grade exactly.
        else:
            ok = all(g == c["expect"] for g in grades)
        # Consistent = both trials gave the same grade (a set of the grades has only one element).
        same = len(set(grades)) == 1

        # bool True counts as 1, False as 0, so these lines add 1 for each passing / consistent case.
        correct += bool(ok)
        consistent += bool(same)
        # One line per case. "(UNSTABLE)" is added when the two trials disagreed.
        print("  [%s]%s %-22s expected %-11s got %s"
              % ("PASS" if ok else "FAIL", "" if same else " (UNSTABLE)",
                 c["name"], c["expect"], grades))

    # Print the totals as counts and percentages.
    n = len(cases)
    print("\n" + "=" * 78)
    print("  accuracy    : %d/%d (%.0f%%)  -- graded as expected" % (correct, n, 100.0 * correct / n))
    print("  consistency : %d/%d (%.0f%%)  -- same grade on both trials" % (consistent, n, 100.0 * consistent / n))
    # If anything failed or was unstable, warn that accuracy differences smaller than this noise are meaningless.
    if correct < n or consistent < n:
        print("\n  Accuracy figures inherit this error. Differences between")
        print("  configurations smaller than the judge's own noise mean nothing.")
    print("=" * 78)


# Run main() only when this file is started directly (py -3.12 eval/verify_judge.py).
if __name__ == "__main__":
    main()
