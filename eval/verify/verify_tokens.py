"""
eval/verify/verify_tokens.py
─────────────────────────────────────────────────────────────────────────────
Tests the claim: "max_tokens=1024 in documind/chain.py can return a BLANK answer."

The claim started from a single observation -- one question in one run came back
empty with finish_reason='length'. One event is an anecdote, not a bug report.
This measures the distribution instead.

Method
------
For several questions, build a realistic 4-chunk context containing the
answer-bearing chunk, then generate repeatedly at each token ceiling. Record:

  completion_tokens  -- hidden reasoning plus visible answer
  finish_reason      -- 'length' means the ceiling was hit
  visible chars      -- 0 means the user sees an empty reply

If the ceiling is genuinely too low, empty answers appear at 1024 and disappear
at a higher ceiling. If they do not, the claim is wrong and should be retracted.

Needs no embedding API -- context is assembled from the cached chunk files.
"""
# WHAT THIS FILE IS: a "stress test" for one claim. Like filling a cup again and again to see if 1024 ml is really
# too small, it asks the real answer model (gpt-oss-20b on Groq) the same questions many times with three
# different output limits (max_tokens 1024, 2048, 4096) and counts how often the answer comes back empty.
# A TOKEN = a small piece of text (about 3-4 characters of English); models count input and output in tokens.
# max_tokens = the most tokens the model may write in one reply. gpt-oss-20b is a REASONING model: it first
# writes hidden "thinking" tokens, then the visible answer, and both count against max_tokens.
# finish_reason = the reason the model stopped: "stop" (it finished) or "length" (it hit max_tokens).
# Real example: question q018 once came back empty with finish_reason="length" in the baseline run. eval/README.md
# reports what this test found: 0 empty answers in 20 trials at 1024, median 170 and max 391 completion tokens.
# Overall flow: for 4 questions -> build a 4-chunk context that contains the answer -> for each ceiling, generate
# 5 times -> count empty replies and "length" stops -> print a table -> print a verdict on the claim.
#
# json: reads the question set and the saved chunk files.
import json
# statistics: Python's built-in maths helpers; we use statistics.median.
# MEDIAN = the middle value after sorting (half the values are below it), so one huge outlier does not move it.
import statistics
# sys: changes the import path.
import sys
# pathlib: finds the folder this file lives in.
import pathlib

# Put eval/ first on Python's search path, so "import config" finds eval/config.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
# config = eval/config.py: model names, folders, index_key() and answer_key_terms().
import config

# Load GOOGLE_API_KEY and GROQ_API_KEY from .env (only the Groq key is really used by this script).
config.load_env()
# Also put the project root on the path, so "from documind import ..." finds the live app's package.
sys.path.insert(0, str(config.PROJECT_DIR))

# Import the app's REAL system prompt and context formatter (read-only), so the test uses exactly what users get.
from documind import SYSTEM_PROMPT, format_docs
# ChatGroq: LangChain's client for Groq (the cloud service that runs gpt-oss-20b).
from langchain_groq import ChatGroq
# ChatPromptTemplate: builds the chat messages from a template. MessagesPlaceholder: a slot for the chat history.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# TRIALS = 5: each question is asked 5 times per ceiling, because one try can hide a rare failure.
TRIALS = 5
# CEILINGS: the 3 max_tokens values compared. 1024 is what documind/settings.py ships; 2048 and 4096 are 2x and 4x it.
CEILINGS = (1024, 2048, 4096)
# The 4 questions used. q018 is the one that came back empty in the baseline run; the others are its neighbours
# in the set plus q003. 4 x 5 x 3 = 60 generations in total.
QUESTION_IDS = ("q018", "q019", "q020", "q003")


# IN: one saved chunk dict {"text": ..., "page": ...} -> OUT: a tiny object with .page_content and .metadata.
# WHY: format_docs() in documind/chain.py expects LangChain Document-like objects; this is the smallest stand-in.
# Example: _D({"text": "CREATE DATABASE startersql;", "page": 4}) -> .page_content "CREATE DATABASE startersql;", .metadata {"page": 4}.
class _D:
    def __init__(self, chunk):
        self.page_content = chunk["text"]
        self.metadata = {"page": chunk["page"]}


# IN: one question dict -> OUT: (context string in the app's format, match score of the best chunk, number of terms).
# WHY: the test must look like real use: 4 chunks, one of which holds the answer, formatted as the app does.
# Example: for q018 -> ("[Page ...]\n...\n\n---\n\n[Page ...]\n...", best score, number of answer terms).
def build_context(q):
    """A realistic 4-chunk context that CONTAINS the answer."""
    # Use the baseline chunk file (1000 characters, 200 overlap) - no embedding API needed, the chunks are saved as JSON.
    key = config.index_key(q["doc_id"], 1000, 200)
    path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")
    chunks = json.loads(path.read_text(encoding="utf-8"))

    # terms = up to 6 distinctive words from the reference answer (long words, numbers, identifiers).
    terms = config.answer_key_terms(q["reference_answer"])
    # Find the chunk that best matches the answer. Start with "nothing found" (index None, score -1).
    best, best_score = None, -1
    # Loop over every chunk with its position i.
    for i, c in enumerate(chunks):
        low = c["text"].lower()
        # Score = how many answer terms appear in the chunk (True counts as 1), plus a bonus of 2 if the chunk is on the
        # question's own page. The bonus of 2 makes the right page win over a chunk elsewhere that shares a term or two.
        score = sum(t in low for t in terms) + (2 if c["page"] == q["gt_page"] else 0)
        # Keep the highest score seen so far.
        if score > best_score:
            best, best_score = i, score

    # The best chunk first, then its neighbours (next, previous, the one after next) that exist inside the list,
    # at most 3 of them -> up to 4 chunks, matching TOP_K_RESULTS = 4 in the app.
    idxs = [best] + [i for i in (best + 1, best - 1, best + 2)
                     if 0 <= i < len(chunks)][:3]
    return format_docs([_D(chunks[i]) for i in idxs]), best_score, len(terms)


# IN: nothing -> OUT: per-question lines, a totals table per ceiling, and a verdict on the "1024 is too low" claim.
# WHY: one observation is an anecdote; 60 generations give a distribution.
# Example verdict (from eval/README.md's results, 0 empty at 1024): "NOT REPRODUCED: no blank answers at 1024 in 20 trials."
def main():
    # Questions as a dictionary: id -> question, so we can look up "q018" directly.
    qa = {q["id"]: q for q in json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))}
    # The same prompt shape as the app: system prompt (with {context}), chat history slot, then the question.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    # Header, e.g. "(4 questions x 5 trials x 3 ceilings = 60 calls)".
    print("=" * 84)
    print("TOKEN CEILING TEST  (%d questions x %d trials x %d ceilings = %d calls)"
          % (len(QUESTION_IDS), TRIALS, len(CEILINGS),
             len(QUESTION_IDS) * TRIALS * len(CEILINGS)))
    print("=" * 84)

    # totals per ceiling: empty replies, "length" stops, number of trials, and every completion-token count.
    totals = {c: {"empty": 0, "length": 0, "n": 0, "tokens": []} for c in CEILINGS}

    # Loop over the chosen questions.
    for qid in QUESTION_IDS:
        q = qa.get(qid)
        # Skip an id that is missing from the set or is a probe (unanswerable).
        if not q or not q["answerable"]:
            continue
        # Build the context and fill the prompt. chat_history=[] = a fresh conversation, no earlier turns.
        context, score, nterms = build_context(q)
        msgs = prompt.format_messages(context=context, chat_history=[],
                                      question=q["question"])
        # Show the context size and the best chunk's score out of the number of terms (the score includes the +2 page bonus, so it can exceed the term count).
        print("\n%s  (context %d chars, answer-term match %d/%d)"
              % (qid, len(context), score, nterms))

        # Loop over the three ceilings.
        for ceiling in CEILINGS:
            # A new model client with this ceiling. temperature=0.1 = same setting as the live app (a little randomness).
            llm = ChatGroq(model=config.GEN_MODEL, temperature=0.1, max_tokens=ceiling)
            # Per-ceiling counters for this question.
            toks, empties, lengths = [], 0, 0
            # Ask the same thing TRIALS (5) times.
            for _ in range(TRIALS):
                # One generation. response_metadata holds Groq's extra info (token usage, finish_reason); {} if missing.
                r = llm.invoke(msgs)
                m = r.response_metadata or {}
                # completion_tokens = tokens the model wrote (hidden reasoning + visible answer). 0 if the field is missing.
                ct = (m.get("token_usage") or {}).get("completion_tokens") or 0
                toks.append(ct)
                # Empty or whitespace-only visible text = the user would see a blank reply.
                if not r.content.strip():
                    empties += 1
                # finish_reason "length" = the model was cut off by max_tokens.
                if m.get("finish_reason") == "length":
                    lengths += 1

            # Add this question's numbers into the totals for this ceiling.
            t = totals[ceiling]
            t["empty"] += empties
            t["length"] += lengths
            t["n"] += TRIALS
            t["tokens"].extend(toks)

            # One line per ceiling: smallest / median / largest token count, empty count, cut-off count.
            print("   max_tokens=%-5d tokens min/med/max %4d/%4d/%4d   empty %d/%d   hit-ceiling %d/%d"
                  % (ceiling, min(toks), int(statistics.median(toks)), max(toks),
                     empties, TRIALS, lengths, TRIALS))

    # Final table: one row per ceiling across all questions.
    print("\n" + "=" * 84)
    print("%-12s %14s %14s %16s" % ("CEILING", "EMPTY ANSWERS", "HIT CEILING", "TOKENS med/max"))
    print("-" * 84)
    # Loop over the ceilings and print each row.
    for c in CEILINGS:
        t = totals[c]
        print("%-12d %8d/%-5d %8d/%-5d %10d/%d"
              % (c, t["empty"], t["n"], t["length"], t["n"],
                 int(statistics.median(t["tokens"])), max(t["tokens"])))
    print("=" * 84)

    # Compare the lowest ceiling (1024) with the highest (4096) to judge the claim.
    base = totals[CEILINGS[0]]
    high = totals[CEILINGS[-1]]
    # Case 1: blanks at 1024 but none at 4096 -> the claim is confirmed.
    if base["empty"] > 0 and high["empty"] == 0:
        print("CONFIRMED: blank answers occur at %d and vanish at %d."
              % (CEILINGS[0], CEILINGS[-1]))
    # Case 2: no blanks at 1024 at all -> the claim is not reproduced (this is what eval/README.md reports).
    elif base["empty"] == 0:
        print("NOT REPRODUCED: no blank answers at %d in %d trials."
              % (CEILINGS[0], base["n"]))
        print("The original empty answer stands as a real but RARE event --")
        print("report it as observed once, not as a reliable failure mode.")
    # Case 3: blanks even at 4096 -> raising the limit is not a full fix; the app must handle empty replies itself
    # (documind.chain.run_qa now retries once on an empty answer).
    else:
        print("MIXED: blank answers occur at both ceilings. Raising the ceiling")
        print("is not a complete fix; handle empty content explicitly.")


# Run main() only when this file is started directly, not when imported.
if __name__ == "__main__":
    main()
