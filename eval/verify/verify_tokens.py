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
# WHAT THIS FILE IS: asks gpt-oss-20b the same questions many times at max_tokens 1024 / 2048 / 4096
# and counts empty answers. token = ~3-4 chars; reasoning model: hidden thinking + answer both count.
# finish_reason = why it stopped: "stop" (done) or "length" (hit max_tokens).
# Example: q018 once came back empty ("length"); README: 0 empty in 20 trials at 1024, median 170, max 391 tokens.
# Flow: 4 questions -> 4-chunk context with the answer -> 5 calls per ceiling -> count empties -> table -> verdict.

import json        # read question set + chunk files
import statistics  # median = middle value, ignores outliers
import sys         # sys.path for imports
import pathlib     # this file's folder

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # add eval/ so "import config" works
import config      # eval/config.py: models, folders, index_key(), answer_key_terms()

config.load_env()  # load API keys (only Groq used here)
# (config.py already put the project root on the import path, so "from documind import ..." works.)

from documind import SYSTEM_PROMPT, format_docs                   # app's REAL prompt + context formatter
from langchain_groq import ChatGroq                               # Groq client (gpt-oss-20b)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # chat template + history slot

# ── BLOCK: settings ─────────────────────────────────────────────────────
TRIALS = 5                                  # calls per question per ceiling
CEILINGS = (1024, 2048, 4096)               # 1024 = what documind/settings.py ships
QUESTION_IDS = ("q018", "q019", "q020", "q003")  # q018 = the empty one; 4 x 5 x 3 = 60 calls


# IN: chunk dict {"text", "page"} -> OUT: object with .page_content + .metadata (what format_docs expects)
# e.g. _D({"text": "CREATE DATABASE startersql;", "page": 4}).metadata -> {"page": 4}
class _D:
    def __init__(self, chunk):
        self.page_content = chunk["text"]                  # chunk text
        self.metadata = {"page": chunk["page"]}            # its page


# IN: question dict -> OUT: (context in app format, best chunk score, number of answer terms)
# e.g. q018 -> ("[Page ...]\n...\n\n---\n\n...", score, terms)
def build_context(q):
    """A realistic 4-chunk context that CONTAINS the answer."""
    # ── Step 1: load baseline chunks (no API) ──
    key = config.index_key(q["doc_id"], 1000, 200)         # e.g. "mysql_cs1000_co200"
    path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")
    chunks = json.loads(path.read_text(encoding="utf-8"))  # [{"page", "text"}, ...]

    # ── Step 2: find the chunk holding the answer ──
    terms = config.answer_key_terms(q["reference_answer"])  # up to 6 distinctive answer words
    best, best_score = None, -1                            # nothing found yet
    for i, c in enumerate(chunks):
        low = c["text"].lower()
        score = sum(t in low for t in terms) + (2 if c["page"] == q["gt_page"] else 0)  # terms found + 2 if right page
        if score > best_score:                             # new best
            best, best_score = i, score

    # ── Step 3: best chunk + up to 3 neighbours (= app's top 4) ──
    idxs = [best] + [i for i in (best + 1, best - 1, best + 2)
                     if 0 <= i < len(chunks)][:3]          # next, previous, after next, if they exist
    return format_docs([_D(chunks[i]) for i in idxs]), best_score, len(terms)  # app-format context


# ── BLOCK: main ─────────────────────────────────────────────────────────
# IN: nothing -> OUT: per-question lines, totals table, verdict on "1024 is too low"
# e.g. "NOT REPRODUCED: no blank answers at 1024 in 20 trials."
def main():
    qa = {q["id"]: q for q in json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))}  # id -> question
    prompt = ChatPromptTemplate.from_messages([            # same shape as the app
        ("system", SYSTEM_PROMPT),                         # rules + {context}
        MessagesPlaceholder("chat_history"),               # earlier turns slot
        ("human", "{question}"),                           # the question
    ])

    print("=" * 84)
    print("TOKEN CEILING TEST  (%d questions x %d trials x %d ceilings = %d calls)"
          % (len(QUESTION_IDS), TRIALS, len(CEILINGS),
             len(QUESTION_IDS) * TRIALS * len(CEILINGS)))  # e.g. 60 calls
    print("=" * 84)

    totals = {c: {"empty": 0, "length": 0, "n": 0, "tokens": []} for c in CEILINGS}  # per-ceiling counters

    # ── Step 1: run every question at every ceiling ──
    for qid in QUESTION_IDS:
        q = qa.get(qid)                                    # look up e.g. "q018"
        if not q or not q["answerable"]:                   # missing or probe
            continue
        context, score, nterms = build_context(q)          # 4-chunk context
        msgs = prompt.format_messages(context=context, chat_history=[],
                                      question=q["question"])  # fresh chat, no history
        # score includes the +2 page bonus, so it can exceed the term count
        print("\n%s  (context %d chars, answer-term match %d/%d)"
              % (qid, len(context), score, nterms))

        for ceiling in CEILINGS:                           # 1024, 2048, 4096
            llm = ChatGroq(model=config.GEN_MODEL, temperature=config.GEN_TEMPERATURE, max_tokens=ceiling)  # 0.1 = app
            toks, empties, lengths = [], 0, 0              # this question's counters
            for _ in range(TRIALS):                        # 5 calls
                r = llm.invoke(msgs)                       # one generation
                m = r.response_metadata or {}              # Groq extras: usage, finish_reason
                ct = (m.get("token_usage") or {}).get("completion_tokens") or 0  # reasoning + answer tokens
                toks.append(ct)
                if not r.content.strip():                  # blank visible reply
                    empties += 1
                if m.get("finish_reason") == "length":     # cut off by max_tokens
                    lengths += 1

            t = totals[ceiling]                            # add into totals
            t["empty"] += empties
            t["length"] += lengths
            t["n"] += TRIALS
            t["tokens"].extend(toks)

            print("   max_tokens=%-5d tokens min/med/max %4d/%4d/%4d   empty %d/%d   hit-ceiling %d/%d"
                  % (ceiling, min(toks), int(statistics.median(toks)), max(toks),
                     empties, TRIALS, lengths, TRIALS))    # one line per ceiling

    # ── Step 2: totals table ──
    print("\n" + "=" * 84)
    print("%-12s %14s %14s %16s" % ("CEILING", "EMPTY ANSWERS", "HIT CEILING", "TOKENS med/max"))
    print("-" * 84)
    for c in CEILINGS:                                     # one row per ceiling
        t = totals[c]
        print("%-12d %8d/%-5d %8d/%-5d %10d/%d"
              % (c, t["empty"], t["n"], t["length"], t["n"],
                 int(statistics.median(t["tokens"])), max(t["tokens"])))
    print("=" * 84)

    # ── Step 3: verdict, 1024 vs 4096 ──
    base = totals[CEILINGS[0]]                             # 1024
    high = totals[CEILINGS[-1]]                            # 4096
    if base["empty"] > 0 and high["empty"] == 0:           # blanks only at 1024
        print("CONFIRMED: blank answers occur at %d and vanish at %d."
              % (CEILINGS[0], CEILINGS[-1]))
    elif base["empty"] == 0:                               # none at 1024 (README result)
        print("NOT REPRODUCED: no blank answers at %d in %d trials."
              % (CEILINGS[0], base["n"]))
        print("The original empty answer stands as a real but RARE event --")
        print("report it as observed once, not as a reliable failure mode.")
    else:                                                  # blanks even at 4096 (run_qa now retries once)
        print("MIXED: blank answers occur at both ceilings. Raising the ceiling")
        print("is not a complete fix; handle empty content explicitly.")


if __name__ == "__main__":  # only when run directly
    main()                  # start
