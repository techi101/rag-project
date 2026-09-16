"""
eval/verify_tokens.py
─────────────────────────────────────────────────────────────────────────────
Tests the claim: "max_tokens=1024 in rag_engine.py can return a BLANK answer."

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
import json
import statistics
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

config.load_env()
sys.path.insert(0, str(config.PROJECT_DIR))

from rag_engine import SYSTEM_PROMPT, format_docs
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

TRIALS = 5
CEILINGS = (1024, 2048, 4096)
QUESTION_IDS = ("q018", "q019", "q020", "q003")


class _D:
    def __init__(self, chunk):
        self.page_content = chunk["text"]
        self.metadata = {"page": chunk["page"]}


def build_context(q):
    """A realistic 4-chunk context that CONTAINS the answer."""
    key = config.index_key(q["doc_id"], 1000, 200)
    path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")
    chunks = json.loads(path.read_text(encoding="utf-8"))

    terms = config.answer_key_terms(q["reference_answer"])
    best, best_score = None, -1
    for i, c in enumerate(chunks):
        low = c["text"].lower()
        score = sum(t in low for t in terms) + (2 if c["page"] == q["gt_page"] else 0)
        if score > best_score:
            best, best_score = i, score

    idxs = [best] + [i for i in (best + 1, best - 1, best + 2)
                     if 0 <= i < len(chunks)][:3]
    return format_docs([_D(chunks[i]) for i in idxs]), best_score, len(terms)


def main():
    qa = {q["id"]: q for q in json.loads(config.QA_SET_PATH.read_text(encoding="utf-8"))}
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    print("=" * 84)
    print("TOKEN CEILING TEST  (%d questions x %d trials x %d ceilings = %d calls)"
          % (len(QUESTION_IDS), TRIALS, len(CEILINGS),
             len(QUESTION_IDS) * TRIALS * len(CEILINGS)))
    print("=" * 84)

    totals = {c: {"empty": 0, "length": 0, "n": 0, "tokens": []} for c in CEILINGS}

    for qid in QUESTION_IDS:
        q = qa.get(qid)
        if not q or not q["answerable"]:
            continue
        context, score, nterms = build_context(q)
        msgs = prompt.format_messages(context=context, chat_history=[],
                                      question=q["question"])
        print("\n%s  (context %d chars, answer-term match %d/%d)"
              % (qid, len(context), score, nterms))

        for ceiling in CEILINGS:
            llm = ChatGroq(model=config.GEN_MODEL, temperature=0.1, max_tokens=ceiling)
            toks, empties, lengths = [], 0, 0
            for _ in range(TRIALS):
                r = llm.invoke(msgs)
                m = r.response_metadata or {}
                ct = (m.get("token_usage") or {}).get("completion_tokens") or 0
                toks.append(ct)
                if not r.content.strip():
                    empties += 1
                if m.get("finish_reason") == "length":
                    lengths += 1

            t = totals[ceiling]
            t["empty"] += empties
            t["length"] += lengths
            t["n"] += TRIALS
            t["tokens"].extend(toks)

            print("   max_tokens=%-5d tokens min/med/max %4d/%4d/%4d   empty %d/%d   hit-ceiling %d/%d"
                  % (ceiling, min(toks), int(statistics.median(toks)), max(toks),
                     empties, TRIALS, lengths, TRIALS))

    print("\n" + "=" * 84)
    print("%-12s %14s %14s %16s" % ("CEILING", "EMPTY ANSWERS", "HIT CEILING", "TOKENS med/max"))
    print("-" * 84)
    for c in CEILINGS:
        t = totals[c]
        print("%-12d %8d/%-5d %8d/%-5d %10d/%d"
              % (c, t["empty"], t["n"], t["length"], t["n"],
                 int(statistics.median(t["tokens"])), max(t["tokens"])))
    print("=" * 84)

    base = totals[CEILINGS[0]]
    high = totals[CEILINGS[-1]]
    if base["empty"] > 0 and high["empty"] == 0:
        print("CONFIRMED: blank answers occur at %d and vanish at %d."
              % (CEILINGS[0], CEILINGS[-1]))
    elif base["empty"] == 0:
        print("NOT REPRODUCED: no blank answers at %d in %d trials."
              % (CEILINGS[0], base["n"]))
        print("The original empty answer stands as a real but RARE event --")
        print("report it as observed once, not as a reliable failure mode.")
    else:
        print("MIXED: blank answers occur at both ceilings. Raising the ceiling")
        print("is not a complete fix; handle empty content explicitly.")


if __name__ == "__main__":
    main()
