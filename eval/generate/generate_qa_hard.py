"""
eval/generate/generate_qa_hard.py
─────────────────────────────────────────────────────────────────────────────
Builds a HARDER question set that actually discriminates between retrievers.

Why this exists
---------------
The first set (generate_qa.py) has a measured 73% mean word overlap with its own
source pages, and BM25-only keyword search scored 97.7% recall@4 on it. A test a
1994 keyword algorithm cannot lose is not measuring semantic retrieval -- it is
measuring vocabulary reuse.

Method
------
Ask for questions that deliberately AVOID the page's wording: synonyms,
paraphrase, and a level of indirection that requires understanding the content
rather than matching strings. Then FILTER OBJECTIVELY -- any question whose
measured overlap exceeds MAX_OVERLAP is rejected and regenerated, up to
MAX_ATTEMPTS. The filter is the same function verify_method.py reports with, so
the generator cannot quietly grade itself on a different scale.

Output: eval/data/qa_set_hard.json

Uses Groq only (no embedding API), so it runs with the Google quota exhausted,
and BM25 scoring in verify_retrieval.py can test the result immediately.
"""
# WHAT THIS FILE IS: writes paraphrased questions so keyword search can't win by copying the page's words.
# Example: h001 "What is the sample name used for the new data container in the introductory instructions?"
#          -> "startersql", gt_page 4, lexical_overlap 0.333 (easy set: "What SQL command is shown for creating...").
# lexical overlap = share of question content words found on its page (config.question_page_overlap).
# README: easy 73% overlap, BM25 recall@4 97.7%; hard 21% overlap, BM25 recall@4 57.6%.
# Flow: pick 12 pages per PDF -> LLM writes Q -> measure overlap -> over 55%? retry with words to avoid -> save JSON.

import json      # write the output JSON
import random    # pick pages at random
import re        # regex: strip fences, split words
import sys       # sys.path for imports
import time      # waits between calls / retries
import hashlib   # sha256 for cache file names
import pathlib   # file paths

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # add eval/ so "import config" works
import config    # eval/config.py: PDFs, models, stopwords, question_page_overlap()

config.load_env()  # load API keys from .env (stops if missing)

from langchain_community.document_loaders import PyPDFLoader  # PDF -> one Document per page
from langchain_groq import ChatGroq                           # Groq chat client (gpt-oss-120b writes questions)

# ── BLOCK: settings ─────────────────────────────────────────────────────
QUESTIONS_PER_DOC = 12      # easy set used 15
MIN_PAGE_CHARS = 450        # skip title / near-empty pages
MAX_OVERLAP = 0.55          # reject anything more lexically leaky than this
MAX_ATTEMPTS = 3            # tries per page, then drop it
SEED = 20260916             # same seed as generate_qa.py, repeatable

HARD_CACHE = config.CACHE_DIR / "qa_hard"       # eval/cache/qa_hard: saved replies
HARD_CACHE.mkdir(parents=True, exist_ok=True)   # create if missing

OUT_PATH = config.QA_SET_HARD_PATH              # eval/data/qa_set_hard.json


# ── BLOCK: prompts ──────────────────────────────────────────────────────
# Paraphrased question. {retry_note} = "" first try; {{ }} = literal braces.
HARD_PROMPT = """You are building a DIFFICULT test set for a document-QA system.

Below is page {page_num} of "{doc_label}".

Write ONE question answerable only from this page, under a hard constraint:

  **Do NOT reuse the page's distinctive wording.**

  - Replace the page's technical terms with everyday synonyms or descriptions.
  - Describe concepts indirectly instead of naming them.
  - Never quote a phrase from the page.
  - A keyword search for your question's words should NOT obviously land here.

Example of the difference:
  TOO EASY : "What does the UPDATE statement do to the salary column?"
             (reuses UPDATE, salary -- both sit on the page)
  GOOD     : "How would someone change what an existing employee earns
             without creating a new record?"

The question must still have ONE specific, verifiable answer from this page, and
must read naturally, as a person unfamiliar with the document's exact phrasing
would ask it.
{retry_note}
Return STRICT JSON only, no prose:
{{"question": "...", "answer": "...", "fact_type": "definition|number|procedure|name|concept"}}

PAGE {page_num} TEXT:
{page_text}
"""

# Retry feedback: overlap, limit, shared words to avoid. {overlap:.0%} = 0.6 -> "60%".
RETRY_NOTE = """
IMPORTANT: your previous attempt reused too many of the page's own words
(overlap {overlap:.0%}, limit {limit:.0%}). These words appeared in BOTH your
question and the page: {shared}
Write a MORE INDIRECT question that avoids them entirely.
"""


# ── BLOCK: helpers ──────────────────────────────────────────────────────
# IN: llm, prompt, cache_key -> OUT: reply text, or "" after 5 fails
# e.g. key "hard::mysql::4::0::20260916" (doc, page, attempt, seed)
def call_llm(llm, prompt, cache_key):
    cache_file = HARD_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".txt")  # key -> file name
    if cache_file.exists():                                # cached?
        return cache_file.read_text(encoding="utf-8")      # reuse, no API call
    delay = 4.0                                            # 4, 8, 16, 32, 64 s
    for attempt in range(5):                               # up to 5 tries
        try:
            out = llm.invoke(prompt).content               # call the model
            cache_file.write_text(out, encoding="utf-8")   # save reply
            time.sleep(1.5)                                # stay under rate limit
            return out
        except Exception as e:                             # rate limit / network
            print("      retry (%s), sleeping %.0fs" % (type(e).__name__, delay))
            time.sleep(delay)                              # wait
            delay *= 2                                     # double next wait
    return ""                                              # all failed


# IN: raw reply -> OUT: dict or None
# e.g. '```json\n{"answer": "startersql"}\n```' -> {"answer": "startersql"}
def extract_json(text):
    if not text:                                           # empty reply
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()  # remove ``` fences
    s, e = text.find("{"), text.rfind("}")                 # first "{" to last "}"
    if s == -1 or e == -1:                                 # no braces
        return None
    try:
        return json.loads(text[s:e + 1])                   # parse
    except json.JSONDecodeError:                           # broken JSON
        return None


# IN: question, lowercase page text -> OUT: up to 10 shared words (sorted), fed into RETRY_NOTE
# e.g. "How would someone change what an employee earns?" -> ["employee", ...]
def shared_words(question, page_lower):
    words = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", question.lower())  # 3+ char words
             if w not in config.QUESTION_STOPWORDS}        # minus "what", "the", ...
    return sorted(w for w in words if w in page_lower)[:10]  # substring test on page, max 10


# ── BLOCK: main ─────────────────────────────────────────────────────────
# IN: PDFs in eval/data/documents.json -> OUT: eval/data/qa_set_hard.json + stats
# e.g. README: 33 answerable (42 with 9 probes added later)
def main():
    random.seed(SEED)                                      # same pages every run
    llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0.7, max_tokens=700)  # 0.7 = more variety than easy (0.4)

    out = []                                               # accepted questions
    qid = 0                                                # ids h001, h002, ...
    stats = {"accepted": 0, "rejected": 0, "attempts": 0}  # printed at the end

    for doc_id, meta in config.DOCUMENTS.items():          # each PDF
        # ── Step 1: pick pages ──
        print("\n[%s] %s" % (doc_id, meta["label"]))
        pages = [p for p in PyPDFLoader(meta["path"]).load()
                 if len(p.page_content.strip()) >= MIN_PAGE_CHARS]  # 450+ chars only
        chosen = random.sample(pages, min(QUESTIONS_PER_DOC, len(pages)))  # up to 12 pages
        chosen.sort(key=lambda p: p.metadata.get("page", 0))  # page order

        for p in chosen:
            page_idx = p.metadata.get("page", 0)           # 0-based page
            text = p.page_content.strip()[:6000]           # 6000 chars to the model
            page_lower = p.page_content.lower()            # full page, for overlap
            print("   page %3d" % (page_idx + 1), end=" ", flush=True)

            # ── Step 2: up to 3 tries, keep lowest overlap ──
            best = None                                    # best candidate so far
            for attempt in range(MAX_ATTEMPTS):
                stats["attempts"] += 1                     # count LLM tries
                note = ""                                  # no feedback first time
                if best is not None:                       # retry: tell it what leaked
                    note = RETRY_NOTE.format(overlap=best["overlap"], limit=MAX_OVERLAP,
                                             shared=", ".join(best["shared"]))
                raw = call_llm(llm, HARD_PROMPT.format(
                    page_num=page_idx + 1, doc_label=meta["label"],  # 1-based page, PDF name
                    page_text=text, retry_note=note),                # text + feedback
                    cache_key="hard::%s::%d::%d::%d" % (doc_id, page_idx, attempt, SEED))  # one cache per try
                parsed = extract_json(raw)                 # reply -> dict
                if not parsed or not parsed.get("question") or not parsed.get("answer"):  # unusable
                    continue

                ov = config.question_page_overlap(parsed["question"], page_lower)  # e.g. 0.333
                cand = {"parsed": parsed, "overlap": ov,
                        "shared": shared_words(parsed["question"], page_lower)}  # words to avoid next
                if best is None or ov < best["overlap"]:   # lower overlap wins
                    best = cand
                if ov <= MAX_OVERLAP:                      # good enough
                    break

            # ── Step 3: accept or drop ──
            if best is None:                               # all replies unusable
                print("SKIP (unparseable)")
                continue

            accepted = best["overlap"] <= MAX_OVERLAP      # within 55%?
            stats["accepted" if accepted else "rejected"] += 1  # count
            if not accepted:                               # still too leaky
                print("DROP overlap=%.0f%%" % (100 * best["overlap"]))
                continue

            qid += 1                                       # next id
            out.append({
                "id": "h%03d" % qid,                       # e.g. "h001"
                "doc_id": doc_id,                          # which PDF
                "question": best["parsed"]["question"].strip(),  # question
                "reference_answer": best["parsed"]["answer"].strip(),  # answer
                "fact_type": best["parsed"].get("fact_type", "unknown"),  # e.g. "name"
                "gt_page": page_idx,                       # 0-based source page
                "answerable": True,                        # real question
                "reviewed": False,                         # not human checked
                "lexical_overlap": round(best["overlap"], 3),  # e.g. 0.333
            })
            print("OK  ov=%.0f%%  %s" % (100 * best["overlap"],
                                         best["parsed"]["question"][:52]))

    # ── Step 4: save + summary ──
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")  # write file

    if out:                                                # avoid divide by zero
        mean = sum(q["lexical_overlap"] for q in out) / len(out)  # mean overlap
        print("\n" + "=" * 70)
        print("Wrote %d questions to %s" % (len(out), OUT_PATH))
        print("  mean lexical overlap : %.0f%%   (old set: 73%%)" % (100 * mean))
        print("  accepted / rejected  : %d / %d" % (stats["accepted"], stats["rejected"]))
        print("  LLM attempts         : %d" % stats["attempts"])
        print("\nNEXT: py -3.12 eval/verify/verify_retrieval.py --hard")
        print("If BM25 recall drops well below 100%, the set now discriminates.")


if __name__ == "__main__":  # only when run directly
    main()                  # start
