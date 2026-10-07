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
# WHAT THIS FILE IS: the "tricky exam setter". generate_qa.py wrote questions that copy the page's own words, so even
# a dumb keyword search could find the page. This file asks for questions in DIFFERENT words (synonyms, descriptions)
# and throws away any question that still copies too many words from its page.
# Real example from eval/data/qa_set_hard.json: h001 "What is the sample name used for the new data container in the
# introductory instructions?" (answer "startersql", gt_page 4, lexical_overlap 0.333). The easy set asked the same
# fact as "What SQL command is shown for creating the example database?".
# Jargon:
#   lexical overlap = the share of a question's content words that also appear on its source page
#     (config.question_page_overlap). 0.333 = one third of the question's words are on the page.
#   BM25 = a classic keyword-search scoring method (no AI): it ranks pages by how often they contain the query words.
# eval/README.md: easy set 73% mean overlap and BM25 recall@4 97.7%; hard set 21% overlap and BM25 recall@4 57.6%.
# Overall flow: load each PDF -> pick 12 pages with enough text -> LLM writes a paraphrased Q + answer -> measure
# overlap -> if above 55%, retry (up to 3 tries) telling the model which words to avoid -> keep the best -> save JSON
#
# json = read and write JSON (plain-text lists/dicts); the output file eval/data/qa_set_hard.json is JSON.
import json
# random = pseudo-random numbers; picks which pages get a question.
import random
# re = regular expressions (text patterns); used to strip ``` fences and to split a question into words.
import re
# sys = system module; sys.path controls where imports are searched.
import sys
# time = waits between calls and before retries.
import time
# hashlib = sha256 fingerprints, used to name cache files.
import hashlib
# pathlib = easy file paths.
import pathlib

# Put the eval/ folder on the import path so "import config" finds eval/config.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
# config = eval/config.py: PDF list, model names, folders, the stopword list and question_page_overlap().
import config

# Load the API keys from the project's .env file (stops if one is missing).
config.load_env()

# PyPDFLoader (LangChain) = reads a PDF into one Document per page.
from langchain_community.document_loaders import PyPDFLoader
# ChatGroq (LangChain) = client for chat models on Groq's cloud (here gpt-oss-120b writes the questions).
from langchain_groq import ChatGroq

# 12 questions per document (the easy set used 15).
QUESTIONS_PER_DOC = 12
# Pages under 450 characters (title pages, near-empty slides) are skipped.
MIN_PAGE_CHARS = 450
# 0.55 = at most 55% of a question's content words may appear on its page. The README notes the easy set averaged 73%,
# so 55% forces clearly different wording while still allowing a few unavoidable words (like a table name).
MAX_OVERLAP = 0.55          # reject anything more lexically leaky than this
# At most 3 tries per page before the page is dropped.
MAX_ATTEMPTS = 3
# Same random seed as generate_qa.py, so runs are repeatable.
SEED = 20260916

# Cache folder eval/cache/qa_hard: saved replies so a re-run makes no new API calls.
HARD_CACHE = config.CACHE_DIR / "qa_hard"
HARD_CACHE.mkdir(parents=True, exist_ok=True)

# Output file: eval/data/qa_set_hard.json.
OUT_PATH = config.QA_SET_HARD_PATH


# Prompt template for a paraphrased question. {retry_note} is "" on the first try and RETRY_NOTE on later tries.
# {{ }} are doubled so .format() leaves real braces in the JSON example.
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

# Extra text added to the prompt on a retry: tells the model its overlap, the limit, and the exact shared words to avoid.
# {overlap:.0%} prints a fraction as a whole percent (0.6 -> "60%").
RETRY_NOTE = """
IMPORTANT: your previous attempt reused too many of the page's own words
(overlap {overlap:.0%}, limit {limit:.0%}). These words appeared in BOTH your
question and the page: {shared}
Write a MORE INDIRECT question that avoids them entirely.
"""


# IN: llm client, prompt text, cache_key -> OUT: the reply text, or "" after 5 failed attempts.
# WHY: same caching + backoff idea as in generate_qa.py: re-runs are free and rate limits (HTTP 429) are survived.
# Example: cache_key "hard::mysql::4::0::20260916" (doc, page, attempt, seed) -> eval/cache/qa_hard/<16 hex chars>.txt
def call_llm(llm, prompt, cache_key):
    cache_file = HARD_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".txt")
    # Cache hit -> return the saved reply without calling the API.
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    # Backoff: waits of 4, 8, 16, 32, 64 seconds between failed attempts.
    delay = 4.0
    # Up to 5 attempts.
    for attempt in range(5):
        # Call the model, save the reply, pause 1.5 s to stay under the free-tier rate limit.
        try:
            out = llm.invoke(prompt).content
            cache_file.write_text(out, encoding="utf-8")
            time.sleep(1.5)
            return out
        # Any error (rate limit, network) -> print it, wait, double the wait, try again.
        except Exception as e:
            print("      retry (%s), sleeping %.0fs" % (type(e).__name__, delay))
            time.sleep(delay)
            delay *= 2
    # All 5 attempts failed.
    return ""


# IN: raw model reply -> OUT: dict from the JSON inside it, or None.
# WHY: the model sometimes wraps JSON in ```json fences or extra words.
# Example: '```json\n{"question": "...", "answer": "startersql"}\n```' -> {"question": "...", "answer": "startersql"}
def extract_json(text):
    # Empty reply -> nothing to parse.
    if not text:
        return None
    # Regex: ^```(?:json)? = backticks (optionally followed by "json") at a line start, | = OR, ```$ = backticks at a
    # line end. re.MULTILINE lets ^ and $ match at every line. The fences are deleted.
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    # The JSON object runs from the first "{" to the last "}".
    s, e = text.find("{"), text.rfind("}")
    # No braces -> not JSON.
    if s == -1 or e == -1:
        return None
    # Parse; broken JSON -> None instead of a crash.
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


# IN: question text, page text in lowercase -> OUT: up to 10 question words that also appear on the page (sorted).
# WHY: these words go into RETRY_NOTE, so the model knows exactly which words to avoid next time.
# Example: "How would someone change what an employee earns?" on a page with "employee" -> ["employee", ...].
def shared_words(question, page_lower):
    # Regex [a-z][a-z0-9_]{2,} = a word that starts with a letter, then 2 or more letters/digits/underscores
    # (so words of 3+ characters). Stopwords like "what", "the", "does" are removed.
    words = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", question.lower())
             if w not in config.QUESTION_STOPWORDS}
    # "w in page_lower" is a substring test on the page text. [:10] = keep at most 10 words.
    return sorted(w for w in words if w in page_lower)[:10]


# IN: nothing (reads the PDFs in eval/data/documents.json) -> OUT: writes eval/data/qa_set_hard.json and prints statistics.
# WHY: builds a question set where keyword search cannot win just by copying words, so dense (embedding) retrieval
# can show whether it is worth the API call.
# Example: eval/README.md reports 33 answerable questions in the hard set (42 including 9 probes added later).
def main():
    # Fixed seed -> the same pages are picked every run.
    random.seed(SEED)
    # Question writer: gpt-oss-120b. temperature 0.7 = more variety than the easy set (0.4), which helps the model
    # find new wording. max_tokens 700 caps the reply (including hidden reasoning).
    llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0.7, max_tokens=700)

    # out = accepted questions, qid = running number for ids h001, h002, ..., stats = counters printed at the end.
    out = []
    qid = 0
    stats = {"accepted": 0, "rejected": 0, "attempts": 0}

    # Loop over each document.
    for doc_id, meta in config.DOCUMENTS.items():
        print("\n[%s] %s" % (doc_id, meta["label"]))
        # Load all pages and keep only those with at least 450 characters.
        pages = [p for p in PyPDFLoader(meta["path"]).load()
                 if len(p.page_content.strip()) >= MIN_PAGE_CHARS]
        # Randomly choose up to 12 pages, sorted by page number.
        chosen = random.sample(pages, min(QUESTIONS_PER_DOC, len(pages)))
        chosen.sort(key=lambda p: p.metadata.get("page", 0))

        # For each chosen page, try to get a low-overlap question.
        for p in chosen:
            # page_idx = 0-based page number; text = first 6000 characters sent to the model; page_lower = lowercase full
            # page text used to measure overlap.
            page_idx = p.metadata.get("page", 0)
            text = p.page_content.strip()[:6000]
            page_lower = p.page_content.lower()
            print("   page %3d" % (page_idx + 1), end=" ", flush=True)

            # best = the lowest-overlap valid question seen so far for this page (None = none yet).
            best = None
            # Up to 3 attempts for this page.
            for attempt in range(MAX_ATTEMPTS):
                stats["attempts"] += 1
                # First attempt: no retry note.
                note = ""
                # A previous attempt exists -> build the retry note with its overlap and shared words.
                if best is not None:
                    note = RETRY_NOTE.format(overlap=best["overlap"], limit=MAX_OVERLAP,
                                             shared=", ".join(best["shared"]))
                # Call the model. The attempt number is part of the cache key, so each retry is a different cached call.
                raw = call_llm(llm, HARD_PROMPT.format(
                    page_num=page_idx + 1, doc_label=meta["label"],
                    page_text=text, retry_note=note),
                    cache_key="hard::%s::%d::%d::%d" % (doc_id, page_idx, attempt, SEED))
                # Parse the reply into a dict.
                parsed = extract_json(raw)
                # Unusable reply (no JSON, no question or no answer) -> try the next attempt.
                if not parsed or not parsed.get("question") or not parsed.get("answer"):
                    continue

                # Measure the overlap of this question with its own page and note the shared words.
                ov = config.question_page_overlap(parsed["question"], page_lower)
                cand = {"parsed": parsed, "overlap": ov,
                        "shared": shared_words(parsed["question"], page_lower)}
                # Keep this candidate if it is the first one or has lower overlap than the best so far.
                if best is None or ov < best["overlap"]:
                    best = cand
                # Good enough (overlap 55% or less) -> stop trying for this page.
                if ov <= MAX_OVERLAP:
                    break

            # All attempts unusable -> skip the page.
            if best is None:
                print("SKIP (unparseable)")
                continue

            # Accept only if even the best attempt is within the 55% limit; count accepted vs rejected.
            accepted = best["overlap"] <= MAX_OVERLAP
            stats["accepted" if accepted else "rejected"] += 1
            # Still too much overlap after 3 tries -> drop the page (no question for it).
            if not accepted:
                print("DROP overlap=%.0f%%" % (100 * best["overlap"]))
                continue

            # Save the accepted question, including its measured lexical_overlap rounded to 3 decimals.
            qid += 1
            out.append({
                "id": "h%03d" % qid,
                "doc_id": doc_id,
                "question": best["parsed"]["question"].strip(),
                "reference_answer": best["parsed"]["answer"].strip(),
                "fact_type": best["parsed"].get("fact_type", "unknown"),
                "gt_page": page_idx,
                "answerable": True,
                "reviewed": False,
                "lexical_overlap": round(best["overlap"], 3),
            })
            print("OK  ov=%.0f%%  %s" % (100 * best["overlap"],
                                         best["parsed"]["question"][:52]))

    # Write all accepted questions to eval/data/qa_set_hard.json.
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print a summary only if at least one question was accepted (avoids dividing by zero).
    if out:
        # Mean overlap across all accepted questions; the print compares it to the old set's 73%.
        mean = sum(q["lexical_overlap"] for q in out) / len(out)
        print("\n" + "=" * 70)
        print("Wrote %d questions to %s" % (len(out), OUT_PATH))
        print("  mean lexical overlap : %.0f%%   (old set: 73%%)" % (100 * mean))
        print("  accepted / rejected  : %d / %d" % (stats["accepted"], stats["rejected"]))
        print("  LLM attempts         : %d" % stats["attempts"])
        print("\nNEXT: py -3.12 eval/verify/verify_retrieval.py --hard")
        print("If BM25 recall drops well below 100%, the set now discriminates.")


# Run main() only when this file is started directly.
if __name__ == "__main__":
    main()
