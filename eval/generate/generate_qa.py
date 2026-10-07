"""
eval/generate/generate_qa.py
─────────────────────────────────────────────────────────────────────────────
Drafts the evaluation question set.

Method
------
For each document we sample pages spread across it, and ask a LARGE model
(gpt-oss-120b) to write one question answerable ONLY from that page, plus a
short reference answer. The page number is recorded as GROUND TRUTH.

That page number is the whole point: it lets us score retrieval objectively.
If the retriever never surfaces a chunk from the page the answer actually
lives on, retrieval failed -- regardless of what the LLM then wrote.

We also add deliberately UNANSWERABLE questions: plausible-sounding things the
document does not contain. A good RAG system must refuse these. A system that
confidently answers them is hallucinating, and that is worth measuring.

Output: eval/data/qa_set.json   (review it by hand before trusting any numbers)
"""
# WHAT THIS FILE IS: the "exam paper setter" of the evaluation. It picks pages from each test PDF and asks a big
# model to write one question per page, and it records WHICH page the question came from (the "ground truth").
# Real example from eval/data/qa_set.json: q001 "What SQL command is shown for creating the example database?",
# reference answer `CREATE DATABASE startersql;`, gt_page 4 (0-based page index of the MySQL handbook).
# It also writes "unanswerable" questions (also called probes): questions that sound right for the document but
# whose answer is not in it. A good RAG app must refuse them; answering one means it is hallucinating (making things up).
# Overall flow: load each PDF -> keep pages with enough text -> randomly pick 15 pages -> LLM writes Q + answer
# per page -> LLM writes 2 unanswerable questions per document -> save everything to eval/data/qa_set.json
#
# json = read and write JSON (a plain-text format for lists and dicts). The question set is saved as JSON.
import json
# random = pseudo-random numbers; used to pick which pages get a question.
import random
# re = regular expressions (regex): small patterns that find or remove text, here used to strip ``` fences.
import re
# sys = system module; sys.path is the list of folders Python searches when importing.
import sys
# time = sleep (wait) between API calls and before retries.
import time
# hashlib = makes a fingerprint (sha256 hash) of text; used to name cache files.
import hashlib
# pathlib = easy file paths.
import pathlib

# This script is in a subfolder of eval/, so go ONE folder up (parent.parent) to reach eval/ and put it on
# the import path; then "import config" finds eval/config.py, from any starting folder.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
# config = eval/config.py: the PDF list (DOCUMENTS), model names, folder paths and load_env().
import config

# Read GOOGLE_API_KEY and GROQ_API_KEY from the project's .env file into environment variables.
# Stops the script at once if a key is missing.
config.load_env()

# PyPDFLoader (LangChain) = reads a PDF into one Document per page (text + metadata["page"], 0-based).
from langchain_community.document_loaders import PyPDFLoader
# ChatGroq (LangChain) = a client for chat models hosted on Groq's cloud (here the large gpt-oss-120b).
from langchain_groq import ChatGroq

# Settings, as constants (fixed values written in CAPITALS):
# 15 questions per document -> 3 documents give up to 45 answerable questions.
QUESTIONS_PER_DOC = 15
# 2 unanswerable questions per document.
UNANSWERABLE_PER_DOC = 2
# A page needs at least 450 characters of text to get a question. Shorter pages (title slides, blank pages)
# rarely hold a specific fact worth asking about.
MIN_PAGE_CHARS = 450
# SEED = the starting number for the random generator. Same seed -> same pages picked every run (reproducible).
# 20260916 looks like a date (16 Sep 2026); any fixed number would work.
SEED = 20260916

# Cache folder eval/cache/qa_gen. A cache = saved answers from earlier calls, so a re-run reads the file
# instead of paying for (and waiting on) the same API call again. mkdir creates the folder if it is missing.
GEN_CACHE = config.CACHE_DIR / "qa_gen"
GEN_CACHE.mkdir(parents=True, exist_ok=True)


# IN: llm client, prompt text, cache_key text -> OUT: the model's reply text ("" if every attempt failed).
# WHY: free-tier APIs fail with 429 "too many requests" (rate limit). This function saves every reply to disk
# and retries with growing waits (backoff), so a re-run costs nothing and a rate limit does not kill the script.
# Example: cache_key "ans::mysql::4::20260916" -> file eval/cache/qa_gen/<first 16 hex chars of its sha256>.txt
def call_llm(llm, prompt, cache_key, max_retries=5):
    """Invoke with on-disk caching and backoff, so re-runs are free and 429s survive."""
    # Cache file name = first 16 characters of the sha256 fingerprint of the key. Same key -> same file name.
    cache_file = GEN_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".txt")
    # Cache hit: this exact call was made before -> return the saved reply, no API call.
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    # First wait is 4 seconds; it doubles after every failure (4, 8, 16, 32, 64 seconds).
    delay = 4.0
    # Try up to max_retries (5) times.
    for attempt in range(max_retries):
        # Success path: call the model, save the reply to the cache file, pause 1.5 s so we stay under the rate limit.
        try:
            out = llm.invoke(prompt).content
            cache_file.write_text(out, encoding="utf-8")
            time.sleep(1.5)          # be polite to the free tier
            return out
        # Failure path: print why, wait, double the wait, and try again.
        except Exception as e:
            msg = str(e)
            # "429" or "rate" in the error text = rate limit hit -> say we are sleeping.
            if "429" in msg or "rate" in msg.lower():
                print("      rate limited, sleeping %.0fs (attempt %d/%d)"
                      % (delay, attempt + 1, max_retries))
            # Any other error -> print its type and the first 120 characters of the message.
            else:
                print("      error: %s: %s" % (type(e).__name__, msg[:120]))
            time.sleep(delay)
            delay *= 2
    # All attempts failed -> empty string; the caller treats it as unparseable and skips that page.
    return ""


# IN: raw reply text from the model -> OUT: a Python dict parsed from the JSON inside it, or None.
# WHY: the model was told to return strict JSON, but sometimes adds ```json fences or a sentence around it.
# Example: '```json\n{"question": "...", "answer": "..."}\n```' -> {"question": "...", "answer": "..."}
def extract_json(text):
    """gpt-oss sometimes wraps JSON in prose or fences; dig it out."""
    # Empty reply (for example all retries failed) -> nothing to parse.
    if not text:
        return None
    # Regex, piece by piece: ^``` = three backticks at the start of a line, (?:json)? = optionally followed by the
    # word "json" (?: means a group that is not captured), | = OR, ```$ = three backticks at the end of a line.
    # flags=re.MULTILINE makes ^ and $ work at every line, not only the whole text. Matches are replaced with "".
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    # Take everything from the first "{" to the last "}" - that is the JSON object.
    start, end = text.find("{"), text.rfind("}")
    # No braces found -> not JSON.
    if start == -1 or end == -1:
        return None
    # Try to turn the text into a dict. Broken JSON raises JSONDecodeError -> return None instead of crashing.
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# Prompt template for answerable questions. {page_num}, {doc_label}, {page_text} are filled in with .format().
# {{ and }} are written double so .format() leaves a real { } in the JSON example.
ANSWERABLE_PROMPT = """You are building a rigorous test set for a document-QA system.

Below is the full text of page {page_num} of "{doc_label}".

Write ONE question that:
  - can be answered using ONLY this page's content
  - targets a SPECIFIC, DISTINCTIVE fact (a number, a name, a definition, a
    command, a step) that is unlikely to be repeated elsewhere in the document
  - is phrased self-containedly, as a real reader would ask it
  - does NOT mention "this page", "the passage", "according to the text", or a
    page number

Also give the correct answer, quoted or closely paraphrased from the page, in
at most two sentences.

Return STRICT JSON only, no prose:
{{"question": "...", "answer": "...", "fact_type": "definition|number|procedure|name|concept"}}

PAGE {page_num} TEXT:
{page_text}
"""

# Prompt template for unanswerable questions: shows a sample of the document and asks for a question
# that fits the topic but whose answer is not in the document.
UNANSWERABLE_PROMPT = """You are building a rigorous test set for a document-QA system.

Below is a sample of "{doc_label}" ({doc_shape}).

Write ONE question that sounds like it plainly belongs to this document's
subject area, but whose answer is NOT present anywhere in a document like this
one. It must be specific and plausible -- not absurd. The goal is to check
whether the QA system correctly says it cannot find the information, instead of
inventing an answer.

Return STRICT JSON only, no prose:
{{"question": "...", "why_absent": "one sentence on why this is not in the document"}}

DOCUMENT SAMPLE:
{sample}
"""


# IN: nothing (reads the PDFs listed in eval/data/documents.json) -> OUT: writes eval/data/qa_set.json and prints a summary.
# WHY: builds the question set that run_eval.py scores the RAG app on.
# Example: 3 documents x (15 + 2) = up to 51 questions; eval/README.md says the saved easy set has 48
# (43 answerable + 5 probes).
def main():
    # Fix the random generator so the same pages are chosen every run.
    random.seed(SEED)
    # The question writer is the large judge model (gpt-oss-120b). temperature 0.4 = a bit of variety in wording
    # (0 = always the most likely word). max_tokens 700 = cap on reply length (it also covers hidden reasoning).
    llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0.4, max_tokens=700)

    # qa_set = the list of question dicts we build. qid = running number for ids q001, q002, ...
    qa_set = []
    qid = 0

    # Loop over each document (doc_id like "mysql", meta = its path, label and shape).
    for doc_id, meta in config.DOCUMENTS.items():
        print("\n[%s] %s" % (doc_id, meta["label"]))
        # Read all pages of this PDF.
        pages = PyPDFLoader(meta["path"]).load()
        print("   loaded %d pages" % len(pages))

        # Keep only pages with at least 450 characters of text.
        usable = [p for p in pages if len(p.page_content.strip()) >= MIN_PAGE_CHARS]
        print("   %d pages have >= %d chars of text" % (len(usable), MIN_PAGE_CHARS))

        # Fewer usable pages than 15 -> warn; we will just use all of them.
        if len(usable) < QUESTIONS_PER_DOC:
            print("   WARNING: only %d usable pages" % len(usable))
        # Randomly pick up to 15 different pages, then sort them by page number so progress prints in order.
        chosen = random.sample(usable, min(QUESTIONS_PER_DOC, len(usable)))
        chosen.sort(key=lambda p: p.metadata.get("page", 0))

        # For each chosen page: ask for one question + answer.
        for p in chosen:
            # 0-based page index from the loader's metadata.
            page_idx = p.metadata.get("page", 0)
            # Send at most the first 6000 characters of the page, to keep the prompt small.
            text = p.page_content.strip()[:6000]
            print("   page %3d ..." % (page_idx + 1), end=" ", flush=True)

            # Fill the template (page number shown 1-based to the model) and call the model through the cache.
            # The cache key includes doc, page and seed, so each page has its own cached reply.
            raw = call_llm(
                llm,
                ANSWERABLE_PROMPT.format(page_num=page_idx + 1,
                                         doc_label=meta["label"],
                                         page_text=text),
                cache_key="ans::%s::%d::%d" % (doc_id, page_idx, SEED),
            )
            # Pull the JSON dict out of the reply.
            parsed = extract_json(raw)
            # Reply missing, broken, or without a question or answer -> skip this page.
            if not parsed or not parsed.get("question") or not parsed.get("answer"):
                print("SKIP (unparseable)")
                continue

            # Save the question with its ground truth page ("gt_page") so retrieval can be scored objectively later.
            qid += 1
            qa_set.append({
                "id": "q%03d" % qid,
                "doc_id": doc_id,
                "question": parsed["question"].strip(),
                "reference_answer": parsed["answer"].strip(),
                "fact_type": parsed.get("fact_type", "unknown"),
                "gt_page": page_idx,              # 0-indexed, matches PyPDFLoader
                "answerable": True,
                "reviewed": False,                # flip to true after human check
            })
            print("OK  %s" % parsed["question"][:58])

        # ── unanswerable probes ────────────────────────────────────────────
        # Sample for the unanswerable prompt: first 900 characters of the first 4 usable pages, joined with blank lines.
        sample = "\n\n".join(p.page_content.strip()[:900] for p in usable[:4])
        # Ask for 2 unanswerable questions for this document.
        for n in range(UNANSWERABLE_PER_DOC):
            print("   unanswerable #%d ..." % (n + 1), end=" ", flush=True)
            raw = call_llm(
                llm,
                UNANSWERABLE_PROMPT.format(doc_label=meta["label"],
                                           doc_shape=meta["shape"],
                                           sample=sample),
                # Separate cache key per probe number n, so the two probes are different calls.
                cache_key="unans::%s::%d::%d" % (doc_id, n, SEED),
            )
            parsed = extract_json(raw)
            # Unparseable reply or no question -> skip.
            if not parsed or not parsed.get("question"):
                print("SKIP")
                continue
            # Save the probe: gt_page None (no correct page) and answerable False.
            qid += 1
            qa_set.append({
                "id": "q%03d" % qid,
                "doc_id": doc_id,
                "question": parsed["question"].strip(),
                "reference_answer": "NOT IN DOCUMENT - the system should say it cannot find this.",
                "fact_type": "unanswerable",
                "why_absent": parsed.get("why_absent", ""),
                "gt_page": None,
                "answerable": False,
                "reviewed": False,
            })
            print("OK  %s" % parsed["question"][:58])

    # Write the full set to eval/data/qa_set.json (indent=2 = readable; ensure_ascii=False keeps non-English characters).
    config.QA_SET_PATH.write_text(
        json.dumps(qa_set, indent=2, ensure_ascii=False), encoding="utf-8")

    # Count answerable questions and print a summary with a reminder to check the answers by hand.
    n_ans = sum(1 for q in qa_set if q["answerable"])
    print("\n" + "=" * 70)
    print("Wrote %d questions to %s" % (len(qa_set), config.QA_SET_PATH))
    print("  answerable   : %d" % n_ans)
    print("  unanswerable : %d" % (len(qa_set) - n_ans))
    print("\nNEXT: read qa_set.json and sanity-check the answers before trusting")
    print("any score. A test built on wrong answers is worse than no test.")


# Run main() only when the file is started directly (py -3.12 eval/generate/generate_qa.py), not when imported.
if __name__ == "__main__":
    main()
