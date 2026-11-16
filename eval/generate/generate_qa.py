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
# WHAT THIS FILE IS: writes the test questions. One question per picked page, and records that page ("ground truth").
# Example: q001 "What SQL command is shown for creating the example database?" -> `CREATE DATABASE startersql;`, gt_page 4.
# Also writes unanswerable probes (answer not in the PDF); answering one = hallucinating.
# Flow: load PDFs -> keep pages with enough text -> pick 15 at random -> LLM writes Q + answer per page
#       -> 2 probes per PDF -> save eval/data/qa_set.json.

import json      # save the set as JSON
import random    # pick pages at random
import re        # regex, strip ``` fences
import sys       # sys.path for imports
import time      # sleep between calls / retries
import hashlib   # sha256 for cache file names
import pathlib   # file paths

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # add eval/ so "import config" works
import config    # eval/config.py: DOCUMENTS, models, folders, load_env()

config.load_env()  # load GOOGLE_API_KEY + GROQ_API_KEY from .env (stops if missing)

from langchain_community.document_loaders import PyPDFLoader  # PDF -> one Document per page (page 0-based)
from langchain_groq import ChatGroq                           # Groq chat client (gpt-oss-120b)

# ── BLOCK: settings ─────────────────────────────────────────────────────
QUESTIONS_PER_DOC = 15     # 3 PDFs -> up to 45 answerable
UNANSWERABLE_PER_DOC = 2   # probes per PDF
MIN_PAGE_CHARS = 450       # skip title/blank pages
SEED = 20260916            # fixed seed = same pages every run

GEN_CACHE = config.CACHE_DIR / "qa_gen"         # eval/cache/qa_gen: saved replies, free re-runs
GEN_CACHE.mkdir(parents=True, exist_ok=True)    # create if missing


# ── BLOCK: LLM call with cache + retry ──────────────────────────────────
# IN: llm, prompt, cache_key -> OUT: reply text ("" if all tries fail)
# e.g. key "ans::mysql::4::20260916" -> eval/cache/qa_gen/<16 hex chars>.txt
def call_llm(llm, prompt, cache_key, max_retries=5):
    """Invoke with on-disk caching and backoff, so re-runs are free and 429s survive."""
    cache_file = GEN_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".txt")  # key -> file name
    if cache_file.exists():                                # asked before?
        return cache_file.read_text(encoding="utf-8")      # reuse, no API call

    delay = 4.0                                            # 4, 8, 16, 32, 64 s backoff
    for attempt in range(max_retries):                     # up to 5 tries
        try:
            out = llm.invoke(prompt).content               # call the model
            cache_file.write_text(out, encoding="utf-8")   # save reply
            time.sleep(1.5)          # be polite to the free tier
            return out
        except Exception as e:                             # API failed
            msg = str(e)                                   # error text
            if "429" in msg or "rate" in msg.lower():      # 429 = rate limit
                print("      rate limited, sleeping %.0fs (attempt %d/%d)"
                      % (delay, attempt + 1, max_retries))
            else:                                          # other error
                print("      error: %s: %s" % (type(e).__name__, msg[:120]))  # type + 120 chars
            time.sleep(delay)                              # wait
            delay *= 2                                     # double next wait
    return ""                                              # all failed, caller skips


# IN: raw reply -> OUT: parsed dict or None
# e.g. '```json\n{"question": "...", "answer": "..."}\n```' -> {"question": ..., "answer": ...}
def extract_json(text):
    """gpt-oss sometimes wraps JSON in prose or fences; dig it out."""
    if not text:                                           # empty reply
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()  # remove ``` fences
    start, end = text.find("{"), text.rfind("}")           # JSON = first "{" to last "}"
    if start == -1 or end == -1:                           # no braces
        return None
    try:
        return json.loads(text[start:end + 1])             # parse
    except json.JSONDecodeError:                           # broken JSON
        return None


# ── BLOCK: prompts ──────────────────────────────────────────────────────
# One question from one page. {{ }} = literal braces after .format().
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

# On-topic question whose answer is NOT in the PDF.
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


# ── BLOCK: main ─────────────────────────────────────────────────────────
# IN: PDFs in eval/data/documents.json -> OUT: eval/data/qa_set.json
# e.g. 3 x (15 + 2) = up to 51; README: saved easy set has 48 (43 + 5 probes)
def main():
    random.seed(SEED)                                      # same pages every run
    llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0.4, max_tokens=700)  # 0.4 = some variety; 700 token cap

    qa_set = []                                            # questions built
    qid = 0                                                # running id: q001, q002, ...

    for doc_id, meta in config.DOCUMENTS.items():          # each PDF, e.g. "mysql"
        # ── Step 1: load and pick pages ──
        print("\n[%s] %s" % (doc_id, meta["label"]))
        pages = PyPDFLoader(meta["path"]).load()           # all pages
        print("   loaded %d pages" % len(pages))

        usable = [p for p in pages if len(p.page_content.strip()) >= MIN_PAGE_CHARS]  # 450+ chars only
        print("   %d pages have >= %d chars of text" % (len(usable), MIN_PAGE_CHARS))

        if len(usable) < QUESTIONS_PER_DOC:                # fewer than 15?
            print("   WARNING: only %d usable pages" % len(usable))  # use them all
        chosen = random.sample(usable, min(QUESTIONS_PER_DOC, len(usable)))  # up to 15 distinct pages
        chosen.sort(key=lambda p: p.metadata.get("page", 0))  # page order for tidy output

        # ── Step 2: one question per chosen page ──
        for p in chosen:
            page_idx = p.metadata.get("page", 0)           # 0-based page index
            text = p.page_content.strip()[:6000]           # max 6000 chars to the model
            print("   page %3d ..." % (page_idx + 1), end=" ", flush=True)  # progress, 1-based

            raw = call_llm(
                llm,
                ANSWERABLE_PROMPT.format(page_num=page_idx + 1,     # 1-based for the model
                                         doc_label=meta["label"],   # PDF name
                                         page_text=text),           # page text
                cache_key="ans::%s::%d::%d" % (doc_id, page_idx, SEED),  # one cache per page
            )
            parsed = extract_json(raw)                     # reply -> dict
            if not parsed or not parsed.get("question") or not parsed.get("answer"):  # bad reply
                print("SKIP (unparseable)")
                continue                                   # next page

            qid += 1                                       # next id
            qa_set.append({
                "id": "q%03d" % qid,                       # e.g. "q001"
                "doc_id": doc_id,                          # which PDF
                "question": parsed["question"].strip(),    # the question
                "reference_answer": parsed["answer"].strip(),  # correct answer
                "fact_type": parsed.get("fact_type", "unknown"),  # e.g. "number"
                "gt_page": page_idx,              # 0-indexed, matches PyPDFLoader
                "answerable": True,                        # real question
                "reviewed": False,                # flip to true after human check
            })
            print("OK  %s" % parsed["question"][:58])

        # ── unanswerable probes ────────────────────────────────────────────
        sample = "\n\n".join(p.page_content.strip()[:900] for p in usable[:4])  # 4 pages x 900 chars
        for n in range(UNANSWERABLE_PER_DOC):              # 2 probes
            print("   unanswerable #%d ..." % (n + 1), end=" ", flush=True)
            raw = call_llm(
                llm,
                UNANSWERABLE_PROMPT.format(doc_label=meta["label"],    # PDF name
                                           doc_shape=meta["shape"],    # PDF shape
                                           sample=sample),             # text sample
                cache_key="unans::%s::%d::%d" % (doc_id, n, SEED),     # one cache per probe
            )
            parsed = extract_json(raw)                     # reply -> dict
            if not parsed or not parsed.get("question"):   # bad reply
                print("SKIP")
                continue
            qid += 1                                       # next id
            qa_set.append({
                "id": "q%03d" % qid,                       # same id series
                "doc_id": doc_id,                          # which PDF
                "question": parsed["question"].strip(),    # the trap
                "reference_answer": "NOT IN DOCUMENT - the system should say it cannot find this.",
                "fact_type": "unanswerable",               # probe type
                "why_absent": parsed.get("why_absent", ""),  # model's reason
                "gt_page": None,                           # no correct page
                "answerable": False,                       # marks a probe
                "reviewed": False,                         # not human checked
            })
            print("OK  %s" % parsed["question"][:58])

    # ── Step 3: save + summary ──
    config.QA_SET_PATH.write_text(
        json.dumps(qa_set, indent=2, ensure_ascii=False), encoding="utf-8")  # eval/data/qa_set.json

    n_ans = sum(1 for q in qa_set if q["answerable"])      # answerable count
    print("\n" + "=" * 70)
    print("Wrote %d questions to %s" % (len(qa_set), config.QA_SET_PATH))
    print("  answerable   : %d" % n_ans)
    print("  unanswerable : %d" % (len(qa_set) - n_ans))
    print("\nNEXT: read qa_set.json and sanity-check the answers before trusting")  # check by hand
    print("any score. A test built on wrong answers is worse than no test.")


if __name__ == "__main__":  # only when run directly
    main()                  # start
