"""
eval/generate_qa_hard.py
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

Output: eval/qa_set_hard.json

Uses Groq only (no embedding API), so it runs with the Google quota exhausted,
and BM25 scoring in verify_retrieval.py can test the result immediately.
"""
import json
import random
import re
import sys
import time
import hashlib
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

config.load_env()

from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

QUESTIONS_PER_DOC = 12
MIN_PAGE_CHARS = 450
MAX_OVERLAP = 0.55          # reject anything more lexically leaky than this
MAX_ATTEMPTS = 3
SEED = 20260916

HARD_CACHE = config.CACHE_DIR / "qa_hard"
HARD_CACHE.mkdir(parents=True, exist_ok=True)

OUT_PATH = config.EVAL_DIR / "qa_set_hard.json"


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

RETRY_NOTE = """
IMPORTANT: your previous attempt reused too many of the page's own words
(overlap {overlap:.0%}, limit {limit:.0%}). These words appeared in BOTH your
question and the page: {shared}
Write a MORE INDIRECT question that avoids them entirely.
"""


def call_llm(llm, prompt, cache_key):
    cache_file = HARD_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".txt")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    delay = 4.0
    for attempt in range(5):
        try:
            out = llm.invoke(prompt).content
            cache_file.write_text(out, encoding="utf-8")
            time.sleep(1.5)
            return out
        except Exception as e:
            print("      retry (%s), sleeping %.0fs" % (type(e).__name__, delay))
            time.sleep(delay)
            delay *= 2
    return ""


def extract_json(text):
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def shared_words(question, page_lower):
    words = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", question.lower())
             if w not in config.QUESTION_STOPWORDS}
    return sorted(w for w in words if w in page_lower)[:10]


def main():
    random.seed(SEED)
    llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0.7, max_tokens=700)

    out = []
    qid = 0
    stats = {"accepted": 0, "rejected": 0, "attempts": 0}

    for doc_id, meta in config.DOCUMENTS.items():
        print("\n[%s] %s" % (doc_id, meta["label"]))
        pages = [p for p in PyPDFLoader(meta["path"]).load()
                 if len(p.page_content.strip()) >= MIN_PAGE_CHARS]
        chosen = random.sample(pages, min(QUESTIONS_PER_DOC, len(pages)))
        chosen.sort(key=lambda p: p.metadata.get("page", 0))

        for p in chosen:
            page_idx = p.metadata.get("page", 0)
            text = p.page_content.strip()[:6000]
            page_lower = p.page_content.lower()
            print("   page %3d" % (page_idx + 1), end=" ", flush=True)

            best = None
            for attempt in range(MAX_ATTEMPTS):
                stats["attempts"] += 1
                note = ""
                if best is not None:
                    note = RETRY_NOTE.format(overlap=best["overlap"], limit=MAX_OVERLAP,
                                             shared=", ".join(best["shared"]))
                raw = call_llm(llm, HARD_PROMPT.format(
                    page_num=page_idx + 1, doc_label=meta["label"],
                    page_text=text, retry_note=note),
                    cache_key="hard::%s::%d::%d::%d" % (doc_id, page_idx, attempt, SEED))
                parsed = extract_json(raw)
                if not parsed or not parsed.get("question") or not parsed.get("answer"):
                    continue

                ov = config.question_page_overlap(parsed["question"], page_lower)
                cand = {"parsed": parsed, "overlap": ov,
                        "shared": shared_words(parsed["question"], page_lower)}
                if best is None or ov < best["overlap"]:
                    best = cand
                if ov <= MAX_OVERLAP:
                    break

            if best is None:
                print("SKIP (unparseable)")
                continue

            accepted = best["overlap"] <= MAX_OVERLAP
            stats["accepted" if accepted else "rejected"] += 1
            if not accepted:
                print("DROP overlap=%.0f%%" % (100 * best["overlap"]))
                continue

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

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    if out:
        mean = sum(q["lexical_overlap"] for q in out) / len(out)
        print("\n" + "=" * 70)
        print("Wrote %d questions to %s" % (len(out), OUT_PATH))
        print("  mean lexical overlap : %.0f%%   (old set: 73%%)" % (100 * mean))
        print("  accepted / rejected  : %d / %d" % (stats["accepted"], stats["rejected"]))
        print("  LLM attempts         : %d" % stats["attempts"])
        print("\nNEXT: py -3.12 eval/verify_retrieval.py --hard")
        print("If BM25 recall drops well below 100%, the set now discriminates.")


if __name__ == "__main__":
    main()
