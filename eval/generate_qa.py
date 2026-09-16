"""
eval/generate_qa.py
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

Output: eval/qa_set.json   (review it by hand before trusting any numbers)
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

QUESTIONS_PER_DOC = 15
UNANSWERABLE_PER_DOC = 2
MIN_PAGE_CHARS = 450
SEED = 20260916

GEN_CACHE = config.CACHE_DIR / "qa_gen"
GEN_CACHE.mkdir(parents=True, exist_ok=True)


def call_llm(llm, prompt, cache_key, max_retries=5):
    """Invoke with on-disk caching and backoff, so re-runs are free and 429s survive."""
    cache_file = GEN_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".txt")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    delay = 4.0
    for attempt in range(max_retries):
        try:
            out = llm.invoke(prompt).content
            cache_file.write_text(out, encoding="utf-8")
            time.sleep(1.5)          # be polite to the free tier
            return out
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                print("      rate limited, sleeping %.0fs (attempt %d/%d)"
                      % (delay, attempt + 1, max_retries))
            else:
                print("      error: %s: %s" % (type(e).__name__, msg[:120]))
            time.sleep(delay)
            delay *= 2
    return ""


def extract_json(text):
    """gpt-oss sometimes wraps JSON in prose or fences; dig it out."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


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


def main():
    random.seed(SEED)
    llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0.4, max_tokens=700)

    qa_set = []
    qid = 0

    for doc_id, meta in config.DOCUMENTS.items():
        print("\n[%s] %s" % (doc_id, meta["label"]))
        pages = PyPDFLoader(meta["path"]).load()
        print("   loaded %d pages" % len(pages))

        usable = [p for p in pages if len(p.page_content.strip()) >= MIN_PAGE_CHARS]
        print("   %d pages have >= %d chars of text" % (len(usable), MIN_PAGE_CHARS))

        if len(usable) < QUESTIONS_PER_DOC:
            print("   WARNING: only %d usable pages" % len(usable))
        chosen = random.sample(usable, min(QUESTIONS_PER_DOC, len(usable)))
        chosen.sort(key=lambda p: p.metadata.get("page", 0))

        for p in chosen:
            page_idx = p.metadata.get("page", 0)
            text = p.page_content.strip()[:6000]
            print("   page %3d ..." % (page_idx + 1), end=" ", flush=True)

            raw = call_llm(
                llm,
                ANSWERABLE_PROMPT.format(page_num=page_idx + 1,
                                         doc_label=meta["label"],
                                         page_text=text),
                cache_key="ans::%s::%d::%d" % (doc_id, page_idx, SEED),
            )
            parsed = extract_json(raw)
            if not parsed or not parsed.get("question") or not parsed.get("answer"):
                print("SKIP (unparseable)")
                continue

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
        sample = "\n\n".join(p.page_content.strip()[:900] for p in usable[:4])
        for n in range(UNANSWERABLE_PER_DOC):
            print("   unanswerable #%d ..." % (n + 1), end=" ", flush=True)
            raw = call_llm(
                llm,
                UNANSWERABLE_PROMPT.format(doc_label=meta["label"],
                                           doc_shape=meta["shape"],
                                           sample=sample),
                cache_key="unans::%s::%d::%d" % (doc_id, n, SEED),
            )
            parsed = extract_json(raw)
            if not parsed or not parsed.get("question"):
                print("SKIP")
                continue
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

    config.QA_SET_PATH.write_text(
        json.dumps(qa_set, indent=2, ensure_ascii=False), encoding="utf-8")

    n_ans = sum(1 for q in qa_set if q["answerable"])
    print("\n" + "=" * 70)
    print("Wrote %d questions to %s" % (len(qa_set), config.QA_SET_PATH))
    print("  answerable   : %d" % n_ans)
    print("  unanswerable : %d" % (len(qa_set) - n_ans))
    print("\nNEXT: read qa_set.json and sanity-check the answers before trusting")
    print("any score. A test built on wrong answers is worse than no test.")


if __name__ == "__main__":
    main()
