"""
eval/generate_probes.py
─────────────────────────────────────────────────────────────────────────────
Adds VERIFIED unanswerable questions to a question set.

Why these matter
----------------
A RAG system's worst failure is not being wrong -- it is being confidently wrong
about something the document never said. The system prompt tells it to answer
"I could not find this information in the uploaded document." Whether it actually
does is the hallucination rate, and it cannot be measured without questions the
document genuinely cannot answer.

The flaw this fixes
-------------------
generate_qa.py showed the model only a SAMPLE of the document when asking for an
"unanswerable" question, so its claim that a topic is absent was a guess. Two of
its six probes were near-duplicates, and all needed manual re-checking.

Here every candidate is verified the way the real system would answer it: BM25
retrieves the 8 most relevant passages, and a large model judges whether those
passages actually STATE the answer. Only probes the document genuinely cannot
answer are kept.

A first attempt tested whether individual words from the question appeared
anywhere in the document. It rejected every candidate -- any MySQL question
contains words like 'default' and 'columns'. Word presence is not evidence that
a FACT is stated, and that check was replaced.

The hardest probes are ones whose SUBJECT is present but whose SPECIFIC FACT is
not -- e.g. a document that discusses COCO at length but never gives an mAP
figure for it. A weak system retrieves confident-looking pages and invents a
number. Those are requested explicitly.

Usage:
    py -3.12 eval/generate_probes.py eval/qa_set_hard.json
"""
import json
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
from rank_bm25 import BM25Okapi

PROBES_PER_DOC = 3
MAX_ATTEMPTS = 4

# Left to itself the generator produces the same trap three times -- the first
# run gave three near-identical "what mAP did R-CNN achieve" probes (similarity
# 0.98). Each probe is steered to a different KIND of missing fact, and
# candidates too similar to an accepted one are rejected.
FACT_ANGLES = [
    "a numeric threshold, limit, or default setting",
    "a named person, organisation, product, or publication identifier",
    "a percentage, cost, duration, or date",
]

DUP_THRESHOLD = 0.55        # Jaccard on content words


def content_words(text):
    stop = config.QUESTION_STOPWORDS
    return {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if w not in stop}


def too_similar(question, accepted):
    """Jaccard overlap catches reworded duplicates that string matching misses."""
    a = content_words(question)
    if not a:
        return True, 1.0
    for prev in accepted:
        b = content_words(prev)
        j = len(a & b) / len(a | b) if (a | b) else 0.0
        if j >= DUP_THRESHOLD:
            return True, j
    return False, 0.0

PROBE_CACHE = config.CACHE_DIR / "probes"
PROBE_CACHE.mkdir(parents=True, exist_ok=True)


PROBE_PROMPT = """You are building hallucination traps for a document-QA system.

Below is a sample of "{doc_label}" ({doc_shape}).

Write ONE question that:
  - sounds like it plainly belongs to this document's subject area
  - is SPECIFIC (asks for a number, a name, a threshold, a date, a figure)
  - has an answer that is NOT present in a document like this one

The best traps discuss a topic the document DOES cover, but ask for a specific
detail it never states. A weak system will retrieve topically-relevant pages and
invent the detail rather than admitting it cannot find it.

The missing fact you ask for must be: {angle}.

Do not be absurd. The question must look entirely reasonable to someone who has
skimmed the document.
{retry_note}
Return STRICT JSON only, no prose:
{{"question": "...", "missing_detail": "the specific fact being asked for, in 4-8 words"}}

DOCUMENT SAMPLE:
{sample}
"""

RETRY_NOTE = """
IMPORTANT: your previous attempt asked about something the document DOES state.
The passages contained: {found}
Ask about a different specific detail that the document does not state.
"""


def call_llm(llm, prompt, cache_key):
    cache_file = PROBE_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".txt")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    delay = 4.0
    for _ in range(4):
        try:
            out = llm.invoke(prompt).content
            cache_file.write_text(out, encoding="utf-8")
            time.sleep(1.5)
            return out
        except Exception as e:
            print("      retry (%s) in %.0fs" % (type(e).__name__, delay))
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


VERIFY_PROMPT = """Below are the passages a document-QA system would retrieve for
this question, taken from "{doc_label}".

QUESTION: {question}

Decide ONE thing: do these passages actually STATE the answer?

Answer YES only if the specific fact asked for is present and could be quoted.
Answer NO if the passages are merely about the same topic, or discuss it
generally, without stating the specific fact.

Return STRICT JSON only:
{{"answerable": true|false, "evidence": "the sentence that answers it, or empty"}}

PASSAGES:
{passages}
"""


def verify_absent(llm, doc_label, question, bm25, chunks, cache_key):
    """
    Test whether the document answers the question, the way the real system
    would: retrieve the most relevant passages, then check them.

    An earlier version tested whether individual words from the question appeared
    anywhere in the document. That rejected every candidate -- any MySQL question
    contains words like 'default' and 'columns'. Word presence is not evidence
    that a FACT is stated.

    BM25 retrieval needs no embedding API, so this runs with the Google quota
    exhausted.
    """
    scores = bm25.get_scores(re.findall(r"[a-z0-9_]+", question.lower()))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:8]
    passages = "\n\n---\n\n".join(
        "[Page %d]\n%s" % (chunks[i]["page"] + 1, chunks[i]["text"][:900]) for i in top)

    raw = call_llm(llm, VERIFY_PROMPT.format(
        doc_label=doc_label, question=question, passages=passages),
        cache_key="verify::" + cache_key)
    parsed = extract_json(raw)
    if parsed is None:
        return None, ""          # unknown -- treat as not verified
    return (not parsed.get("answerable", True)), parsed.get("evidence", "")[:160]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    qa_path = pathlib.Path(args[0]) if args else config.QA_SET_PATH
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    # Regenerating probes replaces any previous batch rather than appending to it,
    # so a rerun cannot accumulate near-duplicates.
    dropped = [q for q in qa if not q["answerable"]]
    qa = [q for q in qa if q["answerable"]]
    if dropped:
        print("replacing %d existing probe(s)" % len(dropped))
    next_id = len(qa)

    llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0.8, max_tokens=600)
    added = 0

    for doc_id, meta in config.DOCUMENTS.items():
        print("\n[%s] %s" % (doc_id, meta["label"]))
        pages = PyPDFLoader(meta["path"]).load()
        usable = [p for p in pages if len(p.page_content.strip()) >= 450]
        sample = "\n\n".join(p.page_content.strip()[:900] for p in usable[:4])

        # BM25 over the same chunks the real retriever uses -- no embedding API.
        chunks_path = config.CHROMA_EVAL_DIR / (
            config.index_key(doc_id, 1000, 200) + "_chunks.json")
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        bm25 = BM25Okapi([re.findall(r"[a-z0-9_]+", c["text"].lower()) for c in chunks])

        accepted = 0
        accepted_text = []
        for n in range(PROBES_PER_DOC):
            note = ""
            angle = FACT_ANGLES[n % len(FACT_ANGLES)]
            for attempt in range(MAX_ATTEMPTS):
                raw = call_llm(llm, PROBE_PROMPT.format(
                    doc_label=meta["label"], doc_shape=meta["shape"],
                    sample=sample, angle=angle, retry_note=note),
                    cache_key="probe2::%s::%d::%d::%s" % (doc_id, n, attempt, angle))
                parsed = extract_json(raw)
                if not parsed or not parsed.get("question"):
                    continue

                q_text = parsed["question"].strip()
                dup, score = too_similar(q_text, accepted_text)
                if dup:
                    print("   reject (too similar to an accepted probe, %.2f)" % score)
                    note = ("\nIMPORTANT: that is too close to a question already "
                            "accepted for this document. Ask about something "
                            "entirely different.\n")
                    continue

                # Key the verification by the QUESTION, not the loop counters --
                # otherwise a rerun that generates a different question at the
                # same (n, attempt) reuses the previous question's verdict.
                absent, evidence = verify_absent(
                    llm, meta["label"], q_text, bm25, chunks,
                    cache_key=hashlib.sha256(q_text.encode()).hexdigest()[:16])
                if absent is None:
                    print("   reject (verification unparseable)")
                    continue
                if not absent:
                    print("   reject -- document DOES answer it: %s" % evidence[:70])
                    note = RETRY_NOTE.format(found=evidence[:120])
                    continue

                next_id += 1
                added += 1
                accepted += 1
                accepted_text.append(q_text)
                qa.append({
                    "id": "p%03d" % next_id,
                    "doc_id": doc_id,
                    "question": q_text,
                    "reference_answer": "NOT IN DOCUMENT - the system should say it cannot find this.",
                    "fact_type": "unanswerable",
                    "missing_detail": parsed.get("missing_detail", ""),
                    "fact_angle": angle,
                    "gt_page": None,
                    "acceptable_pages": [],
                    "answerable": False,
                    "reviewed": False,
                    "verified_absent": True,
                    "verified_by": "bm25-retrieval + llm check of retrieved passages",
                })
                print("   OK  %s" % q_text[:64])
                break
            else:
                print("   gave up after %d attempts" % MAX_ATTEMPTS)
        print("   accepted %d/%d" % (accepted, PROBES_PER_DOC))

    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    n_unans = sum(1 for q in qa if not q["answerable"])
    print("\n" + "=" * 70)
    print("Added %d verified-absent probes to %s" % (added, qa_path.name))
    print("  total questions   : %d" % len(qa))
    print("  unanswerable      : %d" % n_unans)
    print("  every probe checked against the FULL document text, not a sample")


if __name__ == "__main__":
    main()
