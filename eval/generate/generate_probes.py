"""
eval/generate/generate_probes.py
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
    py -3.12 eval/generate/generate_probes.py eval/data/qa_set_hard.json
"""
# WHAT THIS FILE IS: writes hallucination probes (trap questions the PDF can NOT answer) and checks each trap is real.
# The app should say "I could not find this information...". Answering anyway = hallucinating.
# Example: p034 "What is the default maximum connections limit in a fresh MySQL installation?" (verified_absent: true).
# Flow: drop old probes -> per PDF: BM25 over its chunks -> LLM writes a probe per "fact angle"
#       -> reject near-duplicates (Jaccard) -> BM25 top 8 + judge: answer really absent? -> keep -> save file.

import json      # read/write question files (JSON)
import re        # regex: split words, strip ``` fences
import sys       # sys.argv, sys.path
import time      # sleep between calls / before retries
import hashlib   # sha256 for cache names and keys
import pathlib   # file paths

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # add eval/ so "import config" works
import config    # eval/config.py: PDFs, models, folders, stopwords, index_key()

config.load_env()  # load GOOGLE_API_KEY + GROQ_API_KEY from .env (stops if missing)

from langchain_community.document_loaders import PyPDFLoader  # PDF -> one Document per page
from langchain_groq import ChatGroq                           # Groq chat client (gpt-oss-120b writes + judges)
from rank_bm25 import BM25Okapi  # BM25 = keyword search, rare words weigh more, no API needed

# ── BLOCK: settings ─────────────────────────────────────────────────────
PROBES_PER_DOC = 3   # 3 per PDF -> up to 9 (README: hard set has 9 probes)
MAX_ATTEMPTS = 4     # tries per probe

# Left to itself the generator produces the same trap three times -- the first
# run gave three near-identical "what mAP did R-CNN achieve" probes (similarity
# 0.98). Each probe is steered to a different KIND of missing fact, and
# candidates too similar to an accepted one are rejected.
FACT_ANGLES = [                                            # probe n uses angle n
    "a numeric threshold, limit, or default setting",       # angle 0
    "a named person, organisation, product, or publication identifier",  # angle 1
    "a percentage, cost, duration, or date",                # angle 2
]

# Jaccard = shared words / all distinct words (0 to 1). README: worst pair similarity went 1.00 -> 0.17.
DUP_THRESHOLD = 0.55        # Jaccard on content words


# ── BLOCK: duplicate check ──────────────────────────────────────────────
# IN: text -> OUT: set of content words (lowercase, 3+ chars, not a stopword)
# e.g. "What IoU threshold does R-CNN use?" -> {"iou", "threshold", "cnn"}
def content_words(text):
    stop = config.QUESTION_STOPWORDS                       # "what", "the", "does", ...
    return {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if w not in stop}  # 3+ char words minus stopwords


# IN: new question, accepted questions -> OUT: (duplicate?, similarity)
# e.g. two "what mAP did R-CNN achieve" variants -> (True, 0.98)
def too_similar(question, accepted):
    """Jaccard overlap catches reworded duplicates that string matching misses."""
    a = content_words(question)                            # words of the new question
    if not a:                                              # no content words at all
        return True, 1.0                                   # useless, treat as duplicate
    for prev in accepted:                                  # compare with each accepted one
        b = content_words(prev)                            # its words
        j = len(a & b) / len(a | b) if (a | b) else 0.0    # Jaccard, guard divide by zero
        if j >= DUP_THRESHOLD:                             # 0.55+ = reworded copy
            return True, j                                 # duplicate + score
    return False, 0.0                                      # nothing close


PROBE_CACHE = config.CACHE_DIR / "probes"                  # eval/cache/probes: saved replies, free re-runs
PROBE_CACHE.mkdir(parents=True, exist_ok=True)             # create if missing


# ── BLOCK: prompts ──────────────────────────────────────────────────────
# Asks for one trap question. {angle} = kind of missing fact, {retry_note} = feedback; {{ }} = literal braces.
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

# Retry feedback when the judge found the PDF DOES answer the last question.
RETRY_NOTE = """
IMPORTANT: your previous attempt asked about something the document DOES state.
The passages contained: {found}
Ask about a different specific detail that the document does not state.
"""


# ── BLOCK: LLM call with cache + retry ──────────────────────────────────
# IN: llm, prompt, cache_key -> OUT: reply text, or "" after 4 fails
# waits 4, 8, 16, 32 s on errors (free-tier rate limits)
def call_llm(llm, prompt, cache_key):
    cache_file = PROBE_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".txt")  # key -> 16-char name
    if cache_file.exists():                                # cached?
        return cache_file.read_text(encoding="utf-8")      # reuse, no API call
    delay = 4.0                                            # first wait, doubles each fail
    for _ in range(4):                                     # up to 4 attempts
        try:
            out = llm.invoke(prompt).content               # call the model
            cache_file.write_text(out, encoding="utf-8")   # save for next run
            time.sleep(1.5)                                # stay under rate limit
            return out
        except Exception as e:                             # any API error
            print("      retry (%s) in %.0fs" % (type(e).__name__, delay))  # show it
            time.sleep(delay)                              # wait
            delay *= 2                                     # 4 -> 8 -> 16 -> 32
    return ""                                              # all attempts failed


# IN: raw reply -> OUT: parsed dict or None
# e.g. '```json\n{"question": "..."}\n```' -> {"question": "..."}
def extract_json(text):
    if not text:                                           # empty reply
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()  # remove ``` fences
    s, e = text.find("{"), text.rfind("}")                 # first "{" and last "}"
    if s == -1 or e == -1:                                 # no braces = not JSON
        return None
    try:
        return json.loads(text[s:e + 1])                   # parse the object
    except json.JSONDecodeError:                           # broken JSON
        return None


# Judge prompt: "do these passages STATE the answer?" -> {"answerable": ..., "evidence": ...}
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


# ── BLOCK: is the answer really absent? ─────────────────────────────────
# IN: llm, label, question, bm25, chunks, cache_key -> OUT: (True absent / False present / None unreadable, evidence)
# e.g. p034 "default maximum connections limit" -> (True, "")
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
    # ── Step 1: BM25 top 8 passages ──
    scores = bm25.get_scores(re.findall(r"[a-z0-9_]+", question.lower()))  # score every chunk vs question words
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:8]  # best 8 (2x app's k=4, stricter)
    passages = "\n\n---\n\n".join(
        "[Page %d]\n%s" % (chunks[i]["page"] + 1, chunks[i]["text"][:900]) for i in top)  # 1-based page + 900 chars

    # ── Step 2: ask the judge ──
    raw = call_llm(llm, VERIFY_PROMPT.format(
        doc_label=doc_label, question=question, passages=passages),  # fill the prompt
        cache_key="verify::" + cache_key)                  # "verify::" prefix, no clash with writer calls
    parsed = extract_json(raw)                             # read verdict
    if parsed is None:                                     # unreadable
        return None, ""          # unknown -- treat as not verified
    return (not parsed.get("answerable", True)), parsed.get("evidence", "")[:160]  # missing key = present (reject)


# ── BLOCK: main ─────────────────────────────────────────────────────────
# IN: optional question-file path -> OUT: same file + fresh verified probes
# e.g. py -3.12 eval/generate/generate_probes.py eval/data/qa_set_hard.json -> up to 9 probes (p034, ...)
def main():
    # ── Step 1: load the set, drop old probes ──
    args = [a for a in sys.argv[1:] if not a.startswith("-")]  # non-flag words
    qa_path = pathlib.Path(args[0]) if args else config.QA_SET_PATH  # default eval/data/qa_set.json
    qa = json.loads(qa_path.read_text(encoding="utf-8"))   # question list
    # Regenerating probes replaces any previous batch rather than appending to it,
    # so a rerun cannot accumulate near-duplicates.
    dropped = [q for q in qa if not q["answerable"]]       # old probes
    qa = [q for q in qa if q["answerable"]]                # keep real questions only
    if dropped:
        print("replacing %d existing probe(s)" % len(dropped))  # report
    next_id = len(qa)                                      # e.g. 33 answerable -> first probe p034

    llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0.8, max_tokens=600)  # 0.8 = varied retries; 600 token cap
    added = 0                                              # probes added, all PDFs

    # ── Step 2: per PDF, sample + BM25 index ──
    for doc_id, meta in config.DOCUMENTS.items():          # each PDF
        print("\n[%s] %s" % (doc_id, meta["label"]))
        pages = PyPDFLoader(meta["path"]).load()           # all pages
        usable = [p for p in pages if len(p.page_content.strip()) >= 450]  # pages with 450+ chars
        sample = "\n\n".join(p.page_content.strip()[:900] for p in usable[:4])  # 4 pages x 900 chars for writer

        # BM25 over the same chunks the real retriever uses -- no embedding API.
        chunks_path = config.CHROMA_EVAL_DIR / (
            config.index_key(doc_id, 1000, 200) + "_chunks.json")  # e.g. mysql_cs1000_co200_chunks.json
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))  # [{"page", "text"}, ...]
        bm25 = BM25Okapi([re.findall(r"[a-z0-9_]+", c["text"].lower()) for c in chunks])  # chunk -> word list

        # ── Step 3: one probe per angle ──
        accepted = 0                                       # accepted for this PDF
        accepted_text = []                                 # their texts, for duplicate check
        for n in range(PROBES_PER_DOC):                    # n = 0, 1, 2
            note = ""                                      # retry feedback, empty at first
            angle = FACT_ANGLES[n % len(FACT_ANGLES)]      # kind of missing fact (wraps)
            for attempt in range(MAX_ATTEMPTS):            # up to 4 tries
                # 3a: write a candidate
                raw = call_llm(llm, PROBE_PROMPT.format(
                    doc_label=meta["label"], doc_shape=meta["shape"],  # PDF name + shape
                    sample=sample, angle=angle, retry_note=note),       # sample, angle, feedback
                    cache_key="probe2::%s::%d::%d::%s" % (doc_id, n, attempt, angle))  # unique per try
                parsed = extract_json(raw)                 # parse reply
                if not parsed or not parsed.get("question"):  # unusable
                    continue                               # next try

                # 3b: reject near-duplicates
                q_text = parsed["question"].strip()        # the question
                dup, score = too_similar(q_text, accepted_text)  # vs accepted ones
                if dup:
                    print("   reject (too similar to an accepted probe, %.2f)" % score)
                    note = ("\nIMPORTANT: that is too close to a question already "
                            "accepted for this document. Ask about something "
                            "entirely different.\n")       # feedback for next try
                    continue

                # 3c: check the answer is really absent
                # Key the verification by the QUESTION, not the loop counters --
                # otherwise a rerun that generates a different question at the
                # same (n, attempt) reuses the previous question's verdict.
                absent, evidence = verify_absent(
                    llm, meta["label"], q_text, bm25, chunks,
                    cache_key=hashlib.sha256(q_text.encode()).hexdigest()[:16])  # key = question hash
                if absent is None:                         # verdict unreadable
                    print("   reject (verification unparseable)")
                    continue
                if not absent:                             # PDF answers it = not a trap
                    print("   reject -- document DOES answer it: %s" % evidence[:70])
                    note = RETRY_NOTE.format(found=evidence[:120])  # tell model what it found
                    continue

                # 3d: accept and save as unanswerable
                next_id += 1                               # next probe number
                added += 1                                 # total count
                accepted += 1                              # this PDF's count
                accepted_text.append(q_text)               # for later duplicate checks
                qa.append({
                    "id": "p%03d" % next_id,               # e.g. "p034"
                    "doc_id": doc_id,                      # which PDF
                    "question": q_text,                    # the trap
                    "reference_answer": "NOT IN DOCUMENT - the system should say it cannot find this.",
                    "fact_type": "unanswerable",           # probe type
                    "missing_detail": parsed.get("missing_detail", ""),  # what fact is missing
                    "fact_angle": angle,                   # which angle
                    "gt_page": None,                       # no correct page
                    "acceptable_pages": [],                # none
                    "answerable": False,                   # marks a probe
                    "reviewed": False,                     # not human checked
                    "verified_absent": True,               # passed the judge
                    "verified_by": "bm25-retrieval + llm check of retrieved passages",  # how
                })
                print("   OK  %s" % q_text[:64])
                break                                      # done with this probe
            else:                                          # for-else: runs only if all tries failed
                print("   gave up after %d attempts" % MAX_ATTEMPTS)
        print("   accepted %d/%d" % (accepted, PROBES_PER_DOC))  # per-PDF result

    # ── Step 4: save + summary ──
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")  # real questions + new probes
    n_unans = sum(1 for q in qa if not q["answerable"])    # probe count
    print("\n" + "=" * 70)
    print("Added %d verified-absent probes to %s" % (added, qa_path.name))
    print("  total questions   : %d" % len(qa))
    print("  unanswerable      : %d" % n_unans)
    # note: checked vs top 8 BM25 passages from all chunks, not the 4-page sample
    print("  every probe checked against the FULL document text, not a sample")


if __name__ == "__main__":  # only when run directly
    main()                  # start
