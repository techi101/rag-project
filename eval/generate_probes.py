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
# WHAT THIS FILE IS: the "trap maker" of the evaluation. It writes hallucination probes and then double-checks
# that each trap is real. A probe = a question that sounds like it belongs to the document but whose answer
# the document does NOT contain. The app should reply "I could not find this information in the uploaded document."
# If it gives an answer instead, it is hallucinating (inventing facts). Hallucination rate = share of probes answered.
# Real example from eval/qa_set_hard.json: p034 "What is the default maximum connections limit in a fresh MySQL
# installation?" (angle: "a numeric threshold, limit, or default setting"; verified_absent: true).
# How a trap is checked: BM25 (classic keyword search, no AI, no API) pulls the 8 most relevant passages, and a
# "judge model" (the larger gpt-oss-120b used as a grader) decides whether those passages actually STATE the answer.
# Overall flow: read a question file -> drop old probes -> per document: build BM25 over its chunks -> 3 times:
# LLM writes a probe for one "fact angle" -> reject near-duplicates (Jaccard) -> BM25 + judge check -> keep if absent
# -> save the file with the new probes added
#
# json = read/write JSON (plain-text lists and dicts); question files are JSON.
import json
# re = regular expressions (text patterns); used to split text into words and to strip ``` fences.
import re
# sys = system module: sys.argv (command-line words), sys.path (import search folders).
import sys
# time = sleep between calls and before retries.
import time
# hashlib = sha256 fingerprints, used for cache file names and cache keys.
import hashlib
# pathlib = easy file paths.
import pathlib

# Put eval/ on the import path so "import config" finds eval/config.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# config = eval/config.py: PDFs, model names, folders, stopwords, index_key().
import config

# Load GOOGLE_API_KEY and GROQ_API_KEY from the project's .env (stops if missing).
config.load_env()

# PyPDFLoader (LangChain) = reads a PDF, one Document per page.
from langchain_community.document_loaders import PyPDFLoader
# ChatGroq (LangChain) = client for chat models on Groq (gpt-oss-120b writes and judges the probes here).
from langchain_groq import ChatGroq
# rank_bm25 = a small library that implements BM25. BM25Okapi is the standard version: it scores each chunk by
# how many query words it contains, giving more weight to rare words and less to very long chunks.
from rank_bm25 import BM25Okapi

# 3 probes per document -> up to 9 in total (eval/README.md: the hard set has 9 probes).
PROBES_PER_DOC = 3
# Up to 4 tries to get each probe accepted.
MAX_ATTEMPTS = 4

# Left to itself the generator produces the same trap three times -- the first
# run gave three near-identical "what mAP did R-CNN achieve" probes (similarity
# 0.98). Each probe is steered to a different KIND of missing fact, and
# candidates too similar to an accepted one are rejected.
# Probe number n uses angle n (0, 1, 2), so the 3 probes ask for 3 different kinds of missing fact.
FACT_ANGLES = [
    "a numeric threshold, limit, or default setting",
    "a named person, organisation, product, or publication identifier",
    "a percentage, cost, duration, or date",
]

# Jaccard similarity = (words both questions share) / (all distinct words in either question), from 0 to 1.
# 0.55 or more = "same question reworded" -> rejected. eval/README.md says worst pairwise similarity went 1.00 -> 0.17.
DUP_THRESHOLD = 0.55        # Jaccard on content words


# IN: any text -> OUT: the set of its "content words" (lowercase, 3+ characters, not a stopword).
# WHY: compare questions by meaning-carrying words only, ignoring "what", "the", "does".
# Example: "What IoU threshold does R-CNN use?" -> {"iou", "threshold", "cnn"} ("what", "does", "use" are stopwords).
def content_words(text):
    stop = config.QUESTION_STOPWORDS
    # Regex [a-z][a-z0-9_]{2,} = a letter followed by 2 or more letters/digits/underscores, i.e. a word of 3+ characters.
    # So "iou", "threshold", "cnn" are kept; "r" (1 character) is not. Stopwords are removed.
    return {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if w not in stop}


# IN: a new question, list of already accepted questions -> OUT: (is it a duplicate?, similarity score).
# WHY: the generator tends to repeat one trap; this stops near-copies from being accepted.
# Example: two "what mAP did R-CNN achieve" variants share most words -> Jaccard about 0.98 -> (True, 0.98).
def too_similar(question, accepted):
    """Jaccard overlap catches reworded duplicates that string matching misses."""
    a = content_words(question)
    # No content words at all -> treat as a duplicate (score 1.0); such a question is useless.
    if not a:
        return True, 1.0
    # Compare with every accepted question.
    for prev in accepted:
        b = content_words(prev)
        # Jaccard = size of intersection (a & b) / size of union (a | b). Guard against dividing by zero.
        j = len(a & b) / len(a | b) if (a | b) else 0.0
        # Similar enough -> report it as a duplicate with its score.
        if j >= DUP_THRESHOLD:
            return True, j
    # No accepted question is close -> not a duplicate.
    return False, 0.0

# Cache folder eval/cache/probes: saved replies so re-runs make no new API calls.
PROBE_CACHE = config.CACHE_DIR / "probes"
PROBE_CACHE.mkdir(parents=True, exist_ok=True)


# Prompt template that asks for one trap question. {angle} = the kind of missing fact; {retry_note} = feedback
# from a failed try (or ""). {{ }} are doubled so .format() leaves real braces.
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

# Feedback added on a retry when the judge found the document DOES answer the previous question.
RETRY_NOTE = """
IMPORTANT: your previous attempt asked about something the document DOES state.
The passages contained: {found}
Ask about a different specific detail that the document does not state.
"""


# IN: llm client, prompt, cache_key -> OUT: reply text, or "" after 4 failed attempts.
# WHY: caching (free re-runs) + backoff (waits 4, 8, 16, 32 s) to survive rate limits on the free tier.
# Example: cache_key "probe2::mysql::0::0::a numeric threshold, limit, or default setting" -> one .txt cache file.
def call_llm(llm, prompt, cache_key):
    cache_file = PROBE_CACHE / (hashlib.sha256(cache_key.encode()).hexdigest()[:16] + ".txt")
    # Cache hit -> saved reply, no API call.
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    # First wait 4 seconds, doubled after each failure.
    delay = 4.0
    # Up to 4 attempts ("_" = the loop counter is not needed).
    for _ in range(4):
        # Call the model, save the reply, pause 1.5 s to stay under the rate limit.
        try:
            out = llm.invoke(prompt).content
            cache_file.write_text(out, encoding="utf-8")
            time.sleep(1.5)
            return out
        # Error -> print it, wait, double the wait, retry.
        except Exception as e:
            print("      retry (%s) in %.0fs" % (type(e).__name__, delay))
            time.sleep(delay)
            delay *= 2
    # Every attempt failed.
    return ""


# IN: raw model reply -> OUT: dict parsed from the JSON inside it, or None.
# WHY: the model sometimes wraps JSON in ```json fences or prose.
# Example: '```json\n{"question": "...", "missing_detail": "..."}\n```' -> {"question": ..., "missing_detail": ...}
def extract_json(text):
    # Empty reply -> nothing to parse.
    if not text:
        return None
    # Regex: ^```(?:json)? = backticks plus optional "json" at a line start, | = OR, ```$ = backticks at a line end.
    # re.MULTILINE makes ^ and $ apply to every line. These fences are removed.
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    # The JSON object = text from the first "{" to the last "}".
    s, e = text.find("{"), text.rfind("}")
    # No braces -> not JSON.
    if s == -1 or e == -1:
        return None
    # Parse; broken JSON -> None.
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


# Prompt template for the judge: shows the question and the retrieved passages and asks only
# "do these passages STATE the answer?" -> JSON {"answerable": true/false, "evidence": "..."}.
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


# IN: llm, document label, question, BM25 index, the chunks list, cache_key
#   -> OUT: (True = answer really absent / False = document answers it / None = judge reply unreadable, evidence text)
# WHY: verifies a probe under the same conditions the app faces: retrieve passages, then see if they hold the fact.
# Example: p034 "default maximum connections limit" -> no retrieved passage states it -> (True, "").
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
    # Split the question into lowercase words (regex [a-z0-9_]+ = runs of letters, digits or underscores) and get a
    # BM25 score for every chunk.
    scores = bm25.get_scores(re.findall(r"[a-z0-9_]+", question.lower()))
    # Indexes of the 8 highest-scoring chunks (sort chunk numbers by score, biggest first, keep 8).
    # 8 passages = twice the app's k=4, so the check is stricter than what the app would see.
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:8]
    # Build the passages text: "[Page N]" (1-based) + the first 900 characters of each chunk, separated by "---".
    passages = "\n\n---\n\n".join(
        "[Page %d]\n%s" % (chunks[i]["page"] + 1, chunks[i]["text"][:900]) for i in top)

    # Ask the judge. The cache key starts with "verify::" so it never clashes with question-writing calls.
    raw = call_llm(llm, VERIFY_PROMPT.format(
        doc_label=doc_label, question=question, passages=passages),
        cache_key="verify::" + cache_key)
    # Parse the judge's JSON reply.
    parsed = extract_json(raw)
    # Unreadable verdict -> None (the caller rejects the probe).
    if parsed is None:
        return None, ""          # unknown -- treat as not verified
    # absent = NOT answerable. If the key is missing, default answerable=True, so a doubtful probe counts as present
    # (rejected). Evidence is cut to 160 characters.
    return (not parsed.get("answerable", True)), parsed.get("evidence", "")[:160]


# IN: optional question-file path on the command line -> OUT: the same file with fresh verified probes appended.
# WHY: the hallucination rate needs questions the document truly cannot answer.
# Example: py -3.12 eval/generate_probes.py eval/qa_set_hard.json -> adds up to 9 probes (ids like p034).
def main():
    # Non-flag command-line words; the first one is the question file. None given -> eval/qa_set.json.
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    qa_path = pathlib.Path(args[0]) if args else config.QA_SET_PATH
    # Load the question list.
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    # Regenerating probes replaces any previous batch rather than appending to it,
    # so a rerun cannot accumulate near-duplicates.
    # Split the set: old probes (answerable False) are dropped, answerable questions are kept.
    dropped = [q for q in qa if not q["answerable"]]
    qa = [q for q in qa if q["answerable"]]
    # Say how many old probes are being replaced.
    if dropped:
        print("replacing %d existing probe(s)" % len(dropped))
    # New probe ids continue after the number of answerable questions (e.g. 33 answerable -> first probe is p034).
    next_id = len(qa)

    # Writer and judge: gpt-oss-120b. temperature 0.8 = high variety, so retries give genuinely new questions.
    # max_tokens 600 caps each reply (including hidden reasoning).
    llm = ChatGroq(model=config.JUDGE_MODEL, temperature=0.8, max_tokens=600)
    # added = probes added across all documents.
    added = 0

    # Loop over each document.
    for doc_id, meta in config.DOCUMENTS.items():
        print("\n[%s] %s" % (doc_id, meta["label"]))
        # Load pages, keep those with 450+ characters, and build a sample (first 900 characters of the first 4 such pages)
        # for the probe-writing prompt.
        pages = PyPDFLoader(meta["path"]).load()
        usable = [p for p in pages if len(p.page_content.strip()) >= 450]
        sample = "\n\n".join(p.page_content.strip()[:900] for p in usable[:4])

        # BM25 over the same chunks the real retriever uses -- no embedding API.
        # Load the chunks saved by build_index.py for the baseline chunking (chunk_size 1000, overlap 200), for example
        # eval/chroma_eval/mysql_cs1000_co200_chunks.json. Each chunk has "page" and "text".
        chunks_path = config.CHROMA_EVAL_DIR / (
            config.index_key(doc_id, 1000, 200) + "_chunks.json")
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        # Build the BM25 index: each chunk becomes a list of lowercase words.
        bm25 = BM25Okapi([re.findall(r"[a-z0-9_]+", c["text"].lower()) for c in chunks])

        # accepted = probes accepted for this document; accepted_text = their questions (for the duplicate check).
        accepted = 0
        accepted_text = []
        # One probe per angle: n = 0, 1, 2.
        for n in range(PROBES_PER_DOC):
            # note = feedback for the next try ("" at first). angle = which kind of missing fact to ask for
            # (n % 3 wraps around if PROBES_PER_DOC were ever larger than the list).
            note = ""
            angle = FACT_ANGLES[n % len(FACT_ANGLES)]
            # Up to 4 tries for this probe.
            for attempt in range(MAX_ATTEMPTS):
                # Ask for a candidate probe. The cache key holds doc, probe number, attempt and angle.
                raw = call_llm(llm, PROBE_PROMPT.format(
                    doc_label=meta["label"], doc_shape=meta["shape"],
                    sample=sample, angle=angle, retry_note=note),
                    cache_key="probe2::%s::%d::%d::%s" % (doc_id, n, attempt, angle))
                # Parse the reply.
                parsed = extract_json(raw)
                # Unusable reply -> next try.
                if not parsed or not parsed.get("question"):
                    continue

                # Check the candidate against probes already accepted for this document.
                q_text = parsed["question"].strip()
                dup, score = too_similar(q_text, accepted_text)
                # Too similar -> reject, tell the model to ask about something entirely different, and try again.
                if dup:
                    print("   reject (too similar to an accepted probe, %.2f)" % score)
                    note = ("\nIMPORTANT: that is too close to a question already "
                            "accepted for this document. Ask about something "
                            "entirely different.\n")
                    continue

                # Key the verification by the QUESTION, not the loop counters --
                # otherwise a rerun that generates a different question at the
                # same (n, attempt) reuses the previous question's verdict.
                # Check with BM25 + judge whether the document actually answers the question.
                absent, evidence = verify_absent(
                    llm, meta["label"], q_text, bm25, chunks,
                    cache_key=hashlib.sha256(q_text.encode()).hexdigest()[:16])
                # Judge reply unreadable -> reject and try again.
                if absent is None:
                    print("   reject (verification unparseable)")
                    continue
                # The document DOES answer it -> not a valid trap. Show the evidence and feed it back in the retry note.
                if not absent:
                    print("   reject -- document DOES answer it: %s" % evidence[:70])
                    note = RETRY_NOTE.format(found=evidence[:120])
                    continue

                # Passed every check: give it the next id, count it, remember its text, and save it as an unanswerable question
                # with no correct page.
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
                # Accepted -> stop trying for this probe.
                break
            # for ... else: this "else" runs only if the attempt loop finished WITHOUT a break, i.e. all 4 tries failed.
            else:
                print("   gave up after %d attempts" % MAX_ATTEMPTS)
        print("   accepted %d/%d" % (accepted, PROBES_PER_DOC))

    # Save the file: answerable questions + the new probes.
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    # Count probes and print a summary. Note: each probe was checked against the top 8 BM25 passages drawn from all of
    # the document's chunks (not only the 4-page sample the writer saw).
    n_unans = sum(1 for q in qa if not q["answerable"])
    print("\n" + "=" * 70)
    print("Added %d verified-absent probes to %s" % (added, qa_path.name))
    print("  total questions   : %d" % len(qa))
    print("  unanswerable      : %d" % n_unans)
    print("  every probe checked against the FULL document text, not a sample")


# Run main() only when the file is started directly.
if __name__ == "__main__":
    main()
