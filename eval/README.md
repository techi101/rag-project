# Evaluation Harness — DocuMind AI

Measures whether the RAG pipeline actually works, instead of assuming it does.

> **Status: partial.** 20 of 48 questions scored before Google's embedding free
> tier hit its **daily** quota. More importantly, verification showed the first
> question set does not discriminate between retrievers — so a harder set was
> built and is ready to run. See [What the numbers do and don't say](#what-the-numbers-do-and-dont-say).

---

## Why this exists

The app worked "fine when I tried it." That is not a measurement, and it cannot
answer the first question anyone sensible asks: *how do you know your retrieval
works?*

Four things are measured, deliberately kept apart:

| Metric | Question it answers |
|---|---|
| **Recall@k** | Did the retriever surface the page the answer lives on? |
| **Answer accuracy** | Given that context, did the LLM write the right answer? |
| **Hallucination rate** | On questions the document *cannot* answer, did it invent one? |
| **Truncation rate** | Did generation hit the token ceiling and get cut off? |

Separating recall from accuracy is the point. If accuracy is bad, you need to
know whether the retriever handed the model the wrong pages or the model fumbled
the right ones — those have opposite fixes.

---

## The headline finding: the first test set was too easy

**BM25 — plain keyword matching, no embeddings, no API — scored 97.7% recall@4 on
the original question set** (100% on two of the three documents, 93% on the third).
The dense vector retriever scored 100% on the questions it was run on.

That does not mean both are excellent. It means the questions could not tell them
apart. Questions generated *from* a page inherit that page's vocabulary, so a
string matcher finds it without understanding anything.

Measured cause: **73% mean word overlap** between each question and its own
source page; 37% of questions shared ≥80% of their words.

### The fix: a paraphrased question set

`generate_qa_hard.py` demands questions that avoid the page's wording — synonyms,
indirection, description instead of naming — then **filters objectively**: any
question measuring above 55% overlap is rejected, the offending shared words are
fed back, and it is regenerated (up to 3 attempts). The filter uses the same
function the verifier reports with, so the generator cannot grade itself on a
friendlier scale.

| | Original set | Hard set |
|---|---|---|
| Questions | 48 (43 answerable + 5 probes) | 42 (33 answerable + 9 probes) |
| Mean lexical overlap | 73% | **21%** |
| **BM25-only recall@4** | **97.7%** | **57.6%** |

Per document, BM25-only recall@4:

| Set | MySQL | BI | ObjRec | Overall |
|---|---|---|---|---|
| Easy | 100% | 100% | 93% | 97.7% |
| Hard | 50% | 78% | 50% | **57.6%** |

A set a 1994 keyword algorithm cannot lose on measures nothing. At 57.6% there is
finally room for a semantic retriever to prove it is worth the API call.

> Example of the difference — same fact, both sets:
> - Easy: *"What does Mask R-CNN add to Faster R-CNN?"*
> - Hard: *"When extending the earlier object-detecting system that predicts a
>   category and a surrounding rectangle, what additional output is produced?"*

### Hallucination probes — verified, not guessed

9 questions the documents genuinely cannot answer (`generate_probes.py`). Each is
verified the way the real system would answer it: BM25 retrieves the 8 most
relevant passages and a large model judges whether they actually **state** the
fact. Only probes that survive are kept.

Two methods were tried and one discarded:

- **Rejected approach** — check whether words from the question appear anywhere in
  the document. This rejected *every* candidate: any MySQL question contains
  words like `default` and `columns`. Word presence is not evidence that a fact
  is stated.
- **Kept approach** — retrieve, then judge the retrieved passages. Same
  conditions the app faces.

The generator also collapses to one idea if left alone — the first batch produced
three near-identical "what mAP did R-CNN achieve" probes (pairwise similarity
0.98). Fixed by steering each probe to a different *kind* of missing fact
(numeric threshold / named entity / percentage-or-date) and rejecting candidates
with high Jaccard overlap against already-accepted ones. Worst pairwise
similarity went from **1.00 to 0.17**.

The strongest traps have their subject present but the specific fact absent:

> *"What IoU threshold does the original R-CNN use for non-maximum suppression?"*
> The deck discusses IoU thresholds, but never mentions non-maximum suppression.
> A weak system retrieves the IoU page and invents a number.

**The open question this now makes answerable:** does your Google embedding
pipeline actually beat free, local, instant keyword search? Nobody knows yet.
That comparison runs when the quota resets, and either answer is worth having.

---

## The corpus

| Document | Pages | Characters | Shape |
|---|---|---|---|
| MySQL Handbook | 72 | 40k | Technical reference — code, tables, short pages |
| BI Exam Companion | 89 | 202k | Dense continuous prose |
| Object Recognition Slides | 60 | 20k | Lecture slides — sparse, fragmented |

**Why the contrast matters — measured, not assumed:**

| Document | chunks @500 | @1000 | @2000 |
|---|---|---|---|
| MySQL | 115 | **71** | **71** |
| BI | 522 | 275 | 152 |
| Object Recognition | 72 | **60** | **60** |

On two of three documents, raising `chunk_size` from 1000 to 2000 **changes
nothing**. LangChain's splitter works page by page and never merges across pages
(verified: 0 of 792 chunks span two pages), and those documents' pages are
already smaller than the chunk size. Chunk size only matters on documents dense
enough to split.

A corpus of only sparse documents would have "proved" chunk size is irrelevant.
That would have been wrong, and invisible.

---

## Verification — testing the harness's own assumptions

Every claim here rests on something that could be wrong. `verify_*.py` tries to
break each one. **Four real defects were found in this harness and fixed.**

| Check | Result |
|---|---|
| Chunks never span pages | PASS — 0 of 792 chunks span two pages |
| Recall ground truth is unique | **FAILED** → fixed (below) |
| Refusal detector cannot misfire | **FAILED** → fixed (below) |
| Chunk-size sensitivity | PASS — measured, two documents insensitive |
| Questions not lexically leaky | **FAILED** → drove the hard set |
| RRF fusion logic | PASS — 7 synthetic cases |
| Judge accuracy | PASS — 10/10 |
| Judge consistency | PASS — 10/10 identical on repeat |

### Defect 1 — recall punished correct retrievals

Recall originally credited only the single page a question was written from. But
`CREATE DATABASE startersql;` appears on **both page 4 and page 65** of the MySQL
handbook. Returning page 65 was scored a MISS despite being a correct answer.

Fixed by recording every page that genuinely contains the answer
(`annotate_pages.py`) and reporting recall two ways:

- **strict** — only the originating page counts (lower bound)
- **lenient** — any page containing the answer counts (upper bound)

18.6% of answerable questions have more than one valid page. Quoting both bounds
is the honest presentation.

### Defect 2 — the refusal detector fired on real answers

Grading hallucination depends on detecting refusals. The first version matched
the substring `"not in the"`, which fires on *"the index is not in the buffer
pool"*. A second version still fired on a 340-character genuine answer that
happened to contain the phrase mid-sentence.

Fixed: the phrase must appear in the first 100 characters **and** the whole
answer must be under 200 characters. The template refusal is 59 characters; real
answers that merely mention the phrase are longer and mention it later.

### Defect 3 — my own threshold was set by taste

The lexical-overlap check originally passed at 73% because the threshold was 75%,
a number chosen for no reason. BM25's 97.7% score is empirical proof that 73% is
too high. The set-level threshold in `verify_method.py` is now 60%, justified by
that evidence in the code comment. (Separately, `generate_qa_hard.py` rejects any
single question above 55%.)

### Defect 4 — failures were not reproducible

Results stored only the *page numbers* of retrieved chunks. A page holds several
chunks, so the exact context a failure saw could not be rebuilt — which cost real
time when investigating q018. Results now store the retrieved chunk text.

---

## Finding: a blank answer is real but rare — an earlier claim retracted

**First claim (wrong):** "`max_tokens=1024` is too low; blank answers are a coin
flip." Asserted from one observation plus four ad-hoc probes. A proper test
contradicts it. The retraction is kept here on purpose — an eval that only
records confirmations is not an eval.

### What is actually true

In the baseline run, question q018 returned an **empty string** with
`finish_reason="length"` and `completion_tokens=1024`. Retrieval had worked; the
correct page was ranked first. The user would have seen a blank reply.

### What the measurement says

`verify_tokens.py` — 4 questions × 5 trials × 3 ceilings = 60 generations with
realistic retrieved context:

| Ceiling | Empty answers | Hit ceiling | Completion tokens (median / max) |
|---|---|---|---|
| 1024 | **0 / 20** | 0 / 20 | 170 / 391 |
| 2048 | 0 / 20 | 0 / 20 | 179 / 415 |
| 4096 | 0 / 20 | 0 / 20 | 189 / 414 |

Reasoning plus answer typically costs ~170 tokens, peaking at 391 — roughly 2.6×
headroom under the existing 1024 ceiling. The earlier "462–1028 tokens" figure
came from unrepresentative probes, one with no retrieved context at all.

### Corrected conclusion

- Blank answers occur at roughly **1 in 20 questions**, not as a coin flip.
- Raising `max_tokens` is **not proven** to fix it — no empty response occurred at
  any ceiling tested, including 1024.
- The defensible fix ignores `max_tokens`: **treat empty `content` as a failure
  and retry once**, which handles the symptom whatever its cause.

---

## Results so far

**Configuration `baseline`** — exactly what `rag_engine.py` ships:
`chunk_size=1000`, `chunk_overlap=200`, `k=4`, dense-only similarity.
Values are from `results/summary.json`, recomputed from the cache.

| Metric | Easy set | Hard set |
|---|---|---|
| Coverage | 17 of 48 (15 answerable + 2 probes) | 10 of 42 (10 answerable, 0 probes) |
| Recall@4 (strict = lenient) | 100.0% | 100.0% |
| MRR | 0.772 | 0.883 |
| Answer accuracy | 100.0% (15/15) | 100.0% (10/10) |
| Hallucination rate | 0.0% (2 probes, both refused) | not measured (no probes scored) |
| Truncation rate | 0.0% | 0.0% |
| Mean latency | 1.51 s | 0.68 s |
| BM25 recall@4 on the same questions | — | 40.0% (4/10) |

All 27 scored questions are from the MySQL handbook. An earlier draft of this
section quoted a 20-question run (94.4% accuracy, MRR 0.810, one truncated
answer) whose rows are no longer in the results cache; those numbers are
superseded by the table above.

### What the numbers do and don't say

**Do not quote these as a result.** Reasons, in order of severity:

1. **The question set does not discriminate.** BM25 scores 97.7% on it. This
   number describes the questions, not the retriever. Superseded by the hard set.
2. **Coverage is 35% (easy) and 24% (hard)** — 17 of 48 and 10 of 42 questions
   ran before the quota died.
3. **The sample is one document** — every scored question is MySQL, a 71-chunk
   corpus with one chunk per page. No prose or slide-deck questions ran.
4. **n=2 for hallucination.** A 0% rate on two probes means very little.
5. **Single trial per question**, and measured run-to-run variance is real.

The honest summary today: *the pipeline retrieves and answers correctly on easy
questions, and the harness needed four fixes before its own numbers could be
trusted.* That is a real finding. "100% recall" is not.

---

## Configurations to compare

| Name | chunk | overlap | k | retrieval | Status |
|---|---|---|---|---|---|
| `baseline` | 1000 | 200 | 4 | dense | partial (17/48 easy, 10/42 hard) |
| `chunk500` | 500 | 100 | 4 | dense | pending |
| `chunk2000` | 2000 | 400 | 4 | dense | pending — no-op on 2 of 3 documents |
| `k8` | 1000 | 200 | 8 | dense | pending |
| `hybrid` | 1000 | 200 | 4 | BM25 + dense (RRF) | pending — fusion logic verified |
| `hybrid_k8` | 1000 | 200 | 8 | BM25 + dense (RRF) | pending |

Configs sharing a chunking scheme share an index — 3 indexes per document, not 6.

---

## Running it

```bash
py -3.12 eval/generate_qa.py                   # easy set            (cached)
py -3.12 eval/generate_qa_hard.py              # paraphrased set     (cached)
py -3.12 eval/generate_probes.py <set>         # verified unanswerables
py -3.12 eval/verify_qa.py --fix               # QC: dupes, leaks, false probes
py -3.12 eval/annotate_pages.py [path]         # multi-page ground truth
py -3.12 eval/verify_method.py                 # assumption checks   (no API)
py -3.12 eval/verify_retrieval.py [--hard]     # RRF + BM25 baseline (no API)
py -3.12 eval/verify_judge.py                  # judge validation    (Groq only)
py -3.12 eval/verify_tokens.py                 # token ceiling test  (Groq only)
py -3.12 eval/build_index.py [--chunks-only]   # vector stores       (cached)
py -3.12 eval/run_eval.py baseline
py -3.12 eval/run_eval.py --cached-only        # re-score, zero API calls
```

`--chunks-only` splits text without embedding, which is enough for BM25 and runs
with the embedding quota exhausted.

Every LLM call and scored question is cached. Metrics are recomputed from cache
on every run, so changing how recall is defined re-scores old results for free.

### Known blocker

Google's embedding free tier enforces a **per-day** quota (`RESOURCE_EXHAUSTED`,
`PerDay`), exhausted partway through the first baseline run. Remaining when it
resets:

- 3 indexes: `objrec` at 500 / 1000 / 2000
- 31 questions of `baseline` on the easy set
- 32 questions of `baseline` on the hard set, and the hard set across other configurations
- 5 further configurations

Groq's quota was never the constraint — it is still working.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Corpus, configurations, models, shared metric functions |
| `generate_qa.py` | Drafts the original question set |
| `generate_qa_hard.py` | Paraphrased set with objective overlap filtering |
| `generate_probes.py` | Verified unanswerable questions (hallucination traps) |
| `verify_qa.py` | QC: false probes, duplicates, leaked ground truth |
| `annotate_pages.py` | Records every page that answers each question |
| `verify_method.py` | Adversarial checks on harness assumptions |
| `verify_retrieval.py` | RRF fusion tests + BM25-only baseline |
| `verify_judge.py` | Grader accuracy and consistency |
| `verify_tokens.py` | Token ceiling distribution |
| `build_index.py` | One vector store per chunking scheme |
| `run_eval.py` | Scoring harness |
| `qa_set.json` / `qa_set_hard.json` | The question sets (automatically checked; `reviewed` is still `false` on all 90 — not hand-reviewed) |
| `results/`, `cache/` | Raw results and resumable cache |

Nothing here modifies `app.py`, `rag_engine.py`, or `chroma_store/`. The harness
imports the live `SYSTEM_PROMPT` and `format_docs` read-only, so scores describe
the real application rather than a lookalike.
