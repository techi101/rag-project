# What the evaluation found

> Back to the [README](../README.md). The harness itself is documented in [`eval/README.md`](../eval/README.md);
> current numbers are in the README's [results table](../README.md#current-numbers-partial--read-the-caveats).

---

### How the test set is built

Each question is generated from one specific page, and **that page number is recorded as
ground truth**. If no retrieved chunk comes from it, retrieval failed — regardless of what
the model then wrote. That makes retrieval scoring objective rather than a matter of opinion.

The corpus is three documents with deliberately different shapes, so results reveal *where* a
setting helps instead of averaging into mush:

| Document | Pages | Characters | Shape |
|---|---|---|---|
| MySQL Handbook | 72 | 40k | Technical reference — code, tables, short pages |
| BI Exam Companion | 89 | 202k | Dense continuous prose |
| Object Recognition Slides | 60 | 20k | Lecture slides — sparse, fragmented |

---

## Findings

### 1 · The first test set measured nothing

**BM25 — plain keyword matching, no embeddings, no AI, invented in 1994 — scored 97.7%
recall@4 on the original question set.** Identical to the dense vector retriever.

That is not two excellent systems. That is a test that cannot tell them apart. Questions
generated *from* a page inherit that page's vocabulary, so a string matcher finds it without
understanding anything. Measured cause: **73% mean word overlap** between each question and
its own source page.

**The fix:** a second question set that paraphrases instead of quoting — synonyms, indirection,
description instead of naming — with an objective filter that rejects any question measuring
above 55% overlap and regenerates it.

| | Original set | Hard set |
|---|---|---|
| Questions | 48 | 42 |
| Mean lexical overlap | 73% | **21%** |
| **BM25-only recall@4** | **97.7%** | **57.6%** |

> Same fact, both sets:
> - *Easy:* "What does Mask R-CNN add to Faster R-CNN?"
> - *Hard:* "When extending the earlier object-detecting system that predicts a category and a
>   surrounding rectangle, what additional output is produced?"

Running a dumb baseline as a control is the single most useful thing in this project. Without
it, the headline would have been a meaningless "100% recall."

### 2 · Chunk size does nothing on sparse documents

| Document | chunks @500 | @1000 | @2000 |
|---|---|---|---|
| MySQL | 115 | **71** | **71** |
| BI | 522 | 275 | 152 |
| Object Recognition | 72 | **60** | **60** |

The splitter never merges across pages, and two of these documents have pages shorter than
1000 characters — so every page is already one chunk and raising the limit changes nothing.
A corpus of only sparse documents would have "proved" chunk size is irrelevant. It isn't; it
only matters when pages are dense enough to split.

### 3 · Hallucination is measured, not hoped for

Nine questions the documents genuinely cannot answer. Each is verified the way the real system
would answer it — BM25 retrieves the 8 most relevant passages, and a large model checks whether
they actually *state* the fact. Only probes that survive are kept.

The strongest traps have their subject present but the specific fact absent:

> *"What IoU threshold does the original R-CNN use for non-maximum suppression?"*
> The slide deck discusses IoU thresholds, but never mentions non-maximum suppression. A weak
> system retrieves the IoU page and invents a number.

### 4 · Four defects were found in the harness itself

An evaluation you haven't attacked is just a second thing that might be wrong.

| Defect | What it would have caused |
|---|---|
| Recall credited only one page | Punished the retriever for finding a *different* page that also answers the question (18.6% of questions have several) |
| Refusal detector matched `"not in the"` | Graded genuine answers as refusals, understating hallucination |
| A failed embedding batch left a partial index | `bi_cs1000_co200` held **200 of 275 vectors** while looking complete — recall measured against a document missing 27% of its content |
| Rate-limited questions were silently dropped | A run reported results over 20 of 48 questions without saying so |

The third is the one worth remembering. The "is this index built?" check asked *does the
directory exist?* — which a half-written store passes forever. It now counts vectors and
compares against chunks, and a batch that exhausts its retries raises instead of continuing.

### 5 · A claim made, tested, and retracted

An early finding claimed `max_tokens=1024` was too low and blank answers were "a coin flip."
Sixty controlled generations say otherwise: **0 empty answers in 20 trials at 1024**, median
cost 170 tokens against the ceiling. The claim was wrong and is marked as such in
[`eval/README.md`](../eval/README.md) rather than quietly deleted.

What survives is narrower and true: one question in one run *did* return an empty string with
`finish_reason="length"`. Roughly 1 in 20. The defensible fix is not raising `max_tokens` —
no ceiling tested prevented it — but treating empty content as a failure and retrying once.
