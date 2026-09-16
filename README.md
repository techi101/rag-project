# 🧠 DocuMind AI — Chat with Any PDF Document

> A full-stack **Retrieval-Augmented Generation (RAG)** application that lets you hold an
> intelligent conversation with any PDF. Upload a financial report, research paper, legal
> contract or textbook, ask questions in plain English, and get answers grounded in the
> document with page citations.
>
> **Built to run entirely on free tiers** — no credit card, no paid API.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.3-green?style=flat-square)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-red?style=flat-square)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-GPT__OSS__20B-orange?style=flat-square)](https://groq.com)
[![Google](https://img.shields.io/badge/Embeddings-gemini--embedding--001-purple?style=flat-square)](https://aistudio.google.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-yellow?style=flat-square)](https://trychroma.com)

---

## Table of Contents

- [What this project is](#what-this-project-is)
- [Why RAG — the problem it solves](#why-rag--the-problem-it-solves)
- [Architecture](#architecture)
- [The pipeline, step by step](#the-pipeline-step-by-step)
- [Tech stack and why each piece was chosen](#tech-stack-and-why-each-piece-was-chosen)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Evaluation — does it actually work?](#evaluation--does-it-actually-work)
- [What the evaluation found](#what-the-evaluation-found)
- [Known limitations](#known-limitations)
- [What I would build next](#what-i-would-build-next)

---

## What this project is

Two things, and the second is the more interesting one.

**1. A working RAG application.** Upload a PDF, it gets chunked, embedded and stored in a
local vector database. Ask a question, the most relevant passages are retrieved and handed
to an LLM that answers *only* from those passages, citing page numbers. Every processed PDF
is cached by content hash, so re-uploading the same file is instant and costs nothing.

**2. An evaluation harness that measures whether it works.** Most RAG demos stop at "it
seemed fine when I tried it." This one ships with `eval/` — a suite that scores retrieval
recall, answer accuracy, hallucination rate and truncation rate on a purpose-built question
set, with a dumb keyword-search control to check the clever parts are earning their keep.

That second part is where the engineering is. It is documented in detail in
**[`eval/README.md`](eval/README.md)**, and summarised [below](#evaluation--does-it-actually-work).

---

## Why RAG — the problem it solves

Ask a public LLM *"what were the key findings in my company's Q3 internal audit?"* and it
will fail, or worse, invent something. It has never seen your private documents, and it has
no way to tell you that convincingly.

Fine-tuning a model on your documents is expensive, slow, and has to be redone every time a
document changes. **Retrieval-Augmented Generation** sidesteps both problems: leave the model
alone, and instead *find* the relevant passages at question time and paste them into the
prompt. The model becomes a reader rather than a memoriser.

That reframes the engineering problem in a useful way:

> **A RAG system is only as good as its retrieval.** If the right passage never reaches the
> prompt, no amount of prompt engineering saves the answer.

Which is exactly why this project has an evaluation harness that scores retrieval
*separately* from answer quality.

---

## Architecture

```
                         ┌──────────────────────────┐
     PDF upload  ───────▶│  PyPDFLoader             │   one Document per page
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │  RecursiveCharacter      │   1000 chars, 200 overlap
                         │  TextSplitter            │   split per page
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │  Google                  │   3072-dim vectors
                         │  gemini-embedding-001    │
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │  ChromaDB                │   persisted per PDF hash
                         │  chroma_store/<sha256>/  │   → re-upload is free
                         └────────────┬─────────────┘
                                      │
   user question ─────────────────────┤
                                      ▼
                         ┌──────────────────────────┐
                         │  similarity search (k=4) │   cosine over embeddings
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │  Groq · gpt-oss-20b      │   grounded prompt,
                         │  temperature 0.1         │   "answer only from context"
                         └────────────┬─────────────┘
                                      ▼
                            answer + page citations
```

---

## The pipeline, step by step

### 1 · LOAD — read the PDF

```python
loader = PyPDFLoader(pdf_path)
pages  = loader.load()          # one LangChain Document per page
```

Page numbers survive in `metadata["page"]`, which is what makes page citations — and later,
objective retrieval scoring — possible.

### 2 · CHUNK — break it into retrievable pieces

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(pages)
```

Whole documents cannot be sent to an LLM on every question — context windows are finite and
long contexts degrade answer quality. The 200-character overlap stops a fact being severed at
a chunk boundary. The separator list means it prefers to break at paragraphs, then lines, then
sentences, and only splits mid-word as a last resort.

> **Measured behaviour worth knowing:** the splitter works *page by page* and never merges
> across pages — verified, 0 of 792 chunks span two pages. On documents whose pages are
> already shorter than `chunk_size`, raising the chunk size does nothing at all. See
> [the evaluation](#what-the-evaluation-found).

### 3 · EMBED — turn text into meaning-vectors

```python
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

Each chunk becomes a 3072-dimensional vector positioned so that passages *about the same
thing* land near each other — regardless of shared vocabulary. This is what lets the system
answer a question phrased completely differently from the source text.

### 4 · STORE — persist to ChromaDB

```python
pdf_hash    = hashlib.sha256(file_bytes).hexdigest()[:12]
persist_dir = f"./chroma_store/{pdf_hash}"
vector_store = Chroma.from_documents(chunks, embedding_model, persist_dir)
```

Keying the store by a hash of the file's **contents** (not its name) means the same document
uploaded twice — or renamed — reuses the existing index. Embedding is the expensive step, so
this is the single biggest cost saving in the app.

### 5 · RETRIEVE — find the relevant passages

The question is embedded with the same model, and ChromaDB returns the `k=4` nearest chunks
by cosine similarity.

### 6 · GENERATE — answer, grounded

```python
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1, max_tokens=1024)
```

The retrieved passages and the question go to the model under a system prompt that is strict
about grounding:

```
1. ONLY use information explicitly found in the provided context.
2. If the answer is not in the context, clearly say:
   "I could not find this information in the uploaded document."
5. When you use information from a specific part, mention the page number.
```

Rule 2 is the anti-hallucination rule, and the evaluation measures whether the model actually
obeys it — see [hallucination probes](#3-hallucination-is-measured-not-hoped-for).

---

## Tech stack and why each piece was chosen

| Layer | Choice | Why this one |
|---|---|---|
| **UI** | Streamlit | Chat UI in pure Python; deploys free on Streamlit Cloud |
| **Orchestration** | LangChain (LCEL) | Composable chains; loaders and splitters that already handle PDF edge cases |
| **PDF parsing** | PyPDFLoader (pypdf) | Preserves per-page metadata, which page citations depend on |
| **Embeddings** | Google `gemini-embedding-001` | 3072-dim, strong quality, generous free tier |
| **Vector DB** | ChromaDB | Runs locally with no server; persists to disk; no hosting cost |
| **LLM** | Groq `openai/gpt-oss-20b` | Free tier, and Groq's LPU hardware makes generation near-instant |
| **Grading (eval only)** | Groq `openai/gpt-oss-120b` | A *larger* model grades the smaller one — a model grading its own output inflates scores |

### One thing to know about the generation model

`gpt-oss-20b` is a **reasoning model**: it spends completion tokens on hidden reasoning
*before* emitting any visible text. Measured cost is ~170 tokens median, 391 worst case,
against the configured 1024 ceiling. That is comfortable headroom — but it does mean an
answer can occasionally come back empty, which the evaluation quantifies rather than guesses
at.

---

## Quick start

### 1 · Install

```bash
pip install -r requirements.txt
```

### 2 · Get two free API keys

| Key | Where | Used for |
|---|---|---|
| `GOOGLE_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | Embeddings (starts `AIza…`) |
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | Generation (starts `gsk_…`) |

Create a `.env` in the project root — it is gitignored and never leaves your machine:

```env
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

You can also paste keys into the app's sidebar instead, which is how the deployed version
works so that no host key is exposed to users.

### 3 · Run

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`.

> **Free tier note:** Google's embedding API enforces a **per-day** request quota and a
> per-minute rate limit. Processing a large PDF can exhaust the daily allowance in one go,
> after which uploads fail with `RESOURCE_EXHAUSTED` until it resets. This is a quota limit,
> not a bug.

---

## Project structure

```
rag-project/
│
├── app.py                  ← THE FACE: entire Streamlit UI — upload, chat, sidebar,
│                              session state, responsive CSS
│
├── rag_engine.py           ← THE BRAIN: all RAG logic
│     ├── get_pdf_hash()            content hash → per-PDF cache directory
│     ├── process_pdf()             load → chunk → embed → store
│     ├── load_existing_vectorstore()   skip re-embedding a known PDF
│     ├── create_qa_chain()         retriever + LLM + grounded prompt
│     ├── run_qa()                  one Q&A turn, with chat history
│     └── generate_summary()        document summary on upload
│
├── requirements.txt        ← dependencies
├── .env                    ← your API keys (gitignored)
├── .gitignore
├── chroma_store/           ← auto-created vector stores, one per PDF hash
│
└── eval/                   ← THE PROOF: evaluation harness (see eval/README.md)
      ├── config.py                 corpus, configurations, shared metrics
      ├── generate_qa.py            question set from sampled pages
      ├── generate_qa_hard.py       paraphrased set, overlap-filtered
      ├── generate_probes.py        verified unanswerable questions
      ├── build_index.py            vector stores per chunking scheme
      ├── run_eval.py               the scorer
      ├── annotate_pages.py         multi-page ground truth
      ├── verify_*.py               six scripts that attack the harness's own assumptions
      └── qa_set*.json              the question sets
```

---

## Evaluation — does it actually work?

**[Full write-up: `eval/README.md`](eval/README.md)**

The honest answer to "does your RAG app work?" is usually "it seemed fine when I tried it."
That is not a measurement, so this project has one.

### Four metrics, deliberately kept separate

| Metric | Question it answers |
|---|---|
| **Recall@k** | Did the retriever surface the page the answer actually lives on? |
| **Answer accuracy** | Given that context, did the LLM write the right answer? |
| **Hallucination rate** | On questions the document *cannot* answer, did it invent one? |
| **Truncation rate** | Did generation hit the token ceiling and get cut off? |

Separating recall from accuracy is the whole point. If accuracy is poor, you need to know
whether the retriever handed over the wrong pages or the model fumbled the right ones —
those have **opposite fixes**.

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

## What the evaluation found

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
`eval/README.md` rather than quietly deleted.

What survives is narrower and true: one question in one run *did* return an empty string with
`finish_reason="length"`. Roughly 1 in 20. The defensible fix is not raising `max_tokens` —
no ceiling tested prevented it — but treating empty content as a failure and retrying once.

### Current results

| Config | Question set | Coverage | Recall@4 | Accuracy | Hallucination |
|---|---|---|---|---|---|
| `baseline` | easy | 17/48 | 100% | 94.4% | 0% |
| `baseline@hard` | hard | 10/42 | 100% | 100% | 0% |
| *BM25 control* | hard | 33/33 | *57.6%* | — | — |

**These are partial runs and are labelled as such.** On the 10 hard questions scored so far,
dense retrieval went 10/10 where keyword search manages 57.6% on the same set — the first real
evidence the embeddings earn their API call. The full comparison across six configurations is
pending; it is gated on Google's daily free quota, not on anything unfinished in the code.

Reproduce with:

```bash
py -3.12 eval/verify_method.py        # assumption checks     (no API)
py -3.12 eval/verify_retrieval.py --hard   # BM25 control     (no API)
py -3.12 eval/run_eval.py --hard      # full scoring run
```

---

## Known limitations

Stated plainly, because a project that lists none has not been looked at hard enough.

| Limitation | Detail |
|---|---|
| **Scanned PDFs** | PyPDFLoader extracts no text from image-only PDFs. There is no OCR fallback. |
| **Tables** | Chunking flattens tabular layout; table-heavy documents retrieve poorly. |
| **No reranking** | Top-4 by cosine similarity, with no cross-encoder rerank stage. |
| **Dense-only retrieval** | A hybrid BM25+dense retriever is implemented and unit-tested in `eval/`, but the live app still uses dense only. |
| **Occasional blank answer** | ~1 in 20; the app renders it as an empty reply instead of retrying. |
| **Free-tier quotas** | Google embeddings have a per-day cap and a per-minute rate limit. Large PDFs can exhaust the day's allowance. |
| **Single-trial evaluation** | Run-to-run variance at `temperature=0.1` is real; small differences between configurations are not meaningful without repeated trials. |

---

## What I would build next

In the order I would actually do them:

1. **Retry on empty generation.** Smallest change, removes a user-visible failure.
2. **Ship hybrid retrieval.** Already written and tested in `eval/`; the comparison run will
   say whether it beats dense-only on this corpus.
3. **Finish the six-configuration sweep** and put the table here — chunk size, `k`, and
   dense vs hybrid, each with a number.
4. **Add a reranker.** Retrieve 20, rerank to 4 with a cross-encoder. The usual next
   accuracy win once recall is healthy.
5. **OCR fallback** for scanned PDFs, which is currently a silent failure.
6. **Repeated trials per question** so differences between configurations can be
   distinguished from noise.

---

## Credits

Built with LangChain, ChromaDB, Streamlit, Groq and Google AI Studio — all on free tiers.

The evaluation harness, its findings, and the defects it uncovered are documented in full at
**[`eval/README.md`](eval/README.md)**.
