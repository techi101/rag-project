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
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Evaluation — does it actually work?](#evaluation--does-it-actually-work)
- [Known limitations](#known-limitations)
- [What I would build next](#what-i-would-build-next)
- [Further reading](#further-reading)

---

## What this project is

Two things, and the second is the more interesting one.

**1. A working RAG application.** Upload a PDF, it gets chunked, embedded and stored in a
local vector database. Ask a question, the most relevant passages are retrieved and handed
to an LLM that answers *only* from those passages, citing page numbers. Every processed PDF
is cached by content hash, so re-uploading the same file is instant and costs nothing
(the cache is rebuilt if the chunk settings or embedding model change, or if the stored index is incomplete).

**2. An evaluation harness that measures whether it works.** Most RAG demos stop at "it
seemed fine when I tried it." This one ships with `eval/` — a suite that scores retrieval
recall, answer accuracy, hallucination rate and truncation rate on a purpose-built question
set, with a dumb keyword-search control to check the clever parts are earning their keep.

That second part is where the engineering is. It is documented in detail in
**[`eval/README.md`](eval/README.md)**, and summarised [below](#evaluation--does-it-actually-work).

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
├── app.py                  ← THE FACE: the Streamlit UI — upload, chat, sidebar, session state
│                              (Streamlit Cloud runs this file, so it stays at the top level)
│
├── documind/               ← THE BRAIN: all RAG logic, as a Python package
│     ├── settings.py               every setting in ONE place: models, chunk size, top-k, system prompt
│     ├── ingest.py                 load → chunk → embed → store; reuse a saved index (get_pdf_hash,
│     │                             process_pdf, load_existing_vectorstore)
│     ├── chain.py                  retriever + LLM + grounded prompt; one Q&A turn (create_qa_chain,
│     │                             run_qa, format_docs, page_label)
│     └── summary.py                3-bullet document summary on upload (generate_summary)
│
├── assets/
│     └── style.css         ← the app's CSS (dark theme, mobile sizes)
│
├── docs/                   ← the long-form write-ups
│     ├── how-it-works.md           why RAG, architecture, the 6 pipeline steps, tech choices
│     └── evaluation-findings.md    what the evaluation found, defect by defect
│
├── eval/                   ← THE PROOF: evaluation harness (see eval/README.md)
│     ├── config.py                 corpus, configurations, shared metrics (models come from documind/settings.py)
│     ├── build_index.py            vector stores per chunking scheme
│     ├── run_eval.py               the scorer
│     ├── generate/                 build the test sets
│     │     ├── generate_qa.py            question set from sampled pages
│     │     ├── generate_qa_hard.py       paraphrased set, overlap-filtered
│     │     ├── generate_probes.py        verified unanswerable questions
│     │     └── annotate_pages.py         multi-page ground truth
│     ├── verify/                   six scripts that attack the harness's own assumptions
│     ├── data/                     qa_set.json, qa_set_hard.json, documents.example.json
│     └── results/                  summary.json
│
├── .streamlit/config.toml  ← Streamlit theme
├── requirements.txt        ← dependencies
├── .env.example            ← copy to .env and add your API keys (.env is gitignored)
└── chroma_store/           ← auto-created vector stores, one per PDF hash (gitignored)
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

### The headline finding

**BM25 — plain keyword matching, no embeddings — scored 97.7% recall@4 on the first question
set**, the same as the dense retriever: the questions copied their page's wording, so they could
not tell the two apart. A paraphrased question set dropped BM25 to **57.6%**. The full story, and
the four defects found in the harness itself, is in
**[`docs/evaluation-findings.md`](docs/evaluation-findings.md)**.

### Current results

| Config | Question set | Coverage | Recall@4 | Accuracy | Hallucination |
|---|---|---|---|---|---|
| `baseline` | easy | 17/48 | 100% | 100% (15/15) | 0% (2 probes) |
| `baseline@hard` | hard | 10/42 | 100% | 100% (10/10) | — (no probes scored) |
| *BM25 control* | hard | 33/33 | *57.6%* | — | — |
| *BM25 control, same 10 questions* | hard | 10/10 | *40.0%* | — | — |

**These are partial runs and are labelled as such.** Numbers come from
`eval/results/summary.json`. Every question scored so far is from the MySQL handbook — the
Google daily quota ran out before the other two documents were reached — so these say
nothing yet about the dense prose or slide-deck documents.

On the 10 hard questions scored so far, dense retrieval found the right page 10 times out of
10; BM25 on **the same 10 questions** found it 4 times. That is the first like-for-like evidence
that the embeddings earn their API call, on a small sample from one document. The full
comparison across six configurations is pending; it is gated on Google's daily free quota,
not on anything unfinished in the code.

The question sets were generated and filtered automatically (overlap filter, duplicate and
leak checks in `verify_qa.py`, probe verification). They have **not** been hand-reviewed:
`reviewed` is `false` on all 90 questions.

Reproduce with:

```bash
py -3.12 eval/verify/verify_method.py             # assumption checks   (no API)
py -3.12 eval/verify/verify_retrieval.py --hard   # BM25 control        (no API)
py -3.12 eval/run_eval.py --hard                  # full scoring run
```

---

## Known limitations

Stated plainly, because a project that lists none has not been looked at hard enough.

| Limitation | Detail |
|---|---|
| **Scanned PDFs** | PyPDFLoader extracts no text from image-only PDFs. There is no OCR fallback. |
| **Tables** | Chunking flattens tabular layout; table-heavy documents retrieve poorly. |
| **No reranking** | Top-4 nearest chunks, with no cross-encoder rerank stage. |
| **Dense-only retrieval** | A hybrid BM25+dense retriever is implemented and unit-tested in `eval/`, but the live app still uses dense only. |
| **Occasional blank answer** | Seen once in one eval run. The app now retries once and shows a message if it is still empty. |
| **Free-tier quotas** | Google embeddings have a per-day cap and a per-minute rate limit. Large PDFs can exhaust the day's allowance. |
| **Follow-up questions** | The model sees the chat history, but retrieval uses only the latest question, so vague follow-ups ("and the second one?") retrieve poorly. Query rewriting would fix this. |
| **Prompt injection** | Retrieved text is placed inside the system message; a PDF containing instructions could try to override the rules. Untested. |
| **Shared host keys** | On a deployment with host keys set, visitors who leave the key boxes blank use the host's quota. |
| **Single-trial evaluation** | Run-to-run variance at `temperature=0.1` is real; small differences between configurations are not meaningful without repeated trials. |

---

## What I would build next

In the order I would actually do them:

1. ~~**Retry on empty generation.**~~ Done: `run_qa` retries once on an empty answer.
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

## Further reading

| Document | What it covers |
|---|---|
| [`docs/how-it-works.md`](docs/how-it-works.md) | Why RAG, the architecture diagram, each pipeline step with code, why each tool was chosen |
| [`docs/evaluation-findings.md`](docs/evaluation-findings.md) | How the test set is built and everything the evaluation found |
| [`eval/README.md`](eval/README.md) | The evaluation harness in full: how to run it, every script, every number |

---

## Credits

Built with LangChain, ChromaDB, Streamlit, Groq and Google AI Studio — all on free tiers.

The evaluation harness, its findings, and the defects it uncovered are documented in full at
**[`eval/README.md`](eval/README.md)**.
