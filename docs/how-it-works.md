# How DocuMind AI works

> Back to the [README](../README.md). The code for every step below is in [`documind/`](../documind/).

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

Which is exactly why this project has an [evaluation harness](../eval/README.md) that scores retrieval
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
                         │  similarity search (k=4) │   nearest by distance
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
> [the evaluation](evaluation-findings.md#2--chunk-size-does-nothing-on-sparse-documents).

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

The question is embedded with the same model, and ChromaDB returns the `k=4` nearest chunks.
Chroma's default distance is squared L2; gemini-embedding-001 returns unit-length vectors,
and for unit vectors squared L2 = 2 − 2·cosine, so the ranking is exactly the cosine ranking
(checked on a stored index: distances match 2 − 2·cos to four decimals).

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
obeys it — see [hallucination probes](evaluation-findings.md#3--hallucination-is-measured-not-hoped-for).

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
