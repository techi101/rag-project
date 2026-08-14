# 🧠 DocuMind AI — Chat with Any PDF Document

> **A completely FREE full-stack Retrieval-Augmented Generation (RAG) application that lets you have an intelligent conversation with any PDF document. Powered by Groq (GPT OSS 20B) and Google Embeddings.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?style=flat-square)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=flat-square)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-GPT_OSS_20B-orange?style=flat-square)](https://groq.com)
[![Google](https://img.shields.io/badge/Google-Embeddings-purple?style=flat-square)](https://aistudio.google.com)

---

## 📸 What It Looks Like

A dark-mode, chat-style web application where you:
1. Upload any PDF (financial report, research paper, legal contract, textbook).
2. Ask questions about it in plain English.
3. Get accurate, cited answers instantly — generated at lightning speed by Groq.

---

## 🚀 Quick Start (Run in 3 Steps)

### Step 1: Install all required libraries
Open your terminal in this project folder and run:
```bash
pip install -r requirements.txt
```
This installs Streamlit, LangChain, ChromaDB, and the Groq/Google integration libraries.

### Step 2: Get your Free API Keys
This app uses a state-of-the-art hybrid AI stack designed to be **100% free** with no credit card required.

1. **Google API Key** (Used for converting text into vectors):
   - Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and create a key (starts with `AIza...`)
2. **Groq API Key** (Used for the GPT OSS 20B AI brain):
   - Go to [console.groq.com/keys](https://console.groq.com/keys) and create a key (starts with `gsk_...`)

Create a `.env` file in the root folder and add them like this (or just paste them directly into the app's sidebar):
```env
GOOGLE_API_KEY=AIzaYourKeyHere
GROQ_API_KEY=gsk_YourKeyHere
```

### Step 3: Launch the app
```bash
python -m streamlit run app.py
```
This opens a new tab in your browser at `http://localhost:8501`.

---

## 📁 Project Structure Explained

```
rag-document-intelligence/
│
├── app.py              ← THE FACE: The entire Streamlit web UI lives here.
│                          It handles user interactions, displays the chat,
│                          and calls functions from rag_engine.py.
│
├── rag_engine.py       ← THE BRAIN: All the RAG logic lives here.
│                          It processes PDFs, creates embeddings, stores them
│                          in ChromaDB, and runs the Q&A chain.
│
├── requirements.txt    ← THE SHOPPING LIST: All Python libraries this project
│                          needs to run. 
│
├── .env                ← YOUR SECRETS: Where your Google and Groq API keys 
│                          are safely stored locally (ignored by git).
│
├── .gitignore          ← THE FILTER: Tells Git which files NOT to upload to
│                          GitHub to keep your API keys safe.
│
└── chroma_store/       ← THE DATABASE (auto-created): This folder is created
    └── [pdf_hash]/        automatically when you process your first PDF.
        └── ...            Each PDF gets its own unique subfolder.
```

---

## 🧠 How RAG Works — The Architecture

This section explains the entire technology stack from scratch. 

### The Problem RAG Solves
If you go to a public AI and ask: *"What were the key findings in my company's Q3 2025 internal audit report?"*, it will fail because it hasn't seen your private PDF documents. RAG (Retrieval-Augmented Generation) is the architecture that solves this problem.

### The Pipeline (6 Steps)

#### Step 1: LOAD — Read the PDF 📄
```python
loader = PyPDFLoader(pdf_path)
pages = loader.load()
```
The `PyPDFLoader` library opens the PDF file and extracts all the text from every page.

#### Step 2: CHUNK — Break it into Pieces 🧩
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
```
We break the massive PDF into small, 1000-character paragraphs. We do this because AI models have memory limits (context windows). 

#### Step 3: EMBED — Convert Text to Vectors 🔢
```python
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```
We pass every chunk of text through Google's powerful embedding model. It converts the paragraph into a mathematical array (vector) of numbers that represent the core "meaning" of the text.

#### Step 4: STORE — Save to ChromaDB 💾
```python
vector_store = Chroma.from_documents(chunks, embedding_model, persist_dir)
```
We save these vectors locally in a vector database called ChromaDB so we don't have to re-process the PDF ever again.

#### Step 5: RETRIEVE — Find the Answer 🔍
When the user asks a question, we convert their question into a vector using the exact same Google model. We then ask ChromaDB to find the top 4 chunks of text whose vectors are mathematically closest to the question's vector.

#### Step 6: GENERATE — Answer the Question 🤖
```python
llm = ChatGroq(model="openai/gpt-oss-20b")
```
We take those 4 relevant paragraphs, bundle them together with the user's question, and send it to Groq (running GPT OSS 20B). We strictly instruct the model to *only* answer using the provided paragraphs, eliminating hallucinations. Groq's specialized hardware generates the answer almost instantly!
