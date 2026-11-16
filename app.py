"""
app.py — DocuMind AI
RAG document Q&A: Google gemini-embedding-001 + ChromaDB for retrieval,
Groq openai/gpt-oss-20b for answers.
"""

# WHAT THIS FILE IS: the web page (sidebar, welcome page, chat page); all RAG work is in documind/.
# Example: paste Google key ("AIza...") + Groq key ("gsk_..."), upload the MySQL Handbook, click
#   "Analyze Document", ask "What is the main topic of this document?" -> answer + "4 Source Citations".
# Flow: sidebar (keys + upload) -> Analyze -> stats + summary + vector store + QA chain in session_state
#   -> chat page -> question -> run_qa() -> answer + citations
# Streamlit = Python script -> web page; it RE-RUNS this whole file on every click, so memory lives in session_state.
import os  # read API keys from env, delete temp PDF
import html  # html.escape: "<" -> "&lt;", PDF text can't act as HTML
import tempfile  # temp file: PyPDFLoader needs a real path
from pathlib import Path  # finds assets/style.css next to this file
import streamlit as st  # the web UI library
from pypdf import PdfReader  # only to count pages + words for stats
from dotenv import load_dotenv  # reads .env into environment variables

load_dotenv()  # now os.getenv("GROQ_API_KEY") sees .env values

from documind import (  # our RAG "brain" package
    process_pdf,  # build a new vector store
    load_existing_vectorstore,  # reuse a saved one
    create_qa_chain,  # retriever + LLM + prompt
    run_qa,  # answer one question
    generate_summary,  # 3-bullet summary
    page_label,  # metadata page 0 -> "1"
)
from documind.settings import GEN_MODEL  # model name shown on welcome page, never stale

# ── BLOCK: page config ──────────────────────────────────────────────────
# Must be the first Streamlit command.
st.set_page_config(  # browser tab + layout
    page_title="DocuMind AI — Document Intelligence",  # tab title
    page_icon="🧠",  # tab icon
    layout="wide",  # full page width
    initial_sidebar_state="expanded",  # sidebar open at start
)

# ── BLOCK: global CSS (assets/style.css) ────────────────────────────────
# CSS = page styling: Inter font, dark #0D1117 theme, .hero-title/.feat-card/... classes, phone sizes (768px/480px).
_css = (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")  # works from any run folder
st.markdown("<style>" + _css + "</style>", unsafe_allow_html=True)  # unsafe_allow_html = let raw HTML/CSS render


# ── BLOCK: session state defaults ───────────────────────────────────────
# session_state = app memory between reruns, per browser tab. Set defaults only once, or reruns wipe the chat.
for k, v in {  # each (key, default)
    "messages": [], "qa_chain": None, "chat_history_pairs": [],  # chat, chain (None = welcome), LLM memory
    "pdf_name": None, "doc_summary": None, "doc_stats": {},  # file name, summary text, {"pages", "read_time", "words"}
    "processing": False, "pending_question": None,  # True while indexing; starter question waiting
}.items():
    if k not in st.session_state:  # first run in this tab?
        st.session_state[k] = v  # set the default


# ── BLOCK: sidebar ──────────────────────────────────────────────────────
with st.sidebar:  # everything here draws in the left sidebar
    # ── Step 1: logo ──
    st.markdown("""
    <div style="padding:4px 0 8px">
      <span style="font-size:22px;font-weight:800;background:linear-gradient(135deg,#58A6FF,#BC8CFF);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">🧠 DocuMind AI</span><br>
      <span style="font-size:12px;color:#8B949E;">RAG · LangChain · Groq</span>
    </div>
    """, unsafe_allow_html=True)  # gradient title + tagline as raw HTML
    st.divider()  # thin horizontal line

    # ── Step 2: API keys (typed key first, else .env / host secret) ──
    st.markdown("**🔑 Google API Key** (for embeddings)")  # bold label
    user_google_key = st.text_input(  # password box for Google key
        "google_api_key", label_visibility="collapsed",  # internal label, hidden
        value="",  # starts empty
        type="password",  # shows dots
        placeholder="Enter your Google API Key (or leave blank to use host key)"  # grey hint text
    )
    google_api_key = user_google_key if user_google_key else os.getenv("GOOGLE_API_KEY", "")  # typed, else env, else ""
    st.caption("Free key → [aistudio.google.com](https://aistudio.google.com/app/apikey)")  # small grey link

    st.markdown("**🔑 Groq API Key** (for AI chat)")  # bold label
    user_groq_key = st.text_input(  # password box for Groq key
        "groq_api_key", label_visibility="collapsed",  # internal label, hidden
        value="",  # starts empty
        type="password",  # shows dots
        placeholder="Enter your Groq API Key (or leave blank to use host key)"  # grey hint text
    )
    groq_api_key = user_groq_key if user_groq_key else os.getenv("GROQ_API_KEY", "")  # typed, else env, else ""
    st.caption("Free key → [console.groq.com](https://console.groq.com/keys)")  # small grey link

    if not google_api_key or not groq_api_key:  # a key missing?
        st.warning("⚠️ Both API keys required above.")  # yellow warning, page still draws

    st.divider()  # line

    # ── Step 3: PDF upload ──
    st.markdown("**📂 Upload Document**")  # label
    uploaded_file = st.file_uploader(  # None until a file is picked
        "pdf_upload", label_visibility="collapsed", type=["pdf"]  # .pdf only
    )

    if uploaded_file and (not google_api_key or not groq_api_key):  # file but key missing?
        st.warning("⚠️ Enter both API keys above.")  # Analyze button stays hidden

    # ── Step 4: Analyze button (PDF + both keys present) ──
    if uploaded_file and google_api_key and groq_api_key:  # ready to analyze?
        if st.button("🚀 Analyze Document", use_container_width=True, type="primary"):  # True only on the click
            st.session_state.update({  # start fresh for the new document
                "messages": [], "chat_history_pairs": [],  # clear old chat
                "qa_chain": None, "doc_summary": None, "processing": True,  # clear old chain + summary
            })
            with st.spinner("🔍 Embedding & indexing your document..."):  # loading wheel while this runs
                tmp_path = None  # no temp file yet (used in "finally")
                try:  # bad key / quota / scanned PDF -> "except"
                    # ── Step 4a: save the upload as a real .pdf file ──
                    # delete=False: we delete it in "finally"; Windows can't reread a still-open file.
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:  # temp file
                        tmp.write(uploaded_file.read())  # write the uploaded bytes
                        tmp_path = tmp.name  # remember where it is

                    # ── Step 4b: stats (pages, words, read time) ──
                    reader = PdfReader(tmp_path)  # open with pypdf
                    npages = len(reader.pages)  # e.g. 72 pages
                    # Real word count, not a per-page guess; read time at ~200 words/min.
                    nwords = sum(len((p.extract_text() or "").split())  # words on one page ("" if None)
                                 for p in reader.pages)  # summed over all pages
                    st.session_state.doc_stats = {  # shown in sidebar + chat header
                        "pages": npages,  # page count
                        "read_time": f"~{max(1, round(nwords / 200))} min",  # 200 words/min, at least 1 min
                        "words": f"{nwords:,}",  # 12345 -> "12,345"
                    }
                    st.session_state.doc_summary = generate_summary(tmp_path, groq_api_key)  # Groq call on every Analyze

                    # ── Step 4c: reuse saved index or build a new one ──
                    vector_store = load_existing_vectorstore(tmp_path, google_api_key)  # same hash -> no Google calls
                    if not vector_store:  # None = new PDF, changed settings, or incomplete
                        with st.spinner("🧠 Building knowledge base..."):  # second spinner for slow step
                            vector_store = process_pdf(tmp_path, google_api_key)  # chunk + embed + save

                    # ── Step 4d: build the chain once, switch to chat page ──
                    st.session_state.qa_chain = create_qa_chain(vector_store, groq_api_key)  # not None -> chat page
                    st.session_state.pdf_name = uploaded_file.name  # for the chat header
                    st.session_state.processing = False  # done indexing
                    st.success("✅ Document ready!")  # green box
                    st.rerun()  # restart script now, chat page shows

                except Exception as e:  # e.g. RESOURCE_EXHAUSTED quota, "No text could be extracted"
                    st.error(f"❌ {e}")  # red error box
                    st.session_state.processing = False  # re-enable chat box
                finally:  # runs on success or failure
                    # Don't leave a copy of every upload in the temp folder.
                    if tmp_path and os.path.exists(tmp_path):  # temp file exists?
                        try:  # deleting may fail (Windows lock)
                            os.remove(tmp_path)  # delete it
                        except OSError:  # leftover file must not crash the app
                            pass  # ignore

    # ── Step 5: stats + actions (only after a document is processed) ──
    if st.session_state.qa_chain and st.session_state.doc_stats:  # document loaded?
        st.divider()  # line
        st.markdown("**📊 Document Stats**")  # label
        c1, c2 = st.columns(2)  # two side-by-side boxes
        c1.metric("Pages", st.session_state.doc_stats.get("pages", 0))  # metric = big number + label
        c2.metric("Read Time", st.session_state.doc_stats.get("read_time", "—"))  # e.g. "~95 min"
        st.caption(f"{st.session_state.doc_stats.get('words','—')} words")  # small word count, "—" if missing

        st.divider()  # line
        st.markdown("**⚙️ Actions**")  # label
        if st.session_state.messages:  # export only if chat has messages
            log = f"DocuMind AI — Chat Log\nDocument: {st.session_state.pdf_name}\n{'='*48}\n\n"  # title + 48 "=" line
            for m in st.session_state.messages:  # each chat message
                log += f"[{'You' if m['role']=='user' else 'AI'}]\n{m['content']}\n\n"  # "[You]" or "[AI]" + text
            st.download_button("💾 Export Chat", data=log,  # button that downloads the log
                               file_name="documind_chat.txt", mime="text/plain",  # as a plain text file
                               use_container_width=True)  # full width
        if st.button("🗑️ Clear Chat", use_container_width=True):  # clears chat, PDF stays loaded
            st.session_state.messages = []  # visible chat
            st.session_state.chat_history_pairs = []  # LLM memory
            st.rerun()  # redraw now


# ── BLOCK: main area ────────────────────────────────────────────────────
# No QA chain yet -> welcome page; else -> chat page.
if not st.session_state.qa_chain:  # no document yet?
    # ── Step 1: hero (title, subtitle, badges) ──
    # Center column trick — narrow padding columns either side
    _, hero_col, _ = st.columns([1, 8, 1])  # widths 1:8:1, use only middle ("_" = unused)
    with hero_col:  # draw inside middle column
        model_name = GEN_MODEL.split("/")[-1]  # "openai/gpt-oss-20b" -> "gpt-oss-20b"
        st.markdown(f"""
        <div style="text-align:center; padding: 48px 0 32px;">
          <div style="font-size:72px; margin-bottom:16px; filter:drop-shadow(0 0 28px rgba(88,166,255,0.55));
               animation:none;">🧠</div>
          <div class="hero-title">DocuMind AI</div>
          <p class="hero-sub">Upload any PDF and have an intelligent conversation with it.<br>
          Powered by semantic vector search and Groq-hosted {model_name}.</p>
          <div style="text-align:center; margin-bottom:40px;">
            <span class="badge-pill"><b>⚡</b> Groq {model_name}</span>
            <span class="badge-pill"><b>🗄️</b> ChromaDB</span>
            <span class="badge-pill"><b>🔗</b> LangChain</span>
            <span class="badge-pill"><b>🆓</b> Free-tier APIs</span>
          </div>
        </div>
        """, unsafe_allow_html=True)  # HTML using style.css classes

    # ── Step 2: 6 feature cards (2 rows of 3) ──
    _, cards_col, _ = st.columns([0.5, 9, 0.5])  # same centring, 0.5:9:0.5
    with cards_col:  # inside middle column
        r1c1, r1c2, r1c3 = st.columns(3, gap="medium")  # row 1: 3 equal columns

        with r1c1:  # card 1
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">📄</span>
              <div class="feat-title">Any PDF Document</div>
              <div class="feat-desc">Research papers, contracts, financial reports, textbooks — any text-based PDF (scanned PDFs need OCR, not yet supported).</div>
            </div>""", unsafe_allow_html=True)  # .feat-card HTML

        with r1c2:  # card 2
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">🔍</span>
              <div class="feat-title">Semantic Search</div>
              <div class="feat-desc">Finds the most relevant paragraphs by meaning — not just keyword matching. Powered by Google's embedding model.</div>
            </div>""", unsafe_allow_html=True)  # .feat-card HTML

        with r1c3:  # card 3
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">🎯</span>
              <div class="feat-title">Grounded Answers</div>
              <div class="feat-desc">The model is instructed to answer only from your document and to say so when the answer isn't there.</div>
            </div>""", unsafe_allow_html=True)  # .feat-card HTML

        st.write("")  # empty line = space between rows

        r2c1, r2c2, r2c3 = st.columns(3, gap="medium")  # row 2: 3 columns

        with r2c1:  # card 4
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">✨</span>
              <div class="feat-title">Auto Summarization</div>
              <div class="feat-desc">Instantly generates a 3-point executive summary the moment your document is processed.</div>
            </div>""", unsafe_allow_html=True)  # .feat-card HTML

        with r2c2:  # card 5
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">💬</span>
              <div class="feat-title">Conversation Memory</div>
              <div class="feat-desc">Ask follow-up questions. The AI sees your earlier questions and answers in this session.</div>
            </div>""", unsafe_allow_html=True)  # .feat-card HTML

        with r2c3:  # card 6
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">📎</span>
              <div class="feat-title">Source Citations</div>
              <div class="feat-desc">Every answer lists the retrieved passages and their page numbers, so you can check it.</div>
            </div>""", unsafe_allow_html=True)  # .feat-card HTML

    # ── Step 3: "How it works", 4 numbered steps ──
    _, steps_col, _ = st.columns([0.5, 9, 0.5])  # centred column
    with steps_col:  # inside it
        st.markdown('<div class="section-label">⚙️ How it works</div>', unsafe_allow_html=True)  # section label
        s1, s2, s3, s4 = st.columns(4, gap="small")  # 4 columns

        for col, num, title, desc in [  # (column, number, title, text) per step
            (s1, "1", "Upload PDF", "Choose any PDF document from your device."),  # step 1
            (s2, "2", "Chunk & Embed", "Text is split into paragraphs and converted into math vectors."),  # step 2
            (s3, "3", "Stored in ChromaDB", "Vectors saved locally on disk for instant retrieval."),  # step 3
            (s4, "4", "Ask Anything", "The LLM reads the best-matching chunks and generates a cited answer."),  # step 4
        ]:
            with col:  # draw in this step's column
                st.markdown(f"""
                <div class="step-wrap">
                  <div class="step-num">{num}</div>
                  <div class="step-title">{title}</div>
                  <div class="step-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)  # {num}/{title}/{desc} filled from the loop

    st.markdown("<br>", unsafe_allow_html=True)  # bottom spacing


else:  # document loaded -> chat page
    # ── Step 1: header bar (name, pages, read time, words) ──
    # html.escape: a file named "<b>x</b>.pdf" shows as text; .get(...,'?') = "?" if a stat is missing.
    st.markdown(f"""
    <div class="chat-doc-bar">
      <span style="font-size:28px;">📄</span>
      <div>
        <div class="chat-doc-name">{html.escape(st.session_state.pdf_name or "")}</div>
        <div class="chat-doc-meta">
          {st.session_state.doc_stats.get('pages','?')} pages &nbsp;·&nbsp;
          {st.session_state.doc_stats.get('read_time','?')} read &nbsp;·&nbsp;
          {st.session_state.doc_stats.get('words','?')} words &nbsp;·&nbsp;
          <span style="color:#3FB950;">● Vector index ready</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)  # green "Vector index ready" dot

    # ── Step 2: auto summary box ──
    if st.session_state.doc_summary:  # summary exists?
        summary_html = html.escape(st.session_state.doc_summary).replace("\n", "<br>")  # escape LLM text, keep 3 lines
        st.markdown(f"""
        <div class="summary-box">
          <div class="summary-label">✨ AI Executive Summary</div>
          <div class="summary-body">{summary_html}</div>
        </div>
        """, unsafe_allow_html=True)  # summary card

    # ── Step 3: redraw previous messages (page starts blank each rerun) ──
    for message in st.session_state.messages:  # whole chat from memory
        avatar = "👤" if message["role"] == "user" else "🧠"  # person vs brain icon
        with st.chat_message(message["role"], avatar=avatar):  # one chat bubble
            st.markdown(message["content"])  # markdown keeps bullets + bold
            if message.get("sources"):  # AI messages only (user has none)
                # collapsed section, e.g. "4 Source Citations"
                with st.expander(f"📎 {len(message['sources'])} Source Citations", expanded=False):
                    for doc in message["sources"]:  # each chunk
                        pg = page_label(doc.metadata)  # e.g. "5"
                        # 380-char preview (chunk max 1000), newlines -> spaces
                        snippet = doc.page_content[:380].strip().replace("\n", " ")
                        st.markdown(f"**📄 Page {pg}**")  # bold page label
                        st.caption(f"{snippet}...")  # grey preview
                        st.divider()  # line between chunks

    # ── Step 4: starter questions (only when chat empty) ──
    if not st.session_state.messages:  # empty chat?
        st.markdown('<p class="suggestion-hint">💡 Try asking</p>', unsafe_allow_html=True)  # hint text
        sq1, sq2, sq3 = st.columns(3, gap="small")  # 3 columns
        qs = [  # general, so they fit any PDF
            "What is the main topic of this document?",  # question 1
            "What are the key findings or conclusions?",  # question 2
            "Are there any risks or challenges mentioned?",  # question 3
        ]
        for col, q in zip([sq1, sq2, sq3], qs):  # pair each column with a question
            if col.button(q, use_container_width=True):  # clicked?
                st.session_state.pending_question = q  # store it for next rerun
                st.rerun()  # picked up below as if typed

    # ── Step 5: get the question (typed or starter button) ──
    # text once after Enter, else None; disabled while indexing
    user_q = st.chat_input("Ask anything about your document...", disabled=st.session_state.processing)

    if st.session_state.pending_question:  # button clicked last run?
        user_q = st.session_state.pending_question  # use it
        st.session_state.pending_question = None  # clear so it isn't asked again

    # ── Step 6: answer it ──
    if user_q:  # have a question?
        st.session_state.messages.append({"role": "user", "content": user_q})  # save user message first

        with st.chat_message("user", avatar="👤"):  # draw user bubble now
            st.markdown(user_q)  # the question text

        with st.chat_message("assistant", avatar="🧠"):  # AI bubble
            with st.spinner("Searching document and generating answer..."):  # spinner while waiting
                try:  # Google/Groq errors -> "except"
                    result = run_qa(  # retrieve top 4 + ask Groq
                        st.session_state.qa_chain,  # retriever + LLM + prompt
                        user_q,  # the question
                        st.session_state.chat_history_pairs,  # past pairs, for follow-ups
                    )
                    answer = result.get("answer", "Could not generate an answer.")  # fallback if missing
                    sources = result.get("source_documents", [])  # the 4 chunks

                    st.session_state.chat_history_pairs.append((user_q, answer))  # memory for next question
                    st.markdown(answer)  # show the answer

                    if sources:  # show source chunks, same format as above
                        with st.expander(f"📎 {len(sources)} Source Citations", expanded=False):  # collapsed section
                            for doc in sources:  # each chunk
                                pg = page_label(doc.metadata)  # e.g. "5"
                                snippet = doc.page_content[:380].strip().replace("\n", " ")  # 380-char preview
                                st.markdown(f"**📄 Page {pg}**")  # page label
                                st.caption(f"{snippet}...")  # preview
                                st.divider()  # line

                    st.session_state.messages.append({  # save so reruns redraw it (Step 3)
                        "role": "assistant",  # AI message
                        "content": answer,  # its text
                        "sources": sources,  # its chunks
                    })

                except Exception as e:  # show error, keep it visible in chat
                    err = f"⚠️ Error: {str(e)}"  # error text
                    st.error(err)  # red box
                    # Saved to messages but NOT chat_history_pairs, so errors never reach the LLM.
                    st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
