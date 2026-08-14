"""
app.py — DocuMind AI
Premium RAG Document Intelligence powered by Gemini 1.5 Flash + ChromaDB.
"""

import os
import tempfile
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from rag_engine import (
    process_pdf,
    load_existing_vectorstore,
    create_qa_chain,
    run_qa,
    generate_summary,
)

# ── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI — Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [class*="css"], .stMarkdown p {
  font-family: 'Inter', sans-serif !important;
}
.stApp { background: #0D1117 !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px; }

/* ── Global Text Colors (fixes dark-on-dark on mobile/cloud) ─────────────── */
.stApp, .stApp * {
  color: #C9D1D9;
}
.stMarkdown p, .stMarkdown li, .stMarkdown span,
.stMarkdown strong, .stMarkdown b {
  color: #E6EDF3 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
  color: #F0F6FC !important;
}
.stMarkdown a { color: #58A6FF !important; }
.stMarkdown code { color: #F0883E !important; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
  color: #E6EDF3 !important;
}
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] p {
  color: #C9D1D9 !important;
}
.stCaption, .stCaption p { color: #8B949E !important; }
.stAlert p { color: #E6EDF3 !important; }
input, textarea { color: #E6EDF3 !important; }
[data-testid="stChatInput"] textarea { color: #E6EDF3 !important; }
label { color: #C9D1D9 !important; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: #161B22 !important;
  border-right: 1px solid #21262D !important;
}
section[data-testid="stSidebar"] * {
  color: #C9D1D9 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown strong,
section[data-testid="stSidebar"] label {
  color: #E6EDF3 !important;
}
section[data-testid="stSidebar"] .stCaption, 
section[data-testid="stSidebar"] .stCaption p {
  color: #8B949E !important;
}
section[data-testid="stSidebar"] a {
  color: #58A6FF !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, #1F6FEB, #388BFD) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 10px 20px !important;
  transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
  opacity: 0.9 !important;
  transform: translateY(-1px) !important;
}

/* ── Chat messages ─────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: #161B22 !important;
  border: 1px solid #21262D !important;
  border-radius: 12px !important;
  padding: 14px 18px !important;
  margin-bottom: 10px !important;
}

/* ── Metric boxes ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: #161B22;
  border: 1px solid #21262D;
  border-radius: 10px;
  padding: 12px 14px !important;
}
[data-testid="stMetricValue"] { color: #58A6FF !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #8B949E !important; }

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }

/* ── Hero section ──────────────────────────────────────────────────────── */
.hero-title {
  font-size: 56px !important;
  font-weight: 800 !important;
  letter-spacing: -2.5px;
  line-height: 1.1;
  text-align: center;
  background: linear-gradient(135deg, #58A6FF 0%, #BC8CFF 55%, #F78166 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 14px;
}
.hero-sub {
  font-size: 17px;
  color: #8B949E;
  text-align: center;
  line-height: 1.65;
  max-width: 560px;
  margin: 0 auto 28px;
}
.badge-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #1C2128;
  border: 1px solid #30363D;
  color: #8B949E;
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 12.5px;
  font-weight: 500;
  margin: 4px;
}
.badge-pill b { color: #58A6FF; }

/* ── Feature cards ──────────────────────────────────────────────────────── */
.feat-card {
  background: linear-gradient(145deg, #161B22 0%, #1C2128 100%);
  border: 1px solid #21262D;
  border-radius: 16px;
  padding: 26px 22px;
  height: 100%;
  transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
  cursor: default;
}
.feat-card:hover {
  border-color: #388BFD;
  box-shadow: 0 6px 28px rgba(56,139,253,0.18);
  transform: translateY(-3px);
}
.feat-icon { font-size: 30px; margin-bottom: 12px; display: block; }
.feat-title { font-size: 15px; font-weight: 700; color: #E6EDF3; margin-bottom: 8px; }
.feat-desc { font-size: 13.5px; color: #8B949E; line-height: 1.6; }

/* ── Steps ─────────────────────────────────────────────────────────────── */
.step-wrap {
  text-align: center;
  padding: 10px 8px;
}
.step-num {
  width: 38px; height: 38px;
  background: linear-gradient(135deg, #1F6FEB, #BC8CFF);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; font-weight: 700; color: #fff;
  margin: 0 auto 10px;
  box-shadow: 0 4px 14px rgba(56,139,253,0.35);
}
.step-title { font-size: 14px; font-weight: 700; color: #E6EDF3; margin-bottom: 4px; }
.step-desc { font-size: 12.5px; color: #8B949E; line-height: 1.55; }

/* ── Section divider label ──────────────────────────────────────────────── */
.section-label {
  font-size: 11.5px;
  font-weight: 700;
  color: #8B949E;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  text-align: center;
  margin: 32px 0 18px;
}

/* ── Chat page header ───────────────────────────────────────────────────── */
.chat-doc-bar {
  background: linear-gradient(135deg, #161B22, #1C2128);
  border: 1px solid #21262D;
  border-left: 4px solid #388BFD;
  border-radius: 12px;
  padding: 16px 22px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 14px;
}
.chat-doc-name { font-size: 16px; font-weight: 700; color: #E6EDF3; }
.chat-doc-meta { font-size: 13px; color: #8B949E; margin-top: 3px; }

/* ── Summary box ─────────────────────────────────────────────────────────── */
.summary-box {
  background: #161B22;
  border: 1px solid #21262D;
  border-top: 3px solid #58A6FF;
  border-radius: 12px;
  padding: 20px 24px;
  margin-bottom: 22px;
}
.summary-label {
  font-size: 11px; font-weight: 700;
  color: #58A6FF; text-transform: uppercase; letter-spacing: 1.5px;
  margin-bottom: 12px;
}
.summary-body { font-size: 14px; color: #C9D1D9; line-height: 1.75; }

/* ── Suggestion chips ────────────────────────────────────────────────────── */
.suggestion-hint {
  font-size: 12px; font-weight: 600; color: #8B949E;
  text-transform: uppercase; letter-spacing: 1px;
  margin-bottom: 8px;
}

/* ── Mobile Responsive ───────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .block-container {
    padding-top: 1rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
    max-width: 100% !important;
  }

  /* Hero */
  .hero-title {
    font-size: 32px !important;
    letter-spacing: -1.5px;
    margin-bottom: 10px;
  }
  .hero-sub {
    font-size: 14px;
    max-width: 100%;
    margin-bottom: 20px;
    padding: 0 8px;
  }
  .badge-pill {
    font-size: 11px;
    padding: 5px 10px;
    margin: 3px;
  }

  /* Feature cards */
  .feat-card {
    padding: 18px 16px;
    border-radius: 12px;
    margin-bottom: 8px;
  }
  .feat-icon { font-size: 24px; margin-bottom: 8px; }
  .feat-title { font-size: 14px; }
  .feat-desc { font-size: 12.5px; }

  /* Steps */
  .step-wrap { padding: 8px 4px; }
  .step-num { width: 32px; height: 32px; font-size: 14px; }
  .step-title { font-size: 12px; }
  .step-desc { font-size: 11px; }

  /* Chat page */
  .chat-doc-bar {
    padding: 12px 14px;
    gap: 10px;
    flex-wrap: wrap;
  }
  .chat-doc-name { font-size: 14px; }
  .chat-doc-meta { font-size: 11px; }

  [data-testid="stChatMessage"] {
    padding: 10px 12px !important;
    border-radius: 10px !important;
  }

  /* Summary */
  .summary-box {
    padding: 14px 16px;
  }
  .summary-body { font-size: 13px; }

  /* Buttons */
  .stButton > button {
    font-size: 12px !important;
    padding: 8px 14px !important;
  }
}

@media (max-width: 480px) {
  .hero-title {
    font-size: 26px !important;
    letter-spacing: -1px;
  }
  .hero-sub { font-size: 13px; }
  .badge-pill { font-size: 10px; padding: 4px 8px; }
  .feat-card { padding: 14px 12px; }
  .chat-doc-bar { padding: 10px 12px; }
  .chat-doc-name { font-size: 13px; }
}
</style>
""", unsafe_allow_html=True)


# ── Session State ───────────────────────────────────────────────────────────────
for k, v in {
    "messages": [], "qa_chain": None, "chat_history_pairs": [],
    "pdf_name": None, "doc_summary": None, "doc_stats": {},
    "processing": False, "pending_question": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 8px">
      <span style="font-size:22px;font-weight:800;background:linear-gradient(135deg,#58A6FF,#BC8CFF);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">🧠 DocuMind AI</span><br>
      <span style="font-size:12px;color:#8B949E;">RAG · LangChain · Groq</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**🔑 Google API Key** (for embeddings)")
    google_api_key = st.text_input(
        "google_api_key", label_visibility="collapsed",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        placeholder="AIza..."
    )
    st.caption("Free key → [aistudio.google.com](https://aistudio.google.com/app/apikey)")

    st.markdown("**🔑 Groq API Key** (for AI chat)")
    groq_api_key = st.text_input(
        "groq_api_key", label_visibility="collapsed",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_..."
    )
    st.caption("Free key → [console.groq.com](https://console.groq.com/keys)")

    if not google_api_key or not groq_api_key:
        st.warning("⚠️ Both API keys required above.")

    st.divider()

    # Upload
    st.markdown("**📂 Upload Document**")
    uploaded_file = st.file_uploader(
        "pdf_upload", label_visibility="collapsed", type=["pdf"]
    )

    if uploaded_file and (not google_api_key or not groq_api_key):
        st.warning("⚠️ Enter both API keys above.")

    if uploaded_file and google_api_key and groq_api_key:
        if st.button("🚀 Analyze Document", use_container_width=True, type="primary"):
            st.session_state.update({
                "messages": [], "chat_history_pairs": [],
                "qa_chain": None, "doc_summary": None, "processing": True,
            })
            with st.spinner("🔍 Embedding & indexing your document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    reader = PdfReader(tmp_path)
                    npages = len(reader.pages)
                    st.session_state.doc_stats = {
                        "pages": npages,
                        "read_time": f"~{max(1, npages * 2)} min",
                        "words": f"~{npages * 300:,}",
                    }
                    st.session_state.doc_summary = generate_summary(tmp_path, groq_api_key)

                    vector_store = load_existing_vectorstore(tmp_path, google_api_key)
                    if not vector_store:
                        with st.spinner("🧠 Building knowledge base..."):
                            vector_store = process_pdf(tmp_path, google_api_key)

                    st.session_state.qa_chain = create_qa_chain(vector_store, groq_api_key)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.processing = False
                    st.success("✅ Document ready!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ {e}")
                    st.session_state.processing = False

    # Analytics
    if st.session_state.qa_chain and st.session_state.doc_stats:
        st.divider()
        st.markdown("**📊 Document Stats**")
        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.doc_stats.get("pages", 0))
        c2.metric("Read Time", st.session_state.doc_stats.get("read_time", "—"))
        st.caption(f"Est. {st.session_state.doc_stats.get('words','—')} words")

        st.divider()
        st.markdown("**⚙️ Actions**")
        if st.session_state.messages:
            log = f"DocuMind AI — Chat Log\nDocument: {st.session_state.pdf_name}\n{'='*48}\n\n"
            for m in st.session_state.messages:
                log += f"[{'You' if m['role']=='user' else 'AI'}]\n{m['content']}\n\n"
            st.download_button("💾 Export Chat", data=log,
                               file_name="documind_chat.txt", mime="text/plain",
                               use_container_width=True)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history_pairs = []
            st.rerun()


# ── MAIN AREA ───────────────────────────────────────────────────────────────────

if not st.session_state.qa_chain:
    # ── HERO ──────────────────────────────────────────────────────────────────
    # Center column trick — narrow padding columns either side
    _, hero_col, _ = st.columns([1, 8, 1])
    with hero_col:
        st.markdown("""
        <div style="text-align:center; padding: 48px 0 32px;">
          <div style="font-size:72px; margin-bottom:16px; filter:drop-shadow(0 0 28px rgba(88,166,255,0.55));
               animation:none;">🧠</div>
          <div class="hero-title">DocuMind AI</div>
          <p class="hero-sub">Upload any PDF and have an intelligent conversation with it.<br>
          Powered by Google Gemini &amp; semantic vector search.</p>
          <div style="text-align:center; margin-bottom:40px;">
            <span class="badge-pill"><b>⚡</b> Gemini 1.5 Flash</span>
            <span class="badge-pill"><b>🗄️</b> ChromaDB</span>
            <span class="badge-pill"><b>🔗</b> LangChain</span>
            <span class="badge-pill"><b>🆓</b> 100% Free API</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── FEATURE CARDS (using st.columns — perfectly aligned) ──────────────────
    _, cards_col, _ = st.columns([0.5, 9, 0.5])
    with cards_col:
        r1c1, r1c2, r1c3 = st.columns(3, gap="medium")

        with r1c1:
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">📄</span>
              <div class="feat-title">Any PDF Document</div>
              <div class="feat-desc">Research papers, contracts, financial reports, textbooks — any PDF up to 200MB supported.</div>
            </div>""", unsafe_allow_html=True)

        with r1c2:
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">🔍</span>
              <div class="feat-title">Semantic Search</div>
              <div class="feat-desc">Finds the most relevant paragraphs by meaning — not just keyword matching. Powered by Google's embedding model.</div>
            </div>""", unsafe_allow_html=True)

        with r1c3:
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">🎯</span>
              <div class="feat-title">Zero Hallucination</div>
              <div class="feat-desc">Gemini is strictly grounded to your document. If the answer isn't there, it clearly says so.</div>
            </div>""", unsafe_allow_html=True)

        st.write("")

        r2c1, r2c2, r2c3 = st.columns(3, gap="medium")

        with r2c1:
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">✨</span>
              <div class="feat-title">Auto Summarization</div>
              <div class="feat-desc">Instantly generates a 3-point executive summary the moment your document is processed.</div>
            </div>""", unsafe_allow_html=True)

        with r2c2:
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">💬</span>
              <div class="feat-title">Conversation Memory</div>
              <div class="feat-desc">Ask follow-up questions naturally. The AI remembers your full conversation context.</div>
            </div>""", unsafe_allow_html=True)

        with r2c3:
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">📎</span>
              <div class="feat-title">Source Citations</div>
              <div class="feat-desc">Every answer shows exact page numbers from your document for full transparency and verification.</div>
            </div>""", unsafe_allow_html=True)

    # ── HOW IT WORKS ──────────────────────────────────────────────────────────
    _, steps_col, _ = st.columns([0.5, 9, 0.5])
    with steps_col:
        st.markdown('<div class="section-label">⚙️ How it works</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4, gap="small")

        for col, num, title, desc in [
            (s1, "1", "Upload PDF", "Choose any PDF document from your device."),
            (s2, "2", "Chunk & Embed", "Text is split into paragraphs and converted into math vectors."),
            (s3, "3", "Stored in ChromaDB", "Vectors saved locally on disk for instant retrieval."),
            (s4, "4", "Ask Anything", "Gemini reads the best-matching chunks and generates a cited answer."),
        ]:
            with col:
                st.markdown(f"""
                <div class="step-wrap">
                  <div class="step-num">{num}</div>
                  <div class="step-title">{title}</div>
                  <div class="step-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


else:
    # ── ACTIVE CHAT PAGE ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="chat-doc-bar">
      <span style="font-size:28px;">📄</span>
      <div>
        <div class="chat-doc-name">{st.session_state.pdf_name}</div>
        <div class="chat-doc-meta">
          {st.session_state.doc_stats.get('pages','?')} pages &nbsp;·&nbsp;
          {st.session_state.doc_stats.get('read_time','?')} read &nbsp;·&nbsp;
          {st.session_state.doc_stats.get('words','?')} words &nbsp;·&nbsp;
          <span style="color:#3FB950;">● Vector index ready</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Auto-Summary
    if st.session_state.doc_summary:
        summary_html = st.session_state.doc_summary.replace("\n", "<br>")
        st.markdown(f"""
        <div class="summary-box">
          <div class="summary-label">✨ AI Executive Summary</div>
          <div class="summary-body">{summary_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # Previous messages
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🧠"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"📎 {len(message['sources'])} Source Citations", expanded=False):
                    for doc in message["sources"]:
                        pg = doc.metadata.get("page", "?") + 1
                        snippet = doc.page_content[:380].strip().replace("\n", " ")
                        st.markdown(f"**📄 Page {pg}**")
                        st.caption(f"{snippet}...")
                        st.divider()

    # Suggested questions (only when chat empty)
    if not st.session_state.messages:
        st.markdown('<p class="suggestion-hint">💡 Try asking</p>', unsafe_allow_html=True)
        sq1, sq2, sq3 = st.columns(3, gap="small")
        qs = [
            "What is the main topic of this document?",
            "What are the key findings or conclusions?",
            "Are there any risks or challenges mentioned?",
        ]
        for col, q in zip([sq1, sq2, sq3], qs):
            if col.button(q, use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    # Chat input
    user_q = st.chat_input("Ask anything about your document...", disabled=st.session_state.processing)

    if st.session_state.pending_question:
        user_q = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_q)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    result = run_qa(
                        st.session_state.qa_chain,
                        user_q,
                        st.session_state.chat_history_pairs,
                    )
                    answer = result.get("answer", "Could not generate an answer.")
                    sources = result.get("source_documents", [])

                    st.session_state.chat_history_pairs.append((user_q, answer))
                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📎 {len(sources)} Source Citations", expanded=False):
                            for doc in sources:
                                pg = doc.metadata.get("page", "?") + 1
                                snippet = doc.page_content[:380].strip().replace("\n", " ")
                                st.markdown(f"**📄 Page {pg}**")
                                st.caption(f"{snippet}...")
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    err = f"⚠️ Error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
