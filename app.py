"""
app.py — DocuMind AI
RAG document Q&A: Google gemini-embedding-001 + ChromaDB for retrieval,
Groq openai/gpt-oss-20b for answers.
"""

# WHAT THIS FILE IS: the "face" (the shop counter) of DocuMind AI. It draws the web page the user sees:
# a sidebar to paste API keys and upload a PDF, a welcome page, and a chat page. It does NO RAG work
# itself; it hands every real job to rag_engine.py (the "brain") and shows the results.
# Real example: you paste your Google key (starts "AIza...") and Groq key (starts "gsk_..."), upload
# the MySQL Handbook PDF, click "Analyze Document", then click the suggestion
# "What is the main topic of this document?". The answer appears in a chat bubble with an expander
# "4 Source Citations" listing the 4 retrieved chunks and their page numbers.
# Words used below:
#   Streamlit       = a Python library that turns a Python script into a web page (no HTML/JS needed).
#                     IMPORTANT: Streamlit re-runs this WHOLE file from top to bottom on every click or
#                     typed message. That is why anything we want to remember must go in session_state.
#   session_state   = Streamlit's memory box for one browser tab. It survives the re-runs above.
#                     Example: st.session_state.messages keeps the whole chat so far.
#   API key         = a secret password that lets this app use a paid/free online service (Google, Groq).
#   CSS             = the styling language of web pages (colours, fonts, sizes).
# Overall flow: sidebar (keys + upload) -> "Analyze Document" -> stats + summary + vector store + QA chain
#               saved in session_state -> chat page -> question -> run_qa() -> answer + citations shown
#
# os = read environment variables (the API keys from .env) and delete the temporary PDF file.
import os
# html = Python's built-in HTML helper. html.escape() turns "<" into "&lt;" etc., so text from the PDF
# (file name, summary) is shown as plain text and can never act as HTML code inside our page.
import html
# tempfile = makes temporary files. The uploaded PDF is only in memory; PyPDFLoader needs a real file path.
import tempfile
# streamlit = the web UI library described above. "as st" = short name, so we write st.button(...).
import streamlit as st
# PdfReader (from pypdf, a PDF-reading library) = used here only to count pages and words for the stats box.
from pypdf import PdfReader
# load_dotenv (from python-dotenv) = reads the .env file and puts GOOGLE_API_KEY / GROQ_API_KEY into
# the environment, where os.getenv() can find them.
from dotenv import load_dotenv

# Load environment variables from .env file
# After this line, os.getenv("GROQ_API_KEY") returns the key written in .env (if a .env file exists).
load_dotenv()

# Our own functions from rag_engine.py (the brain):
#   process_pdf = build a new vector store, load_existing_vectorstore = reuse a saved one,
#   create_qa_chain = make retriever + LLM + prompt, run_qa = answer one question,
#   generate_summary = 3-bullet summary, page_label = turn metadata page 0 into "1".
from rag_engine import (
    process_pdf,
    load_existing_vectorstore,
    create_qa_chain,
    run_qa,
    generate_summary,
    page_label,
)

# ── Page Config ─────────────────────────────────────────────────────────────────
# set_page_config must be the first Streamlit command. It sets the browser tab title and icon (a brain
# emoji), uses the full page width ("wide"), and opens the sidebar on start ("expanded").
st.set_page_config(
    page_title="DocuMind AI — Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────────
# Inject custom CSS (styling) into the page. Everything between the triple quotes is CSS, not Python.
# It sets the Inter font (loaded from Google Fonts), the dark GitHub-like colours (#0D1117 background),
# the styles for our own HTML blocks (.hero-title, .feat-card, .step-num, .chat-doc-bar, .summary-box)
# and smaller sizes for phones: "@media (max-width: 768px)" = only apply on screens 768 pixels wide or
# less (tablets/phones), and "@media (max-width: 480px)" = only on small phones.
# "!important" in CSS = override Streamlit's own built-in style.
# unsafe_allow_html=True = let st.markdown render raw HTML/CSS (Streamlit blocks it by default for safety).
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
# Give every session_state key a starting value, but ONLY the first time (the "if not in" check below).
# Without the check, every re-run would wipe the chat. What each key holds:
#   messages           = chat shown on screen: list of {"role": "user"/"assistant", "content": text, "sources": chunks}
#   qa_chain           = the dict from create_qa_chain() (None = no document processed yet -> welcome page)
#   chat_history_pairs = (question, answer) pairs passed to run_qa() as conversation memory
#   pdf_name           = the uploaded file name (uploaded_file.name), shown in the chat header
#   doc_summary        = text from generate_summary()
#   doc_stats          = {"pages": ..., "read_time": ..., "words": ...} for the stats box
#   processing         = True while a PDF is being indexed (disables the chat box)
#   pending_question   = a suggestion button's question waiting to be asked on the next re-run
# Loop over every (key, default value) pair of the dict.
for k, v in {
    "messages": [], "qa_chain": None, "chat_history_pairs": [],
    "pdf_name": None, "doc_summary": None, "doc_stats": {},
    "processing": False, "pending_question": None,
}.items():
    # Key missing -> this is the first run in this browser tab -> set the default.
    if k not in st.session_state:
        st.session_state[k] = v


# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
# Everything inside "with st.sidebar:" is drawn in the left sidebar.
with st.sidebar:
    # Sidebar logo: "DocuMind AI" in gradient text + the line "RAG · LangChain · Groq" (raw HTML).
    st.markdown("""
    <div style="padding:4px 0 8px">
      <span style="font-size:22px;font-weight:800;background:linear-gradient(135deg,#58A6FF,#BC8CFF);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">🧠 DocuMind AI</span><br>
      <span style="font-size:12px;color:#8B949E;">RAG · LangChain · Groq</span>
    </div>
    """, unsafe_allow_html=True)
    # st.divider() = a thin horizontal line.
    st.divider()

    st.markdown("**🔑 Google API Key** (for embeddings)")
    # Password-style text box for the Google key. "google_api_key" is the box's internal label; it is hidden
    # (label_visibility="collapsed") because the bold text above already labels it. type="password" shows dots.
    user_google_key = st.text_input(
        "google_api_key", label_visibility="collapsed",
        value="",
        type="password",
        placeholder="Enter your Google API Key (or leave blank to use host key)"
    )
    # Use the key typed in the box; if the box is empty, fall back to GOOGLE_API_KEY from .env / host secrets
    # (or "" if neither exists). This "x if x else y" form is a one-line if/else.
    google_api_key = user_google_key if user_google_key else os.getenv("GOOGLE_API_KEY", "")
    # st.caption = small grey text. Here a link to where a free key can be made.
    st.caption("Free key → [aistudio.google.com](https://aistudio.google.com/app/apikey)")

    st.markdown("**🔑 Groq API Key** (for AI chat)")
    # Same password box, for the Groq key (used for answers and the summary).
    user_groq_key = st.text_input(
        "groq_api_key", label_visibility="collapsed",
        value="",
        type="password",
        placeholder="Enter your Groq API Key (or leave blank to use host key)"
    )
    # Typed Groq key first, else GROQ_API_KEY from the environment, else "".
    groq_api_key = user_groq_key if user_groq_key else os.getenv("GROQ_API_KEY", "")
    st.caption("Free key → [console.groq.com](https://console.groq.com/keys)")

    # If either key is still empty, show a yellow warning. The app still draws, but cannot process a PDF.
    if not google_api_key or not groq_api_key:
        st.warning("⚠️ Both API keys required above.")

    st.divider()

    # Upload
    # PDF upload box. type=["pdf"] = the file picker accepts only .pdf files.
    # uploaded_file = None until the user picks a file; then it is an in-memory file object.
    st.markdown("**📂 Upload Document**")
    uploaded_file = st.file_uploader(
        "pdf_upload", label_visibility="collapsed", type=["pdf"]
    )

    # A file is chosen but a key is missing -> warn (the Analyze button is not shown in this case).
    if uploaded_file and (not google_api_key or not groq_api_key):
        st.warning("⚠️ Enter both API keys above.")

    # A file is chosen AND both keys exist -> show the Analyze button.
    if uploaded_file and google_api_key and groq_api_key:
        # st.button returns True only on the re-run caused by clicking it. type="primary" = highlighted button.
        if st.button("🚀 Analyze Document", use_container_width=True, type="primary"):
            # New document: clear the old chat, history, chain and summary, and mark that processing has started.
            st.session_state.update({
                "messages": [], "chat_history_pairs": [],
                "qa_chain": None, "doc_summary": None, "processing": True,
            })
            # st.spinner shows a turning wheel with this text while the code inside the "with" block runs.
            with st.spinner("🔍 Embedding & indexing your document..."):
                # tmp_path starts as None so the "finally" block below knows whether a temp file was created.
                tmp_path = None
                # try: if anything below fails (bad key, quota used up, scanned PDF), jump to "except" and show the error.
                try:
                    # Save the uploaded bytes into a real temporary .pdf file on disk, because PyPDFLoader and PdfReader
                    # need a file path. delete=False = keep the file after the "with" block closes it (we delete it ourselves
                    # in "finally"; on Windows a file still open cannot be read by another reader).
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Open the PDF with pypdf just to count pages.
                    reader = PdfReader(tmp_path)
                    npages = len(reader.pages)
                    # Count the real words instead of guessing per page;
                    # read time assumes ~200 words per minute.
                    # For every page: extract_text() (or "" if it returns None), split on spaces into words, count them,
                    # and add up the counts of all pages.
                    nwords = sum(len((p.extract_text() or "").split())
                                 for p in reader.pages)
                    # Save the stats shown in the sidebar and chat header.
                    # read_time: words / 200 (200 words per minute), rounded, but never below 1 minute (max(1, ...)).
                    # words: f"{nwords:,}" adds thousands commas, e.g. 12345 -> "12,345".
                    st.session_state.doc_stats = {
                        "pages": npages,
                        "read_time": f"~{max(1, round(nwords / 200))} min",
                        "words": f"{nwords:,}",
                    }
                    # Ask Groq for the 3-bullet summary of the first 5 pages (a Groq call happens on every Analyze click).
                    st.session_state.doc_summary = generate_summary(tmp_path, groq_api_key)

                    # Try to reuse a saved vector store for this exact PDF (same content hash) -> no Google calls needed.
                    vector_store = load_existing_vectorstore(tmp_path, google_api_key)
                    # None = no usable saved store (new PDF, changed settings, or incomplete) -> build it now.
                    if not vector_store:
                        # A second spinner while the slow step runs: chunk + embed with Google + save to ChromaDB.
                        with st.spinner("🧠 Building knowledge base..."):
                            vector_store = process_pdf(tmp_path, google_api_key)

                    # Build retriever + LLM + prompt once and keep them in session_state, so each question reuses them.
                    # qa_chain is no longer None, so the next re-run shows the chat page instead of the welcome page.
                    st.session_state.qa_chain = create_qa_chain(vector_store, groq_api_key)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.processing = False
                    # Green success box, then st.rerun() = restart the script now so the chat page appears immediately.
                    st.success("✅ Document ready!")
                    st.rerun()

                # except: show the error text in a red box (e.g. Google's RESOURCE_EXHAUSTED when the daily quota is
                # used up, or the "No text could be extracted" message for a scanned PDF) and stop "processing".
                except Exception as e:
                    st.error(f"❌ {e}")
                    st.session_state.processing = False
                # finally: runs whether it worked or failed.
                finally:
                    # The PDF is only needed while indexing; don't leave a copy
                    # of every upload in the temp folder.
                    # If the temp file was created and still exists, delete it.
                    if tmp_path and os.path.exists(tmp_path):
                        # try/except OSError: if deleting fails (for example the file is locked on Windows), ignore it;
                        # a leftover temp file must not crash the app.
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

    # Analytics
    # The stats and actions only appear after a document has been processed.
    if st.session_state.qa_chain and st.session_state.doc_stats:
        st.divider()
        st.markdown("**📊 Document Stats**")
        # Two side-by-side boxes: page count and read time. st.metric = a big number with a small label.
        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.doc_stats.get("pages", 0))
        c2.metric("Read Time", st.session_state.doc_stats.get("read_time", "—"))
        # Word count as small text. "—" is shown if the key is missing.
        st.caption(f"{st.session_state.doc_stats.get('words','—')} words")

        st.divider()
        st.markdown("**⚙️ Actions**")
        # The Export button only makes sense if there is at least one message.
        if st.session_state.messages:
            # Build a plain text chat log: a title, the document name, a line of 48 "=" signs as a divider.
            log = f"DocuMind AI — Chat Log\nDocument: {st.session_state.pdf_name}\n{'='*48}\n\n"
            # Add each message as "[You]" or "[AI]" followed by its text and a blank line.
            for m in st.session_state.messages:
                log += f"[{'You' if m['role']=='user' else 'AI'}]\n{m['content']}\n\n"
            # st.download_button = a button that downloads the given text as the file "documind_chat.txt".
            # mime "text/plain" tells the browser it is a plain text file.
            st.download_button("💾 Export Chat", data=log,
                               file_name="documind_chat.txt", mime="text/plain",
                               use_container_width=True)
        # Clear Chat: empty the visible chat and the memory, then re-run so the page updates. The PDF stays loaded.
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history_pairs = []
            st.rerun()


# ── MAIN AREA ───────────────────────────────────────────────────────────────────

# MAIN AREA: two possible pages. No QA chain yet -> welcome (hero) page. Otherwise -> chat page ("else" below).
if not st.session_state.qa_chain:
    # ── HERO ──────────────────────────────────────────────────────────────────
    # Center column trick — narrow padding columns either side
    # st.columns([1, 8, 1]) = 3 columns with widths in ratio 1:8:1. Only the middle one is used ("_" = unused),
    # so the content sits centred with empty space on both sides.
    _, hero_col, _ = st.columns([1, 8, 1])
    with hero_col:
        # Big title, subtitle and 4 small "badge" pills, written as HTML with the CSS classes defined above.
        st.markdown("""
        <div style="text-align:center; padding: 48px 0 32px;">
          <div style="font-size:72px; margin-bottom:16px; filter:drop-shadow(0 0 28px rgba(88,166,255,0.55));
               animation:none;">🧠</div>
          <div class="hero-title">DocuMind AI</div>
          <p class="hero-sub">Upload any PDF and have an intelligent conversation with it.<br>
          Powered by semantic vector search and Groq-hosted gpt-oss-20b.</p>
          <div style="text-align:center; margin-bottom:40px;">
            <span class="badge-pill"><b>⚡</b> Groq gpt-oss-20b</span>
            <span class="badge-pill"><b>🗄️</b> ChromaDB</span>
            <span class="badge-pill"><b>🔗</b> LangChain</span>
            <span class="badge-pill"><b>🆓</b> Free-tier APIs</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── FEATURE CARDS (using st.columns — perfectly aligned) ──────────────────
    # Same centring trick (ratio 0.5 : 9 : 0.5) for the 6 feature cards.
    _, cards_col, _ = st.columns([0.5, 9, 0.5])
    with cards_col:
        # First row: 3 equal columns, one card in each.
        r1c1, r1c2, r1c3 = st.columns(3, gap="medium")

        # Each card is one HTML block using the .feat-card / .feat-icon / .feat-title / .feat-desc styles.
        with r1c1:
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">📄</span>
              <div class="feat-title">Any PDF Document</div>
              <div class="feat-desc">Research papers, contracts, financial reports, textbooks — any text-based PDF (scanned PDFs need OCR, not yet supported).</div>
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
              <div class="feat-title">Grounded Answers</div>
              <div class="feat-desc">The model is instructed to answer only from your document and to say so when the answer isn't there.</div>
            </div>""", unsafe_allow_html=True)

        # st.write("") = an empty line, as vertical space between the two rows of cards.
        st.write("")

        # Second row of 3 cards.
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
              <div class="feat-desc">Ask follow-up questions. The AI sees your earlier questions and answers in this session.</div>
            </div>""", unsafe_allow_html=True)

        with r2c3:
            st.markdown("""
            <div class="feat-card">
              <span class="feat-icon">📎</span>
              <div class="feat-title">Source Citations</div>
              <div class="feat-desc">Every answer lists the retrieved passages and their page numbers, so you can check it.</div>
            </div>""", unsafe_allow_html=True)

    # ── HOW IT WORKS ──────────────────────────────────────────────────────────
    # "How it works" section: 4 numbered steps in 4 columns.
    _, steps_col, _ = st.columns([0.5, 9, 0.5])
    with steps_col:
        st.markdown('<div class="section-label">⚙️ How it works</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4, gap="small")

        # Loop over 4 (column, number, title, description) tuples and draw one step box in each column.
        for col, num, title, desc in [
            (s1, "1", "Upload PDF", "Choose any PDF document from your device."),
            (s2, "2", "Chunk & Embed", "Text is split into paragraphs and converted into math vectors."),
            (s3, "3", "Stored in ChromaDB", "Vectors saved locally on disk for instant retrieval."),
            (s4, "4", "Ask Anything", "The LLM reads the best-matching chunks and generates a cited answer."),
        ]:
            with col:
                # f-string with triple quotes: {num}, {title}, {desc} are filled in from the loop values.
                st.markdown(f"""
                <div class="step-wrap">
                  <div class="step-num">{num}</div>
                  <div class="step-title">{title}</div>
                  <div class="step-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

    # "<br>" = one HTML line break, extra space at the bottom of the welcome page.
    st.markdown("<br>", unsafe_allow_html=True)


# else: a document is loaded (qa_chain exists) -> show the chat page.
else:
    # ── ACTIVE CHAT PAGE ───────────────────────────────────────────────────────
    # Header bar: file name, pages, read time, words and a green "Vector index ready" dot.
    # html.escape() on the file name: a file named like "<b>x</b>.pdf" is shown as text, not run as HTML.
    # .get('pages','?') = show "?" if a stat is missing.
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
    """, unsafe_allow_html=True)

    # Auto-Summary
    # Show the summary box only if a summary exists.
    if st.session_state.doc_summary:
        # Escape the LLM's text (so it cannot inject HTML), then turn each newline into "<br>" so the 3 bullets
        # stay on separate lines inside the HTML box.
        summary_html = html.escape(st.session_state.doc_summary).replace("\n", "<br>")
        st.markdown(f"""
        <div class="summary-box">
          <div class="summary-label">✨ AI Executive Summary</div>
          <div class="summary-body">{summary_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # Previous messages
    # Re-draw the whole chat from memory. This is needed because Streamlit re-runs the script on every action
    # and the page starts blank each time.
    for message in st.session_state.messages:
        # Person icon for the user's messages, brain icon for the AI's.
        avatar = "👤" if message["role"] == "user" else "🧠"
        # st.chat_message draws one chat bubble for that role.
        with st.chat_message(message["role"], avatar=avatar):
            # st.markdown renders the text, so the LLM's bullet points and bold text display nicely.
            st.markdown(message["content"])
            # AI messages carry their source chunks; user messages do not (.get returns None -> skipped).
            if message.get("sources"):
                # A collapsed (expanded=False) section titled e.g. "4 Source Citations" (4 = TOP_K_RESULTS chunks).
                with st.expander(f"📎 {len(message['sources'])} Source Citations", expanded=False):
                    # One entry per chunk: its page number and a preview of its text.
                    for doc in message["sources"]:
                        pg = page_label(doc.metadata)
                        # Preview = the first 380 characters of the chunk, trimmed, with newlines turned into spaces so it is one
                        # compact paragraph (380 is just a preview length; the full chunk can be up to 1000 characters).
                        snippet = doc.page_content[:380].strip().replace("\n", " ")
                        st.markdown(f"**📄 Page {pg}**")
                        st.caption(f"{snippet}...")
                        st.divider()

    # Suggested questions (only when chat empty)
    # Empty chat -> offer 3 ready-made starter questions as buttons.
    if not st.session_state.messages:
        st.markdown('<p class="suggestion-hint">💡 Try asking</p>', unsafe_allow_html=True)
        sq1, sq2, sq3 = st.columns(3, gap="small")
        # The 3 starter questions. They are general, so they work for any PDF.
        qs = [
            "What is the main topic of this document?",
            "What are the key findings or conclusions?",
            "Are there any risks or challenges mentioned?",
        ]
        # zip() pairs each column with one question: (sq1, q1), (sq2, q2), (sq3, q3).
        for col, q in zip([sq1, sq2, sq3], qs):
            # Clicked -> store the question and re-run. On the re-run it is picked up just below as if typed.
            if col.button(q, use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    # Chat input
    # The chat text box pinned at the bottom. Returns the typed text once, on the re-run after Enter; else None.
    # disabled while a PDF is being processed.
    user_q = st.chat_input("Ask anything about your document...", disabled=st.session_state.processing)

    # A starter-question button was clicked on the previous run -> use that question, then clear it
    # so it is not asked again on the next re-run.
    if st.session_state.pending_question:
        user_q = st.session_state.pending_question
        st.session_state.pending_question = None

    # A question exists (typed or from a button) -> answer it.
    if user_q:
        # Save the user's message in the chat memory first.
        st.session_state.messages.append({"role": "user", "content": user_q})

        # Draw the user's bubble right away.
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_q)

        # The AI's bubble, with a spinner while the answer is being made.
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Searching document and generating answer..."):
                # try: any error from Google/Groq (bad key, rate limit, network) is caught below and shown in the chat.
                try:
                    # The real work: retrieve the top 4 chunks and ask Groq, passing the past (question, answer) pairs
                    # so follow-up questions have context.
                    result = run_qa(
                        st.session_state.qa_chain,
                        user_q,
                        st.session_state.chat_history_pairs,
                    )
                    # .get with a default: if "answer" were missing, show "Could not generate an answer." instead of crashing.
                    answer = result.get("answer", "Could not generate an answer.")
                    sources = result.get("source_documents", [])

                    # Remember this turn as a (question, answer) pair for the next question's chat history.
                    st.session_state.chat_history_pairs.append((user_q, answer))
                    st.markdown(answer)

                    # Show the source chunks under the answer, in the same collapsed format as above.
                    if sources:
                        with st.expander(f"📎 {len(sources)} Source Citations", expanded=False):
                            for doc in sources:
                                pg = page_label(doc.metadata)
                                snippet = doc.page_content[:380].strip().replace("\n", " ")
                                st.markdown(f"**📄 Page {pg}**")
                                st.caption(f"{snippet}...")
                                st.divider()

                    # Save the AI's message with its sources, so the next re-run can draw it again (loop at the top of this page).
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                # except: show the error in red and also save it as an AI message (with no sources), so it stays visible.
                # It is NOT added to chat_history_pairs, so errors are not sent to the LLM as past answers.
                except Exception as e:
                    err = f"⚠️ Error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
