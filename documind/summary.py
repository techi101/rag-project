"""
documind/summary.py — the 3-bullet summary shown in the sidebar right after upload.
"""

# WHAT THIS FILE IS: a quick first look at the uploaded PDF, shown in the sidebar.
# No index or retrieval: sends the first 5 pages' text to the LLM, asks for 3 bullets.
# Example: generate_summary("handbook.pdf", "gsk_...") -> "- Introduces SQL and MySQL...\n- ...\n- ..." (illustrative).
# Flow: PDF -> first 5 pages -> prompt -> Groq LLM -> 3 bullets

from langchain_community.document_loaders import PyPDFLoader  # PDF -> one Document per page
from langchain_groq import ChatGroq  # calls an LLM hosted on Groq
from langchain_core.prompts import ChatPromptTemplate  # prompt with {blanks} to fill
from langchain_core.output_parsers import StrOutputParser  # LLM reply -> plain text

from documind.settings import GEN_MODEL, SUMMARY_PAGES, SUMMARY_TEMPERATURE  # model, 5 pages, 0.3


# IN: PDF path + Groq API key -> OUT: 3-bullet summary text (or an error line)
# e.g. MySQL Handbook -> "- Introduces SQL and MySQL...\n- ...\n- ..." (illustrative)
def generate_summary(pdf_path: str, groq_api_key: str) -> str:
    """
    Reads the first few pages of the document and generates a 3-bullet-point
    executive summary to instantly orient the user.
    """
    try:  # bad key / rate limit / no internet -> except
        # ── Step 1: read the first 5 pages ──
        loader = PyPDFLoader(pdf_path)  # reader for this PDF
        pages = loader.load()  # one Document per page
        
        # fewer pages = fewer tokens = faster and cheaper
        text_to_summarize = "\n".join([page.page_content for page in pages[:SUMMARY_PAGES]])
        
        # ── Step 2: set up the LLM ──
        llm = ChatGroq(  # separate client for the summary
            model=GEN_MODEL,  # "openai/gpt-oss-20b"
            groq_api_key=groq_api_key,  # user's Groq key
            temperature=SUMMARY_TEMPERATURE,  # 0.3: freer wording than Q&A's 0.1
        )  # no max_tokens: Groq's default applies
        
        # ── Step 3: 2-message prompt (rules + document text) ──
        prompt = ChatPromptTemplate.from_messages([  # list of (role, text) messages
            # system = standing rules: exactly 3 bullets, no intro or outro
            ("system", "You are an expert analyst. Provide a brief, 3-bullet-point executive summary of the following document excerpt. Be concise and professional. Do not include any introductory or concluding remarks, just the 3 bullet points."),
            ("human", "Document text:\n{text}")  # {text} = the 5 pages
        ])
        
        # ── Step 4: run prompt -> LLM -> text ──
        chain = prompt | llm | StrOutputParser()  # "|" pipes each step's output on
        return chain.invoke({"text": text_to_summarize})  # fill {text}, return the bullets
    except Exception as e:  # anything failed
        return f"- Could not generate summary. ({str(e)})"  # sidebar shows the error, no crash
