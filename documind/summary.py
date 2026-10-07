"""
documind/summary.py — the 3-bullet summary shown in the sidebar right after upload.
"""

# WHAT THIS FILE IS: a quick first look at the document. It does NOT use the index or retrieval: it simply
# sends the text of the first 5 pages to the LLM and asks for 3 bullet points.
# Real example: generate_summary("handbook.pdf", "gsk_...") -> "- Introduces SQL and MySQL...\n- ...\n- ..."
#   (illustrative text; the real one depends on the PDF and the model).
#

# PyPDFLoader = reads a PDF with the pypdf library and returns one LangChain "Document" per page.
from langchain_community.document_loaders import PyPDFLoader
# ChatGroq = calls an LLM hosted by Groq. ChatPromptTemplate = a prompt with blanks to fill in.
# StrOutputParser = takes the LLM's reply object and gives back just the plain text string.
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from documind.settings import GEN_MODEL


# IN: path of the PDF + Groq API key  ->  OUT: a short 3-bullet summary text (or an error line).
# WHY: right after upload the user sees what the document is about before asking anything.
# Example: for the MySQL Handbook -> "- Introduces SQL and MySQL...\n- ...\n- ..." (illustrative text).
def generate_summary(pdf_path: str, groq_api_key: str) -> str:
    """
    Reads the first few pages of the document and generates a 3-bullet-point
    executive summary to instantly orient the user.
    """
    # try: any failure (bad key, rate limit, no internet) is caught below so the upload still works.
    try:
        # Read the PDF again, one Document per page.
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Only use the first 5 pages to save tokens and speed up summarizing
        # pages[:5] = the first 5 pages only (fewer pages = fewer tokens = faster and cheaper).
        # Their texts are joined with a newline into one string.
        text_to_summarize = "\n".join([page.page_content for page in pages[:5]])
        
        # A separate LLM client for the summary. temperature 0.3 = a bit more free wording than the Q&A (0.1),
        # fine for a summary. No max_tokens set here, so Groq's default limit applies.
        llm = ChatGroq(
            model=GEN_MODEL,
            groq_api_key=groq_api_key,
            temperature=0.3,
        )
        
        # A 2-message prompt: system instructions (3 bullets, no intro) + the document text in the {text} blank.
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert analyst. Provide a brief, 3-bullet-point executive summary of the following document excerpt. Be concise and professional. Do not include any introductory or concluding remarks, just the 3 bullet points."),
            ("human", "Document text:\n{text}")
        ])
        
        # Same pipe as in run_qa(): prompt -> LLM -> plain text; then run it with the first 5 pages' text.
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"text": text_to_summarize})
    # except: return the error as a bullet line, so the sidebar shows what went wrong instead of crashing.
    except Exception as e:
        return f"- Could not generate summary. ({str(e)})"
