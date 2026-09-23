import os
import io
import re
import numpy as np
from typing import List, Dict, Any
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

# Optional Gemini Integration
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI PDF Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4F46E5;
        padding: 10px 15px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


def extract_text_from_pdf(uploaded_file) -> List[Dict[str, Any]]:
    """Extracts text page by page from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    pages_data = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            pages_data.append({"page": page_num, "text": text})
    return pages_data


def chunk_text(pages_data: List[Dict[str, Any]], chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
    """Splits extracted page text into smaller overlapping chunks."""
    chunks = []
    chunk_id = 0
    for item in pages_data:
        text = item["text"]
        page = item["page"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "id": chunk_id,
                    "page": page,
                    "content": chunk_content
                })
                chunk_id += 1
            start += (chunk_size - overlap)
            if start >= len(text):
                break
    return chunks


def simple_keyword_search(query: str, chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    """Fallback ranking based on token overlap when no embedding model is configured."""
    query_tokens = set(re.findall(r'\w+', query.lower()))
    scored_chunks = []
    for chunk in chunks:
        chunk_tokens = set(re.findall(r'\w+', chunk["content"].lower()))
        overlap = len(query_tokens.intersection(chunk_tokens))
        if overlap > 0:
            scored_chunks.append((overlap, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_chunks[:top_k]] or chunks[:top_k]


def answer_with_gemini(api_key: str, question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """Generates an answer using Google Gemini model with the provided context."""
    if not HAS_GENAI:
        return "Error: `google-generativeai` package is not installed. Please run `pip install -r requirements.txt`."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        context = "\n\n".join([f"[Page {c['page']}]: {c['content']}" for c in relevant_chunks])
        prompt = f"""You are a helpful AI assistant answering questions based on the provided PDF document context.
If the answer cannot be found in the context, politely state that the information is not present in the document.

Context:
{context}

Question:
{question}

Answer:"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while contacting the Gemini API: {e}"


# --- Session State Initialization ---
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Enter your Google Gemini API key to enable AI-powered Q&A."
    )
    
    st.markdown("---")
    st.subheader("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.session_state.processed_file_name != uploaded_file.name:
            with st.spinner("Processing PDF document..."):
                pages = extract_text_from_pdf(uploaded_file)
                chunks = chunk_text(pages)
                st.session_state.chunks = chunks
                st.session_state.processed_file_name = uploaded_file.name
                st.session_state.chat_history = []
                st.success(f"Extracted {len(pages)} pages ({len(chunks)} chunks).")

    if st.session_state.chunks:
        st.info(f"Loaded: **{st.session_state.processed_file_name}**")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# --- Main App Header ---
st.markdown('<div class="main-title">📄 AI PDF Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any document and ask questions to get context-aware answers instantly.</div>', unsafe_allow_html=True)

# --- Chat Interface ---
if not st.session_state.chunks:
    st.info("👈 Please upload a PDF in the sidebar to get started!")
else:
    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Reference Passages"):
                    for s in msg["sources"]:
                        st.markdown(f"**Page {s['page']}**")
                        st.markdown(f"> {s['content']}")

    # User input
    if user_query := st.chat_input("Ask a question about the document..."):
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve relevant passages
        relevant_chunks = simple_keyword_search(user_query, st.session_state.chunks)

        with st.chat_message("assistant"):
            if api_key_input:
                with st.spinner("Analyzing document and generating answer..."):
                    answer = answer_with_gemini(api_key_input, user_query, relevant_chunks)
            else:
                answer = (
                    "⚠️ **No API Key Provided**: Operating in demo/retrieval-only mode.\n\n"
                    "Here are the most relevant excerpts found for your query from the document:\n\n"
                    + "\n\n".join([f"- **(Page {c['page']})**: {c['content']}" for c in relevant_chunks])
                    + "\n\n*Add a Gemini API key in the sidebar to get synthesized conversational answers!*"
                )
            
            st.markdown(answer)
            if api_key_input and relevant_chunks:
                with st.expander("📚 View Reference Passages"):
                    for s in relevant_chunks:
                        st.markdown(f"**Page {s['page']}**")
                        st.markdown(f"> {s['content']}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": relevant_chunks if api_key_input else []
        })
