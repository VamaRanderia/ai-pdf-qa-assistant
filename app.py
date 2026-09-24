import os
import io
import json
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

from src.config import RAGConfig
from src.pdf_processor import PDFProcessor, DocumentChunk
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline, RAGResponse

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI PDF Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.6rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .citation-badge {
        display: inline-block;
        background: #e0e7ff;
        color: #4338ca;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .citation-card {
        border-left: 3px solid #6366f1;
        background: #f9fafb;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_config = RAGConfig()
    st.session_state.vector_store = VectorStore()
    st.session_state.rag_pipeline = RAGPipeline(
        config=st.session_state.rag_config,
        vector_store=st.session_state.vector_store
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = {}

if "total_chunks_count" not in st.session_state:
    st.session_state.total_chunks_count = 0


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=st.session_state.rag_config.gemini_api_key or "",
        help="Get a free Gemini API key at: https://aistudio.google.com/app/apikey"
    )

    if api_key_input != st.session_state.rag_config.gemini_api_key:
        st.session_state.rag_config.gemini_api_key = api_key_input
        st.session_state.rag_pipeline.set_api_key(api_key_input)
        if api_key_input:
            st.success("API key updated!")

    selected_model = st.selectbox(
        "LLM Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Select Gemini model for generation."
    )
    st.session_state.rag_config.llm_model_name = selected_model

    with st.expander("🛠️ Advanced RAG Parameters"):
        chunk_size = st.slider("Chunk Size (characters)", 300, 1500, 600, step=50)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 300, 120, step=20)
        top_k = st.slider("Top-K Passages to Retrieve", 1, 10, 4)
        use_hybrid = st.checkbox("Enable Hybrid Search (Keyword Boost)", value=True)

        st.session_state.rag_config.chunk_size = chunk_size
        st.session_state.rag_config.chunk_overlap = chunk_overlap
        st.session_state.rag_config.top_k = top_k
        st.session_state.rag_config.use_hybrid_search = use_hybrid
        st.session_state.vector_store.use_hybrid_search = use_hybrid

    st.markdown("---")
    st.subheader("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or multiple PDF documents to analyze."
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_docs]
        if new_files:
            with st.spinner("Processing documents & indexing vectors..."):
                processor = PDFProcessor(
                    chunk_size=st.session_state.rag_config.chunk_size,
                    chunk_overlap=st.session_state.rag_config.chunk_overlap
                )
                all_new_chunks = []
                for file_obj in new_files:
                    pages = processor.extract_pages_from_stream(file_obj, filename=file_obj.name)
                    chunks, meta = processor.chunk_document(pages)
                    st.session_state.processed_docs[file_obj.name] = meta
                    all_new_chunks.extend(chunks)

                st.session_state.vector_store.add_chunks(all_new_chunks)
                st.session_state.total_chunks_count += len(all_new_chunks)
                st.success(f"Indexed {len(new_files)} document(s) successfully!")

    # Document stats
    if st.session_state.processed_docs:
        st.markdown("**Indexed Documents:**")
        for doc_name, meta in st.session_state.processed_docs.items():
            st.caption(f"• **{doc_name}** ({meta.total_pages} pages, {meta.total_chunks} chunks)")

        if st.button("🗑️ Reset All Documents"):
            st.session_state.vector_store.clear()
            st.session_state.processed_docs = {}
            st.session_state.total_chunks_count = 0
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.chat_history:
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


# --- Main Content Area ---
st.markdown('<div class="main-title">📄 AI PDF Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    'Upload your research papers, manuals, or contracts and get accurate, cited answers in seconds.'
    '</div>',
    unsafe_allow_html=True
)

# Metric status cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Documents Loaded", len(st.session_state.processed_docs))
with col2:
    st.metric("Total Indexed Chunks", st.session_state.total_chunks_count)
with col3:
    status_label = "Gemini LLM Active" if st.session_state.rag_config.gemini_api_key else "Offline / Extractive"
    st.metric("Operating Mode", status_label)

st.markdown("---")

# Empty State Notice
if not st.session_state.processed_docs:
    st.info("👈 **Upload one or more PDF files** using the sidebar to begin asking questions.")
    
    st.markdown("#### 💡 Quick Features:")
    st.markdown("""
    - **Multi-Document Support**: Query across several uploaded PDFs at once.
    - **Grounded Citations**: Every answer provides page numbers and exact context snippets.
    - **Hybrid Retrieval**: Combines semantic embeddings with keyword boosting for higher precision.
    - **Offline Fallback**: Works immediately even without an API key using built-in extractive search!
    """)
else:
    # Render chat conversation
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("📚 View Reference Citations"):
                    for idx, cit in enumerate(message["citations"], 1):
                        st.markdown(
                            f"<div class='citation-card'>"
                            f"<span class='citation-badge'>Citation {idx}</span> "
                            f"<strong>{cit['doc_name']}</strong> — Page {cit['page']}<br>"
                            f"<em>\"{cit['snippet']}\"</em>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # Chat Input
    if user_query := st.chat_input("Ask a question about your documents..."):
        # Display user question
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching document context and generating answer..."):
                response: RAGResponse = st.session_state.rag_pipeline.query(
                    question=user_query,
                    conversation_history=st.session_state.chat_history
                )
                
                st.markdown(response.answer)

                citations_data = [
                    {
                        "doc_name": c.doc_name,
                        "page": c.page,
                        "chunk_id": c.chunk_id,
                        "snippet": c.snippet
                    }
                    for c in response.citations
                ]

                if citations_data:
                    with st.expander("📚 View Reference Citations"):
                        for idx, cit in enumerate(citations_data, 1):
                            st.markdown(
                                f"<div class='citation-card'>"
                                f"<span class='citation-badge'>Citation {idx}</span> "
                                f"<strong>{cit['doc_name']}</strong> — Page {cit['page']}<br>"
                                f"<em>\"{cit['snippet']}\"</em>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.answer,
                "citations": citations_data,
                "model_used": response.model_used
            })

    # Export Chat Option
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        export_text = f"# AI PDF Q&A Assistant - Chat Export\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n"
        for m in st.session_state.chat_history:
            role_title = "**User**" if m["role"] == "user" else "**Assistant**"
            export_text += f"{role_title}:\n{m['content']}\n\n"
            if m.get("citations"):
                export_text += "*Citations:*\n"
                for c in m["citations"]:
                    export_text += f"- [{c['doc_name']}, Page {c['page']}]: {c['snippet']}\n"
                export_text += "\n"
            export_text += "---\n\n"

        st.download_button(
            label="📥 Download Q&A Transcript (Markdown)",
            data=export_text,
            file_name=f"pdf_qa_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
