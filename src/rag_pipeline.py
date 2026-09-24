import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from src.config import RAGConfig, default_config
from src.pdf_processor import DocumentChunk
from src.vector_store import VectorStore

# Optional Gemini SDK import
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


@dataclass
class Citation:
    """Citation reference for a specific claim in the answer."""
    doc_name: str
    page: int
    chunk_id: int
    snippet: str


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    question: str
    answer: str
    sources: List[DocumentChunk] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    has_llm_answer: bool = False
    model_used: str = "Extractive (Offline)"


class RAGPipeline:
    """
    Orchestrates the retrieval and generative question-answering pipeline.
    """

    def __init__(self, config: Optional[RAGConfig] = None, vector_store: Optional[VectorStore] = None):
        self.config = config or default_config
        self.vector_store = vector_store or VectorStore(
            api_key=self.config.gemini_api_key,
            embedding_model=self.config.embedding_model_name,
            use_hybrid_search=self.config.use_hybrid_search,
            keyword_weight=self.config.keyword_boost_weight
        )

    def set_api_key(self, api_key: str):
        """Dynamically update Gemini API key."""
        self.config.gemini_api_key = api_key
        self.vector_store.api_key = api_key
        if api_key and HAS_GENAI:
            try:
                genai.configure(api_key=api_key)
                self.vector_store.is_using_gemini = True
            except Exception:
                self.vector_store.is_using_gemini = False

    def build_context(self, retrieved: List[Tuple[DocumentChunk, float]]) -> str:
        """Formats retrieved chunks into a clean, annotated context block."""
        context_blocks = []
        for chunk, score in retrieved:
            block = (
                f"--- [Document: {chunk.doc_name} | Page: {chunk.page_number} | "
                f"Chunk ID: {chunk.chunk_id}] ---\n"
                f"{chunk.text}"
            )
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def build_prompt(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Constructs an instruction-tuned prompt enforcing citation accuracy."""
        history_text = ""
        if conversation_history:
            history_lines = []
            for msg in conversation_history[-6:]:  # last 3 turns
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_lines.append(f"{role}: {msg.get('content', '')}")
            history_text = "Prior Conversation:\n" + "\n".join(history_lines) + "\n\n"

        prompt = f"""You are an expert AI Document Assistant.
Answer the user's question accurately using ONLY the provided context excerpts below.

Guidelines:
1. Ground every claim directly in the context.
2. For each key point, cite the source document and page number in brackets, e.g., `[Doc: filename.pdf, Page: 3]`.
3. If the answer cannot be determined from the context, explicitly state: "The provided documents do not contain sufficient information to answer this question."
4. Be concise, well-structured, and factual.

{history_text}Context:
{context}

Question:
{question}

Answer:"""
        return prompt

    def generate_extractive_fallback(
        self, question: str, retrieved: List[Tuple[DocumentChunk, float]]
    ) -> str:
        """
        Extractive answer when no API key is provided or offline.
        Synthesizes the most relevant excerpts.
        """
        if not retrieved:
            return "No matching information found in the uploaded documents."

        answer_parts = [
            "ℹ️ **Running in Extractive / Demo Mode** (No Gemini API key supplied).\n\n"
            "Here are the most relevant findings from your documents:\n"
        ]
        for chunk, score in retrieved:
            answer_parts.append(
                f"- **[{chunk.doc_name} - Page {chunk.page_number}]** (Relevance: {score:.0%}):\n"
                f"  > \"{chunk.text}\"\n"
            )

        answer_parts.append(
            "\n*💡 To enable full conversational answers, enter a Gemini API key in the sidebar.*"
        )
        return "\n".join(answer_parts)

    def query(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> RAGResponse:
        """
        Executes end-to-end question answering:
        Retrieval -> Context Synthesis -> Generative / Extractive Response.
        """
        # 1. Retrieve relevant chunks
        retrieved = self.vector_store.search(
            query=question,
            top_k=self.config.top_k,
            similarity_threshold=self.config.similarity_threshold
        )

        sources = [c for c, _ in retrieved]
        citations = [
            Citation(
                doc_name=c.doc_name,
                page=c.page_number,
                chunk_id=c.chunk_id,
                snippet=c.text[:150] + "..." if len(c.text) > 150 else c.text
            )
            for c in sources
        ]

        if not retrieved:
            return RAGResponse(
                question=question,
                answer="No relevant content found in the uploaded documents for this query. "
                       "Please try rephrasing or checking if the topic is covered in your PDF.",
                sources=[],
                citations=[],
                has_llm_answer=False,
                model_used="None"
            )

        # 2. Check if Gemini API is available and key is configured
        api_key = self.config.gemini_api_key
        if api_key and HAS_GENAI:
            try:
                genai.configure(api_key=api_key)
                context = self.build_context(retrieved)
                prompt = self.build_prompt(question, context, conversation_history)
                
                model = genai.GenerativeModel(self.config.llm_model_name)
                generation_config = {
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens
                }
                
                response = model.generate_content(prompt, generation_config=generation_config)
                generated_text = response.text.strip()

                return RAGResponse(
                    question=question,
                    answer=generated_text,
                    sources=sources,
                    citations=citations,
                    has_llm_answer=True,
                    model_used=self.config.llm_model_name
                )
            except Exception as exc:
                fallback = self.generate_extractive_fallback(question, retrieved)
                return RAGResponse(
                    question=question,
                    answer=f"⚠️ LLM Error ({exc}). Falling back to extracted citations:\n\n{fallback}",
                    sources=sources,
                    citations=citations,
                    has_llm_answer=False,
                    model_used="Extractive Fallback"
                )

        # 3. Extractive response when running offline or without key
        fallback_answer = self.generate_extractive_fallback(question, retrieved)
        return RAGResponse(
            question=question,
            answer=fallback_answer,
            sources=sources,
            citations=citations,
            has_llm_answer=False,
            model_used="Extractive (Offline)"
        )
