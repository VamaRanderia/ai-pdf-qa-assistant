import math
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.pdf_processor import DocumentChunk

import warnings

# Support modern google.genai SDK
try:
    from google import genai
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False

# Support legacy google.generativeai SDK with clean warning suppression
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    try:
        import google.generativeai as legacy_genai
        HAS_LEGACY_GENAI = True
    except ImportError:
        HAS_LEGACY_GENAI = False

HAS_GENAI = HAS_NEW_GENAI or HAS_LEGACY_GENAI


class OfflineTFIDFEmbedder:
    """
    Lightweight, deterministic local TF-IDF embedder.
    Provides fast vector search capabilities without requiring external network calls or API keys.
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.is_fitted: bool = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_]{2,}\b', text.lower())

    def fit(self, documents: List[str]):
        doc_count = len(documents)
        if doc_count == 0:
            return

        term_doc_counts: Dict[str, int] = {}
        all_terms = set()

        for doc in documents:
            tokens = set(self.tokenize(doc))
            for t in tokens:
                term_doc_counts[t] = term_doc_counts.get(t, 0) + 1
                all_terms.add(t)

        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
        self.idf = {
            term: math.log((1 + doc_count) / (1 + term_doc_counts[term])) + 1.0
            for term in self.vocabulary
        }
        self.is_fitted = True

    def transform(self, text: str) -> np.ndarray:
        if not self.is_fitted or not self.vocabulary:
            return np.zeros(max(1, len(self.vocabulary)), dtype=np.float32)

        vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        tokens = self.tokenize(text)
        if not tokens:
            return vector

        term_counts: Dict[str, int] = {}
        for t in tokens:
            if t in self.vocabulary:
                term_counts[t] = term_counts.get(t, 0) + 1

        for term, count in term_counts.items():
            idx = self.vocabulary[term]
            tf = count / len(tokens)
            vector[idx] = tf * self.idf[term]

        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


class VectorStore:
    """
    In-memory vector store supporting dense embeddings (Gemini) and sparse hybrid retrieval.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: str = "models/text-embedding-004",
        use_hybrid_search: bool = True,
        keyword_weight: float = 0.3
    ):
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.use_hybrid_search = use_hybrid_search
        self.keyword_weight = keyword_weight

        self.chunks: List[DocumentChunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.offline_embedder = OfflineTFIDFEmbedder()
        self.is_using_gemini: bool = False
        self.new_client = None

        if self.api_key:
            if HAS_NEW_GENAI:
                try:
                    self.new_client = genai.Client(api_key=self.api_key)
                    self.is_using_gemini = True
                except Exception:
                    self.new_client = None

            if not self.is_using_gemini and HAS_LEGACY_GENAI:
                try:
                    legacy_genai.configure(api_key=self.api_key)
                    self.is_using_gemini = True
                except Exception:
                    self.is_using_gemini = False

    def clear(self):
        """Reset the vector store."""
        self.chunks = []
        self.vectors = None
        self.offline_embedder = OfflineTFIDFEmbedder()

    def get_embedding(self, text: str) -> np.ndarray:
        """Computes embedding vector for a piece of text."""
        # Normalize model name for embeddings
        model_name = self.embedding_model
        if "/" in model_name:
            clean_model = model_name.split("/")[-1]
        else:
            clean_model = model_name

        if self.is_using_gemini:
            # 1. Try modern google-genai SDK
            if HAS_NEW_GENAI and self.new_client is not None:
                try:
                    res = self.new_client.models.embed_content(
                        model=clean_model,
                        contents=text
                    )
                    if hasattr(res, "embeddings") and res.embeddings:
                        vec = np.array(res.embeddings[0].values, dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        return vec / norm if norm > 0 else vec
                except Exception:
                    pass

            # 2. Try legacy google.generativeai SDK
            if HAS_LEGACY_GENAI:
                try:
                    result = legacy_genai.embed_content(
                        model=model_name if "models/" in model_name else f"models/{model_name}",
                        content=text,
                        task_type="retrieval_document"
                    )
                    vec = np.array(result["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    return vec / norm if norm > 0 else vec
                except Exception:
                    pass

        return self.offline_embedder.transform(text)

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Indexes a list of DocumentChunk objects."""
        if not chunks:
            return

        self.chunks.extend(chunks)
        all_texts = [c.text for c in self.chunks]

        # Fit local TF-IDF embedder on the full corpus
        self.offline_embedder.fit(all_texts)

        # Generate vectors
        embeddings_list = []
        for c in self.chunks:
            vec = self.get_embedding(c.text)
            embeddings_list.append(vec)

        self.vectors = np.array(embeddings_list, dtype=np.float32)

    def _keyword_score(self, query: str, doc_text: str) -> float:
        """Calculates token overlap score between query and document text."""
        q_tokens = set(re.findall(r'\b\w{2,}\b', query.lower()))
        if not q_tokens:
            return 0.0
        d_tokens = set(re.findall(r'\b\w{2,}\b', doc_text.lower()))
        overlap = len(q_tokens.intersection(d_tokens))
        return min(1.0, overlap / len(q_tokens))

    def search(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves the top-k most relevant DocumentChunks for a given query.
        Returns list of (DocumentChunk, score) tuples.
        """
        if not self.chunks or self.vectors is None or len(self.chunks) == 0:
            return []

        # Vector score via cosine similarity
        query_vec = self.get_embedding(query)
        if query_vec.shape[0] != self.vectors.shape[1]:
            # Reshape or re-transform if dimension mismatch
            query_vec = self.offline_embedder.transform(query)

        norm_q = np.linalg.norm(query_vec)
        if norm_q > 0:
            query_vec = query_vec / norm_q

        # Cosine similarity dot product
        vector_scores = np.dot(self.vectors, query_vec)

        # Ensure values in [0, 1] range
        vector_scores = np.clip(vector_scores, 0.0, 1.0)

        final_scores = []
        for idx, chunk in enumerate(self.chunks):
            v_score = float(vector_scores[idx])
            if self.use_hybrid_search:
                k_score = self._keyword_score(query, chunk.text)
                combined = (1.0 - self.keyword_weight) * v_score + self.keyword_weight * k_score
            else:
                combined = v_score

            if combined >= similarity_threshold:
                final_scores.append((chunk, combined))

        # Sort descending by score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]
