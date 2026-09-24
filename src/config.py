import os
from dataclasses import dataclass, field
from typing import Optional
# Load .env file automatically if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class RAGConfig:
    """Central configuration for RAG pipeline, embedding, and LLM generation."""
    
    # Model configuration
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    llm_model_name: str = "gemini-1.5-flash"
    embedding_model_name: str = "models/text-embedding-004"
    
    # Text chunking configuration
    chunk_size: int = 600
    chunk_overlap: int = 120
    
    # Retrieval configuration
    top_k: int = 4
    similarity_threshold: float = 0.15
    use_hybrid_search: bool = True
    keyword_boost_weight: float = 0.3
    
    # Generation parameters
    temperature: float = 0.2
    max_output_tokens: int = 1024

    def validate(self) -> bool:
        """Validate key configurations."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        return True


default_config = RAGConfig()
