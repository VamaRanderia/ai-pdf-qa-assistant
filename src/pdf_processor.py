import io
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, BinaryIO
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


@dataclass
class DocumentChunk:
    """Represents a discrete semantic chunk of a document."""
    chunk_id: int
    doc_name: str
    page_number: int
    text: str
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentMetadata:
    """Metadata summary for a processed PDF document."""
    filename: str
    total_pages: int
    total_characters: int
    total_chunks: int


class PDFProcessor:
    """Handles text extraction and semantic chunking from PDF files."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Cleans and normalizes extracted text."""
        if not raw_text:
            return ""
        # Replace non-breaking spaces and normalize control characters
        text = raw_text.replace("\xa0", " ").replace("\r\n", "\n")
        # Collapse multiple spaces or tabs
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse 3 or more newlines into double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def extract_pages_from_stream(
        self, stream: BinaryIO, filename: str = "document.pdf"
    ) -> List[Dict[str, Any]]:
        """Extracts text page by page from a binary file-like stream."""
        if not HAS_PYPDF:
            raise ImportError(
                "pypdf is required to process PDF documents. Please run: pip install pypdf"
            )
        reader = PdfReader(stream)
        pages_data = []
        for index, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            cleaned = self.clean_text(extracted)
            if cleaned:
                pages_data.append({
                    "page": index,
                    "text": cleaned,
                    "doc_name": filename
                })
        return pages_data

    def extract_pages_from_bytes(
        self, file_bytes: bytes, filename: str = "document.pdf"
    ) -> List[Dict[str, Any]]:
        """Extracts text page by page from raw bytes."""
        stream = io.BytesIO(file_bytes)
        return self.extract_pages_from_stream(stream, filename)

    def extract_pages_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extracts text page by page from a local PDF file path."""
        with open(file_path, "rb") as f:
            filename = re.split(r'[/\\]', file_path)[-1]
            return self.extract_pages_from_stream(f, filename)

    def chunk_document(
        self,
        pages_data: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Tuple[List[DocumentChunk], DocumentMetadata]:
        """
        Splits pages into overlapping chunks preserving sentence and paragraph boundaries.
        """
        size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        
        if overlap >= size:
            overlap = max(0, size // 4)

        chunks: List[DocumentChunk] = []
        global_chunk_id = 0
        total_chars = 0
        doc_name = pages_data[0]["doc_name"] if pages_data else "document.pdf"
        max_page = max([p["page"] for p in pages_data]) if pages_data else 0

        for page_item in pages_data:
            text = page_item["text"]
            page_num = page_item["page"]
            page_doc = page_item.get("doc_name", doc_name)
            total_chars += len(text)

            page_chunks = self._recursive_split(text, size, overlap)
            for c_text in page_chunks:
                clean_chunk = c_text.strip()
                if clean_chunk:
                    # Approximate token count (roughly 4 characters per token in English)
                    token_count = max(1, len(clean_chunk) // 4)
                    chunks.append(DocumentChunk(
                        chunk_id=global_chunk_id,
                        doc_name=page_doc,
                        page_number=page_num,
                        text=clean_chunk,
                        token_count=token_count,
                        metadata={
                            "char_length": len(clean_chunk),
                            "page": page_num,
                            "doc_name": page_doc
                        }
                    ))
                    global_chunk_id += 1

        metadata = DocumentMetadata(
            filename=doc_name,
            total_pages=max_page,
            total_characters=total_chars,
            total_chunks=len(chunks)
        )
        return chunks, metadata

    def _recursive_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Splits a single block of text recursively using natural separators."""
        if len(text) <= chunk_size:
            return [text]

        # Separators ranked by priority
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
        return self._split_with_separators(text, separators, chunk_size, overlap)

    def _split_with_separators(
        self, text: str, separators: List[str], chunk_size: int, overlap: int
    ) -> List[str]:
        """Internal helper to split text hierarchically."""
        if not separators:
            # Hard cutoff if no separators left
            chunks = []
            start = 0
            step = max(1, chunk_size - overlap)
            while start < len(text):
                chunks.append(text[start:start + chunk_size])
                start += step
            return chunks

        sep = separators[0]
        splits = text.split(sep)
        chunks = []
        current_chunk = []
        current_len = 0

        for part in splits:
            part_len = len(part) + len(sep)
            if current_len + part_len > chunk_size and current_chunk:
                combined = sep.join(current_chunk)
                chunks.append(combined)
                # Apply overlap by keeping tail elements
                tail = []
                tail_len = 0
                for item in reversed(current_chunk):
                    if tail_len + len(item) + len(sep) <= overlap:
                        tail.insert(0, item)
                        tail_len += len(item) + len(sep)
                    else:
                        break
                current_chunk = tail
                current_len = tail_len

            current_chunk.append(part)
            current_len += part_len

        if current_chunk:
            combined = sep.join(current_chunk)
            chunks.append(combined)

        # Refine any chunks that still exceed chunk_size using next separator
        refined_chunks = []
        for ch in chunks:
            if len(ch) > chunk_size:
                sub_chunks = self._split_with_separators(ch, separators[1:], chunk_size, overlap)
                refined_chunks.extend(sub_chunks)
            else:
                refined_chunks.append(ch)

        return refined_chunks
