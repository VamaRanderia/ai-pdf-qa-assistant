import unittest
from src.pdf_processor import PDFProcessor, DocumentChunk


class TestPDFProcessor(unittest.TestCase):

    def test_clean_text(self):
        raw = "Hello   world!\r\n\r\n\r\n\r\nThis is a    test.\xa0Done."
        cleaned = PDFProcessor.clean_text(raw)
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\xa0", cleaned)
        self.assertNotIn("\r\n", cleaned)
        self.assertEqual("Hello world!\n\nThis is a test. Done.", cleaned)

    def test_chunk_document_basic(self):
        processor = PDFProcessor(chunk_size=100, chunk_overlap=20)
        sample_pages = [
            {
                "page": 1,
                "text": "Artificial intelligence is revolutionizing document processing. Machine learning models can extract insights from unstructured text rapidly and accurately.",
                "doc_name": "ai_overview.pdf"
            },
            {
                "page": 2,
                "text": "Retrieval Augmented Generation combines vector search with large language models to answer domain-specific questions.",
                "doc_name": "ai_overview.pdf"
            }
        ]
        chunks, metadata = processor.chunk_document(sample_pages)

        self.assertGreater(len(chunks), 0)
        self.assertEqual(metadata.total_pages, 2)
        self.assertEqual(metadata.filename, "ai_overview.pdf")
        self.assertEqual(metadata.total_chunks, len(chunks))

        # Check chunk structure
        first_chunk = chunks[0]
        self.assertIsInstance(first_chunk, DocumentChunk)
        self.assertEqual(first_chunk.chunk_id, 0)
        self.assertEqual(first_chunk.page_number, 1)
        self.assertEqual(first_chunk.doc_name, "ai_overview.pdf")
        self.assertGreater(len(first_chunk.text), 0)
        self.assertGreater(first_chunk.token_count, 0)

    def test_chunk_document_overlap(self):
        processor = PDFProcessor(chunk_size=50, chunk_overlap=15)
        text = "The quick brown fox jumps over the lazy dog. The fox is very fast and agile. The dog was sleeping soundly."
        pages = [{"page": 1, "text": text, "doc_name": "sample.pdf"}]
        chunks, _ = processor.chunk_document(pages)
        
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(chunk.page_number, 1)
            self.assertEqual(chunk.doc_name, "sample.pdf")


if __name__ == "__main__":
    unittest.main()
