import unittest
from src.config import RAGConfig
from src.pdf_processor import DocumentChunk
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline, RAGResponse


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.config = RAGConfig(
            gemini_api_key="",
            top_k=2,
            similarity_threshold=0.01
        )
        self.vector_store = VectorStore(api_key=None)
        
        # Populate test chunks
        chunks = [
            DocumentChunk(
                chunk_id=0,
                doc_name="finance_report.pdf",
                page_number=1,
                text="Net revenue for Q3 increased by 14% reaching 4.2 billion dollars."
            ),
            DocumentChunk(
                chunk_id=1,
                doc_name="finance_report.pdf",
                page_number=2,
                text="Operating expenses grew by 5% driven primarily by research and development investments."
            )
        ]
        self.vector_store.add_chunks(chunks)
        self.pipeline = RAGPipeline(config=self.config, vector_store=self.vector_store)

    def test_build_context(self):
        retrieved = self.vector_store.search("revenue growth", top_k=2)
        context = self.pipeline.build_context(retrieved)
        self.assertIn("finance_report.pdf", context)
        self.assertIn("Page: 1", context)
        self.assertIn("4.2 billion dollars", context)

    def test_build_prompt(self):
        context = "Some sample context from page 1"
        prompt = self.pipeline.build_prompt("What is the revenue?", context)
        self.assertIn("What is the revenue?", prompt)
        self.assertIn("Some sample context from page 1", prompt)
        self.assertIn("Context:", prompt)

    def test_query_extractive_fallback(self):
        # Query without Gemini API key -> should produce graceful extractive response
        resp = self.pipeline.query("What was the net revenue in Q3?")
        self.assertIsInstance(resp, RAGResponse)
        self.assertFalse(resp.has_llm_answer)
        self.assertGreater(len(resp.sources), 0)
        self.assertEqual(resp.sources[0].doc_name, "finance_report.pdf")
        self.assertIn("4.2 billion dollars", resp.answer)
        self.assertEqual(len(resp.citations), len(resp.sources))

    def test_query_no_results(self):
        # Empty vector store
        empty_pipeline = RAGPipeline(config=self.config, vector_store=VectorStore())
        resp = empty_pipeline.query("Is there any quantum computing data?")
        self.assertEqual(len(resp.sources), 0)
        self.assertIn("No relevant content found", resp.answer)


if __name__ == "__main__":
    unittest.main()
