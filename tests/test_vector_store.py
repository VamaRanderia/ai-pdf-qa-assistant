import unittest
import numpy as np
from src.pdf_processor import DocumentChunk
from src.vector_store import VectorStore, OfflineTFIDFEmbedder


class TestVectorStore(unittest.TestCase):

    def test_offline_tfidf_embedder(self):
        docs = [
            "Quantum computers leverage superposition and entanglement.",
            "Classical computers process digital bits using binary logic gates.",
            "Machine learning algorithms train on historical data to predict outcomes."
        ]
        embedder = OfflineTFIDFEmbedder()
        embedder.fit(docs)

        self.assertTrue(embedder.is_fitted)
        self.assertGreater(len(embedder.vocabulary), 0)

        vec = embedder.transform("quantum superposition")
        self.assertEqual(len(vec), len(embedder.vocabulary))
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_vector_store_search(self):
        store = VectorStore(api_key=None, use_hybrid_search=True, keyword_weight=0.3)
        
        chunks = [
            DocumentChunk(
                chunk_id=0,
                doc_name="physics.pdf",
                page_number=1,
                text="Thermodynamics is the branch of physics that deals with heat, work, and temperature."
            ),
            DocumentChunk(
                chunk_id=1,
                doc_name="cs.pdf",
                page_number=3,
                text="A binary search algorithm has a logarithmic time complexity of O(log n)."
            ),
            DocumentChunk(
                chunk_id=2,
                doc_name="biology.pdf",
                page_number=10,
                text="Photosynthesis is the biological process used by plants to convert light into chemical energy."
            )
        ]
        
        store.add_chunks(chunks)
        self.assertEqual(len(store.chunks), 3)

        # Query related to physics
        results = store.search("What is thermodynamics and heat?", top_k=2)
        self.assertGreater(len(results), 0)
        best_chunk, score = results[0]
        self.assertEqual(best_chunk.chunk_id, 0)
        self.assertEqual(best_chunk.doc_name, "physics.pdf")
        self.assertGreater(score, 0.0)

        # Query related to biology
        results_bio = store.search("plants converting light into energy", top_k=1)
        self.assertEqual(len(results_bio), 1)
        self.assertEqual(results_bio[0][0].chunk_id, 2)


if __name__ == "__main__":
    unittest.main()
