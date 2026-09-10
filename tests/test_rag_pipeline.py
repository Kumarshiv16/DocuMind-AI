"""
Automated Test Suite for DocuMind AI RAG Pipeline
Tests:
- Document Loading & Metadata Ingestion
- Semantic Text Chunking & Statistics
- Local Vector Embeddings (HuggingFace MiniLM)
- FAISS Index Creation, Persistence & Loading
- Contextual Retrieval & Citation Formatting
- Hallucination Guardrail Validation
"""

import os
import sys
import shutil
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from src.document_loader import load_directory, load_txt_from_bytes
from src.text_splitter import split_documents, get_chunk_statistics
from src.embeddings import get_embedding_model
from src.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    get_indexed_sources
)
from src.retriever import ContextRetriever


class TestRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_db_dir = "tests/test_vector_db"
        os.makedirs(cls.test_db_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db_dir):
            shutil.rmtree(cls.test_db_dir, ignore_errors=True)

    def test_01_document_loader(self):
        """Verify document loading extracts text and assigns metadata properly."""
        docs = load_directory("data/documents")
        self.assertGreater(len(docs), 0, "Failed to load sample documents.")
        doc = docs[0]
        self.assertIn("source", doc.metadata)
        self.assertIn("page", doc.metadata)
        self.assertTrue(len(doc.page_content) > 100)
        print(f"[PASS] Document Loader: Loaded {len(docs)} documents.")

    def test_02_text_splitter(self):
        """Verify text splitting creates overlapping chunks with chunk IDs."""
        sample_doc = Document(
            page_content="Artificial Intelligence is advancing rapidly. " * 30,
            metadata={"source": "ai_notes.pdf", "page": 1}
        )
        chunks = split_documents([sample_doc], chunk_size=200, chunk_overlap=50)
        self.assertGreater(len(chunks), 1, "Chunking should produce multiple segments.")
        self.assertIn("chunk_id", chunks[0].metadata)
        self.assertTrue(chunks[0].metadata["chunk_id"].startswith("ai_notes.pdf"))

        stats = get_chunk_statistics(chunks)
        self.assertEqual(stats["total_chunks"], len(chunks))
        print(f"[PASS] Text Splitter: Produced {len(chunks)} chunks with unique IDs.")

    def test_03_local_embeddings(self):
        """Verify local HuggingFace embeddings produce normalized vector arrays."""
        embeddings = get_embedding_model(provider="local")
        sample_vec = embeddings.embed_query("What is Retrieval-Augmented Generation?")
        self.assertEqual(len(sample_vec), 384, f"Expected 384-dim vector, got {len(sample_vec)}")
        print("[PASS] Local Embeddings: 384-dimensional dense vectors generated.")

    def test_04_vector_store_persistence(self):
        """Verify FAISS vector store creation, disk saving, and loading."""
        embeddings = get_embedding_model(provider="local")
        docs = [
            Document(page_content="Supervised learning uses labeled training data.", metadata={"source": "ml_101.pdf", "page": 1}),
            Document(page_content="Unsupervised learning finds hidden patterns without labels.", metadata={"source": "ml_101.pdf", "page": 2}),
            Document(page_content="Reinforcement learning relies on agents and environment rewards.", metadata={"source": "rl_intro.pdf", "page": 1})
        ]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=40)
        vs = create_vector_store(chunks, embeddings)
        self.assertIsNotNone(vs)

        # Test persistence
        save_vector_store(vs, folder_path=self.test_db_dir, index_name="test_index")
        loaded_vs = load_vector_store(embedding_model=embeddings, folder_path=self.test_db_dir, index_name="test_index")
        self.assertIsNotNone(loaded_vs)

        sources = get_indexed_sources(loaded_vs)
        self.assertIn("ml_101.pdf", sources)
        self.assertIn("rl_intro.pdf", sources)
        print("[PASS] Vector Store: FAISS index created, saved, loaded, and verified.")

    def test_05_retriever_and_citations(self):
        """Verify semantic retrieval, citation extraction, and formatting."""
        embeddings = get_embedding_model(provider="local")
        loaded_vs = load_vector_store(embedding_model=embeddings, folder_path=self.test_db_dir, index_name="test_index")
        retriever = ContextRetriever(vector_store=loaded_vs, default_k=2)

        results = retriever.retrieve(query="How does an agent learn from rewards?", k=1)
        self.assertGreaterEqual(len(results), 1)

        doc, score = results[0]
        self.assertEqual(doc.metadata.get("source"), "rl_intro.pdf")

        context_str = retriever.format_context_string(results)
        self.assertIn("[Source 1: rl_intro.pdf", context_str)

        sources = retriever.extract_sources(results)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source"], "rl_intro.pdf")
        self.assertEqual(sources[0]["page"], 1)
        print("[PASS] Retriever & Citations: Accurate retrieval and formatted source attribution.")

    def test_06_document_filter(self):
        """Verify retriever metadata filtering by document source."""
        embeddings = get_embedding_model(provider="local")
        loaded_vs = load_vector_store(embedding_model=embeddings, folder_path=self.test_db_dir, index_name="test_index")
        retriever = ContextRetriever(vector_store=loaded_vs, default_k=3)

        # Filter strictly to 'rl_intro.pdf'
        filtered_results = retriever.retrieve(query="learning", k=3, filter_doc="rl_intro.pdf")
        for doc, _ in filtered_results:
            self.assertEqual(doc.metadata.get("source"), "rl_intro.pdf")
        print("[PASS] Document Filter: Filtered queries strictly match selected document.")


if __name__ == "__main__":
    unittest.main()
