"""
Retriever Module
Performs semantic similarity retrieval on FAISS vector stores, supporting top-k selection,
document-level metadata filtering, similarity score calculations, and citation formatting.
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class ContextRetriever:
    """Configurable retriever wrapper around FAISS with filtering and citation support."""

    def __init__(self, vector_store: FAISS, default_k: int = 4):
        self.vector_store = vector_store
        self.default_k = default_k

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_doc: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k most semantically relevant documents along with similarity scores.

        Args:
            query: The natural language search query.
            k: Number of documents to retrieve (defaults to self.default_k).
            filter_doc: Optional filename to restrict search to a specific document.

        Returns:
            List of (Document, score) tuples.
        """
        top_k = k or self.default_k
        filter_dict = {"source": filter_doc} if filter_doc and filter_doc != "All Documents" else None

        try:
            if filter_dict:
                # FAISS similarity search with metadata filter
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k
                )
            return results
        except Exception as e:
            # Fallback if filter not supported by underlying FAISS version
            print(f"Retrieval warning: {e}. Falling back to standard search.")
            docs = self.vector_store.similarity_search(query=query, k=top_k)
            if filter_doc and filter_doc != "All Documents":
                docs = [d for d in docs if d.metadata.get("source") == filter_doc]
            return [(d, 0.0) for d in docs]

    @staticmethod
    def format_context_string(retrieved_items: List[Tuple[Document, float]]) -> str:
        """
        Formats retrieved chunks into an unambiguous context string for the LLM prompt,
        clearly labeling document names and page numbers.
        """
        if not retrieved_items:
            return "No relevant documents found."

        context_blocks = []
        for idx, (doc, _) in enumerate(retrieved_items, start=1):
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", 1)
            content = doc.page_content.strip()
            block = f"[Source {idx}: {source} — Page {page}]\n{content}"
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)

    @staticmethod
    def extract_sources(retrieved_items: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """
        Extracts structured source citations for UI display.
        Returns a list of dicts with source, page, score, and content snippet.
        """
        sources = []
        seen = set()
        for doc, score in retrieved_items:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 1)
            key = (source, page)
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:280] + "..."

            sources.append({
                "source": source,
                "page": page,
                "score": round(float(score), 4),
                "snippet": snippet,
                "chunk_id": doc.metadata.get("chunk_id", "")
            })
            seen.add(key)

        return sources
