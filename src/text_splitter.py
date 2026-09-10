"""
Text Splitter Module
Splits ingested documents into semantically coherent chunks using RecursiveCharacterTextSplitter,
ensuring chunk overlap to maintain context across boundaries while preserving metadata.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Splits documents into overlapping chunks while preserving and enhancing metadata.

    Args:
        documents: List of Document objects from document_loader.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters to overlap between contiguous chunks.

    Returns:
        List of chunked Document objects with unique chunk_id metadata.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        is_separator_regex=False
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata with unique chunk identifier
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "doc")
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{source}_p{page}_c{idx}"

    return chunks


def get_chunk_statistics(chunks: List[Document]) -> Dict[str, Any]:
    """Computes summary statistics for a collection of chunked documents."""
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "documents": {}
        }

    lengths = [len(c.page_content) for c in chunks]
    doc_counts: Dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        doc_counts[src] = doc_counts.get(src, 0) + 1

    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_chunk_size": min(lengths) if lengths else 0,
        "max_chunk_size": max(lengths) if lengths else 0,
        "documents": doc_counts
    }
