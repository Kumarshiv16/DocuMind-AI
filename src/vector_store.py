"""
Vector Store Module
Manages FAISS vector indexing, local disk persistence, incremental document additions,
and document source metadata inspection.
"""

import os
from typing import List, Optional, Set
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(
    chunks: List[Document],
    embedding_model: Embeddings
) -> FAISS:
    """
    Creates a new FAISS vector store from a list of document chunks.

    Args:
        chunks: List of split Document objects.
        embedding_model: Embeddings provider instance.

    Returns:
        Initialized FAISS vector store.
    """
    if not chunks:
        raise ValueError("Cannot create vector store from empty document list.")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    return vector_store


def save_vector_store(
    vector_store: FAISS,
    folder_path: str = "vector_db",
    index_name: str = "index"
) -> str:
    """
    Persists the FAISS index files (index.faiss and index.pkl) to disk.

    Args:
        vector_store: Active FAISS store.
        folder_path: Destination folder.
        index_name: Base filename.

    Returns:
        Absolute path to saved vector database directory.
    """
    os.makedirs(folder_path, exist_ok=True)
    vector_store.save_local(folder_path=folder_path, index_name=index_name)
    return os.path.abspath(folder_path)


def load_vector_store(
    embedding_model: Embeddings,
    folder_path: str = "vector_db",
    index_name: str = "index"
) -> Optional[FAISS]:
    """
    Loads a persisted FAISS vector store from disk.

    Args:
        embedding_model: Embeddings provider used during creation.
        folder_path: Source folder where index is saved.
        index_name: Base filename.

    Returns:
        Loaded FAISS instance or None if not found.
    """
    faiss_file = os.path.join(folder_path, f"{index_name}.faiss")
    pkl_file = os.path.join(folder_path, f"{index_name}.pkl")

    if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
        return None

    try:
        vector_store = FAISS.load_local(
            folder_path=folder_path,
            embeddings=embedding_model,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        print(f"Error loading vector store from {folder_path}: {e}")
        return None


def add_documents_to_store(
    vector_store: FAISS,
    new_chunks: List[Document]
) -> FAISS:
    """Incrementally adds new chunks to an existing FAISS vector store."""
    if new_chunks:
        vector_store.add_documents(new_chunks)
    return vector_store


def get_indexed_sources(vector_store: FAISS) -> List[str]:
    """Extracts a sorted list of unique document filenames present in the vector store."""
    sources: Set[str] = set()
    try:
        docstore = getattr(vector_store, "docstore", None)
        if docstore and hasattr(docstore, "_dict"):
            for doc in docstore._dict.values():
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
    except Exception as e:
        print(f"Error extracting indexed sources: {e}")

    return sorted(list(sources))
