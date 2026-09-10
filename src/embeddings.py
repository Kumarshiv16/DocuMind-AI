"""
Embeddings Module
Provides a unified factory to initialize vector embedding models.
Supports:
1. Local HuggingFace embeddings ('sentence-transformers/all-MiniLM-L6-v2') - Free, runs locally, no API key.
2. Google Gemini embeddings ('models/text-embedding-004') - Fast cloud embeddings.
3. OpenAI embeddings ('text-embedding-3-small') - High performance cloud embeddings.
"""

import os
from typing import Optional
from langchain_core.embeddings import Embeddings


def get_embedding_model(
    provider: str = "local",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Embeddings:
    """
    Returns an embedding instance based on the chosen provider.

    Args:
        provider: 'local' (HuggingFace), 'gemini' (Google), or 'openai'
        api_key: Optional API key for cloud providers
        model_name: Optional explicit model override
    """
    provider_clean = (provider or "local").strip().lower()

    if provider_clean in ["gemini", "google"]:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google API Key is required for Gemini embeddings.")
        model = model_name or "models/text-embedding-004"
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=key)

    elif provider_clean == "openai":
        from langchain_openai import OpenAIEmbeddings
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key is required for OpenAI embeddings.")
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbeddings(model=model, openai_api_key=key)

    else:
        # Default: Local HuggingFace sentence-transformers
        # Runs offline on CPU, completely free
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
