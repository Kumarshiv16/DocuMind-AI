"""
DocuMind AI - Modern Multi-Document RAG Web Application
Features:
- Current Google GenAI SDK (google-genai Client & models.generate_content)
- Multi-format document ingestion (PDF, DOCX, TXT)
- Semantic Vector Search with FAISS & local HuggingFace embeddings
- Source & page citations
- Anti-hallucination strict grounded mode
- Conversational memory query reformulation
- Real-time RAG performance analytics (latency, similarity, context tokens)
- Professional responsive Dark UI
- Direct launch support: 'python app.py' or 'streamlit run app.py'
"""

import os
import sys
import json
import time
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

from src.document_loader import load_uploaded_files, load_directory
from src.text_splitter import split_documents, get_chunk_statistics
from src.embeddings import get_embedding_model
from src.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    get_indexed_sources
)
from src.retriever import ContextRetriever
from src.chatbot import RAGChatbot

# Streamlit Page Setup
st.set_page_config(
    page_title="DocuMind AI | Multi-Doc RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Responsive Dark UI Styling
st.markdown("""
<style>
    /* Global dark canvas styling */
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
    }
    
    /* Header typography */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.4rem;
    }

    /* Citation cards */
    .citation-card {
        background-color: #1E293B;
        border-left: 4px solid #38BDF8;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .citation-header {
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 4px;
    }
    
    /* Groundedness badges */
    .badge-grounded {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-ungrounded {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Analytics Metric Pill */
    .analytics-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 0.8rem;
        color: #94A3B8;
    }
    .analytics-val {
        color: #38BDF8;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "chunk_stats" not in st.session_state:
    st.session_state.chunk_stats = {}


def get_secret(key_name: str, fallback_key: str = "", default: str = "") -> str:
    """Safely retrieves secrets from os.environ or st.secrets (for Streamlit Community Cloud)."""
    val = os.getenv(key_name)
    if val:
        return val
    if fallback_key and os.getenv(fallback_key):
        return os.getenv(fallback_key)
    try:
        if hasattr(st, "secrets"):
            if key_name in st.secrets:
                return str(st.secrets[key_name])
            if fallback_key and fallback_key in st.secrets:
                return str(st.secrets[fallback_key])
    except Exception:
        pass
    return default


# Sidebar Controls
with st.sidebar:
    st.markdown("### 🤖 **DocuMind AI**")
    st.caption("Intelligent Multi-Doc RAG Chatbot")
    st.markdown("---")

    # 1. Gemini / LLM Model Configuration
    st.subheader("⚡ 1. LLM Engine")
    llm_provider = st.selectbox(
        "Provider",
        options=["Google Gemini", "Groq (Llama 3)", "OpenAI"],
        index=0
    )

    prov_code = "gemini"
    default_model = get_secret("GEMINI_MODEL", default="gemini-3.8-flash")
    api_key_env = get_secret("GEMINI_API_KEY", fallback_key="GOOGLE_API_KEY")

    if llm_provider == "Groq (Llama 3)":
        prov_code = "groq"
        default_model = get_secret("GROQ_MODEL", default="llama-3.1-8b-instant")
        api_key_env = get_secret("GROQ_API_KEY")
    elif llm_provider == "OpenAI":
        prov_code = "openai"
        default_model = get_secret("OPENAI_MODEL", default="gpt-4o-mini")
        api_key_env = get_secret("OPENAI_API_KEY")

    model_input = st.text_input("Model Name", value=default_model)
    api_key_input = st.text_input(
        f"{llm_provider} API Key",
        value=api_key_env,
        type="password",
        help="Loaded automatically from .env or Streamlit Secrets. Keep API keys secure!"
    )

    # Embedding model selection
    embedding_choice = st.selectbox(
        "Embeddings",
        options=[
            "Local: all-MiniLM-L6-v2 (100% Free & Offline)",
            "Google: text-embedding-004",
            "OpenAI: text-embedding-3-small"
        ],
        index=0,
        help="Local embeddings run on CPU without consuming API quotas."
    )

    embed_provider = "local"
    if "Google" in embedding_choice:
        embed_provider = "gemini"
    elif "OpenAI" in embedding_choice:
        embed_provider = "openai"

    st.markdown("---")

    # 2. Multi-Document Upload
    st.subheader("📁 2. Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload course notes, syllabi, manuals, research papers, etc."
    )

    col1, col2 = st.columns(2)
    with col1:
        load_sample = st.button("📚 Load Sample Doc", use_container_width=True)
    with col2:
        process_btn = st.button("⚡ Index Docs", type="primary", use_container_width=True)

    if load_sample:
        with st.spinner("Indexing sample AI & ML handbook..."):
            try:
                embed_model = get_embedding_model(provider=embed_provider, api_key=api_key_input)
                sample_docs = load_directory("data/documents")
                if sample_docs:
                    chunks = split_documents(sample_docs, chunk_size=900, chunk_overlap=150)
                    vs = create_vector_store(chunks, embed_model)
                    save_vector_store(vs, folder_path="vector_db")
                    st.session_state.vector_store = vs
                    st.session_state.indexed_files = get_indexed_sources(vs)
                    st.session_state.chunk_stats = get_chunk_statistics(chunks)
                    st.success(f"Indexed {len(chunks)} chunks from sample doc!")
                else:
                    st.warning("data/documents folder is empty.")
            except Exception as e:
                st.error(f"Error loading sample: {str(e)}")

    if process_btn and uploaded_files:
        with st.spinner("Parsing documents -> Chunking -> Vectorizing with FAISS..."):
            try:
                embed_model = get_embedding_model(provider=embed_provider, api_key=api_key_input)
                raw_docs = load_uploaded_files(uploaded_files)
                if not raw_docs:
                    st.warning("No readable text found in uploaded files.")
                else:
                    chunks = split_documents(raw_docs, chunk_size=1000, chunk_overlap=200)
                    vs = create_vector_store(chunks, embed_model)
                    save_vector_store(vs, folder_path="vector_db")
                    st.session_state.vector_store = vs
                    st.session_state.indexed_files = get_indexed_sources(vs)
                    st.session_state.chunk_stats = get_chunk_statistics(chunks)
                    st.success(f"Indexed {len(uploaded_files)} file(s) into {len(chunks)} chunks!")
            except Exception as e:
                st.error(f"Ingestion failed: {str(e)}")

    # Display Active Index Info
    if st.session_state.vector_store is not None:
        st.markdown("---")
        st.subheader("📊 Vector DB Status")
        st.write(f"**Indexed Chunks:** `{st.session_state.chunk_stats.get('total_chunks', 0)}`")
        st.write(f"**Documents ({len(st.session_state.indexed_files)}):**")
        for f in st.session_state.indexed_files:
            st.markdown(f"- 📄 `{f}`")

    st.markdown("---")

    # 3. RAG Settings & Anti-Hallucination Mode
    st.subheader("🎯 3. Retrieval & Guardrails")
    strict_mode = st.toggle(
        "🛡️ Strict Grounded Mode",
        value=True,
        help="When enabled, prevents hallucination by requiring answers to strictly exist in documents."
    )

    filter_options = ["All Documents"] + st.session_state.indexed_files
    selected_filter = st.selectbox(
        "Document Filter",
        options=filter_options,
        index=0,
        help="Limit queries to a specific uploaded document."
    )

    top_k = st.slider("Top Chunks (k)", min_value=1, max_value=8, value=4)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    st.markdown("---")

    # Chat Actions
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with c2:
        chat_json = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            "💾 Export JSON",
            data=chat_json,
            file_name="chat_transcript.json",
            mime="application/json",
            use_container_width=True
        )


# Main Interface Area
st.markdown('<div class="hero-title">DocuMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Multi-Document RAG with Google GenAI SDK, Semantic Search, Exact Page Citations & Anti-Hallucination Guardrails.</div>',
    unsafe_allow_html=True
)

# Auto-load existing vector_db if present on disk
if st.session_state.vector_store is None and os.path.exists("vector_db/index.faiss"):
    try:
        embed_model = get_embedding_model(provider=embed_provider, api_key=api_key_input)
        loaded_vs = load_vector_store(embedding_model=embed_model, folder_path="vector_db")
        if loaded_vs:
            st.session_state.vector_store = loaded_vs
            st.session_state.indexed_files = get_indexed_sources(loaded_vs)
    except Exception:
        pass


# Quick Start Suggestions
if not st.session_state.messages:
    st.info("💡 **Quick Start:** Upload documents in the sidebar or click **'📚 Load Sample Doc'** to immediately query the AI handbook!")
    st.markdown("**Example Questions to Ask:**")
    example_questions = [
        "What are the core stages in a RAG pipeline?",
        "What is the difference between supervised and unsupervised learning?",
        "What evaluation metrics are used in RAG?",
        "What does the document say about hallucination mitigation?"
    ]
    p_cols = st.columns(len(example_questions))
    for i, eq in enumerate(example_questions):
        if p_cols[i].button(f"👉 {eq}", key=f"ex_{i}"):
            st.session_state.pending_query = eq
            st.rerun()

# Check for pending query from button clicks
if "pending_query" in st.session_state and st.session_state.pending_query:
    active_prompt = st.session_state.pending_query
    del st.session_state["pending_query"]
else:
    active_prompt = None


# Display Chat History
for msg in st.session_state.messages:
    role = msg.get("role")
    with st.chat_message(role):
        st.markdown(msg.get("content", ""))

        if role == "assistant":
            is_grounded = msg.get("is_grounded", True)
            if is_grounded:
                st.markdown('<div class="badge-grounded">🛡️ Grounded in Uploaded Documents</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-ungrounded">✕ Out of Scope / Not in Documents</div>', unsafe_allow_html=True)

            # RAG Analytics Bar
            analytics = msg.get("analytics")
            if analytics:
                c_a, c_b, c_c, c_d = st.columns(4)
                with c_a:
                    st.caption(f"⏱️ **Latency:** `{analytics.get('total_latency_ms', 0)}ms`")
                with c_b:
                    st.caption(f"🔎 **Retrieval:** `{analytics.get('retrieval_latency_ms', 0)}ms`")
                with c_c:
                    st.caption(f"📄 **Chunks:** `{analytics.get('chunks_retrieved', 0)}`")
                with c_d:
                    st.caption(f"🎯 **Top Score:** `{analytics.get('top_similarity_score', 0.0)}`")

            # Conversational Query Reformulation Expander
            standalone = msg.get("standalone_query")
            if standalone and standalone != msg.get("original_query"):
                with st.expander("🧠 Conversational Context Query"):
                    st.caption(f"**Standalone search query:** `{standalone}`")

            # Source Citations Cards
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 View {len(sources)} Source Citations & References"):
                    for s_idx, src in enumerate(sources, 1):
                        st.markdown(
                            f"""<div class="citation-card">
                            <div class="citation-header">Source {s_idx}: 📄 {src.get('source')} (Page {src.get('page')})</div>
                            <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:4px;">Similarity Distance: {src.get('score')}</div>
                            <div>"{src.get('snippet')}"</div>
                            </div>""",
                            unsafe_allow_html=True
                        )


# Chat Input
user_input = st.chat_input("Ask a question about your uploaded documents...") or active_prompt

if user_input:
    # 1. Guard check: Vector store indexed?
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload documents or click **'📚 Load Sample Doc'** in the sidebar first!")
        st.stop()

    # 2. Guard check: API key provided?
    if not api_key_input:
        st.error(f"⚠️ Please enter your {llm_provider} API Key in the sidebar or set GEMINI_API_KEY in your .env file.")
        st.stop()

    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving document context & generating grounded answer..."):
            try:
                retriever = ContextRetriever(
                    vector_store=st.session_state.vector_store,
                    default_k=top_k
                )

                chatbot = RAGChatbot(
                    retriever=retriever,
                    provider=prov_code,
                    api_key=api_key_input,
                    model_name=model_input,
                    temperature=temperature,
                    strict_grounded=strict_mode
                )

                # Execute RAG pipeline
                result = chatbot.answer_question(
                    query=user_input,
                    chat_history=st.session_state.messages[:-1],
                    k=top_k,
                    filter_doc=selected_filter
                )

                answer_text = result["answer"]
                sources = result.get("sources", [])
                is_grounded = result.get("is_grounded", True)
                standalone_query = result.get("standalone_query", user_input)
                analytics = result.get("analytics", {})

                # Render response
                st.markdown(answer_text)

                if is_grounded:
                    st.markdown('<div class="badge-grounded">🛡️ Grounded in Uploaded Documents</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-ungrounded">✕ Out of Scope / Not in Documents</div>', unsafe_allow_html=True)

                # Analytics Bar
                if analytics:
                    c_a, c_b, c_c, c_d = st.columns(4)
                    with c_a:
                        st.caption(f"⏱️ **Latency:** `{analytics.get('total_latency_ms', 0)}ms`")
                    with c_b:
                        st.caption(f"🔎 **Retrieval:** `{analytics.get('retrieval_latency_ms', 0)}ms`")
                    with c_c:
                        st.caption(f"📄 **Chunks:** `{analytics.get('chunks_retrieved', 0)}`")
                    with c_d:
                        st.caption(f"🎯 **Top Score:** `{analytics.get('top_similarity_score', 0.0)}`")

                if standalone_query != user_input:
                    with st.expander("🧠 Conversational Context Query"):
                        st.caption(f"**Standalone search query:** `{standalone_query}`")

                if sources:
                    with st.expander(f"📚 View {len(sources)} Source Citations & References"):
                        for s_idx, src in enumerate(sources, 1):
                            st.markdown(
                                f"""<div class="citation-card">
                                <div class="citation-header">Source {s_idx}: 📄 {src.get('source')} (Page {src.get('page')})</div>
                                <div style="font-size:0.8rem; color:#94A3B8; margin-bottom:4px;">Similarity Distance: {src.get('score')}</div>
                                <div>"{src.get('snippet')}"</div>
                                </div>""",
                                unsafe_allow_html=True
                            )

                # Append assistant message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                    "is_grounded": is_grounded,
                    "original_query": user_input,
                    "standalone_query": standalone_query,
                    "analytics": analytics
                })

            except Exception as e:
                st.error(f"Error during RAG execution: {str(e)}")


# Support direct terminal invocation via 'python app.py'
if __name__ == "__main__":
    # If invoked directly via 'python app.py' outside streamlit
    if not os.environ.get("STREAMLIT_SERVER_PORT") and "streamlit" not in sys.argv[0]:
        try:
            from streamlit.web import cli as stcli
            sys.argv = ["streamlit", "run", __file__]
            sys.exit(stcli.main())
        except Exception as e:
            print(f"To run with streamlit: streamlit run app.py (Error: {e})")
