# 🤖 DocuMind AI: Intelligent Multi-Document RAG Chatbot

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-purple.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **DocuMind AI** is an enterprise-grade Retrieval-Augmented Generation (RAG) assistant that empowers users to chat with multiple documents (PDFs, Word documents, text notes) using semantic search, exact page citations, conversational context memory, and strict anti-hallucination guardrails.

---

## 📌 Architecture Diagram

```
                       [ 📄 User Documents (PDF / DOCX / TXT) ]
                                          │
                                          ▼
                         1. Document Loader (pypdf, python-docx)
                         Extracts text & metadata (source, page)
                                          │
                                          ▼
                      2. Text Splitter (RecursiveCharacterSplitter)
                         Chunks: 1000 chars | Overlap: 200 chars
                                          │
                                          ▼
                       3. Embedding Model (HuggingFace / Gemini)
                         Dense Vector Representations (e.g. 384-d)
                                          │
                                          ▼
                        4. Vector Database (FAISS Index in vector_db/)
                         Fast L2 / Cosine Similarity Index
                                          │
  [ 💬 User Query ]                       │
         │                                │
         ▼                                │
  5. Conversational Memory                │
     Rephrases pronouns ("it", "they")    │
     into standalone search query         │
         │                                │
         ▼                                ▼
  6. Context Retriever ───────────────────┘
     Top-k Semantic Matching + Metadata Filter
         │
         ▼
  7. Strict Guardrail Prompt
     "Answer ONLY from context. If not found:
     'I couldn't find this information in the uploaded documents.'"
         │
         ▼
  8. Large Language Model (Gemini / Groq / OpenAI)
         │
         ▼
  9. Final Answer + Collapsible Exact Page Citations
         │
         ▼
  [ 🖥️ Streamlit Interactive UI ]
```

---

## ⭐ Key Features

1. **Multi-Document Ingestion**:
   - Upload multiple PDFs, Word documents (`.docx`), or text files simultaneously.
   - Preserves page numbers and document filenames for auditability.

2. **Semantic Search with FAISS**:
   - Outperforms naive keyword matching by searching semantic meaning in dense vector space.
   - Defaults to local HuggingFace embeddings (`all-MiniLM-L6-v2`) which run 100% free and offline on your CPU.
   - Cloud embeddings (Google Gemini, OpenAI) also supported.

3. **Source-Based Answers & Page Citations**:
   - Displays exact file names, page numbers, and similarity confidence scores for each fact.
   - Collapsible citation cards in the UI prevent clutter.

4. **Context-Aware Conversational Memory**:
   - Remembers previous turns. If the user asks *"What is supervised learning?"* followed by *"Give me an example of it"*, the system automatically reformulates the follow-up to *"Give me an example of supervised learning"* before querying the vector store.

5. **Document-Level Filtering**:
   - Switch between querying across **All Documents** or restricting searches to an individual document (e.g., *syllabus.pdf*).

6. **Anti-Hallucination Guardrails**:
   - If information is not in the uploaded documents, the system refuses to extrapolate or invent facts and responds:
     > *"I couldn't find this information in the uploaded documents."*

7. **Export & Chat Management**:
   - Export full conversation transcripts to JSON.
   - One-click conversation reset.

---

## 📂 Project Structure

```
DocuMind-AI/
│
├── app.py                      # Streamlit UI & interaction engine
├── requirements.txt            # Pinned dependencies
├── .env.example                # Template for API keys
├── README.md                   # Comprehensive documentation & interview guide
│
├── src/                        # Modular RAG architecture
│   ├── __init__.py
│   ├── document_loader.py      # PDF, DOCX, TXT parsers with page metadata
│   ├── text_splitter.py        # RecursiveCharacterTextSplitter with chunk IDs
│   ├── embeddings.py           # HuggingFace, Gemini, and OpenAI embedding factory
│   ├── vector_store.py         # FAISS index persistence, loading & incremental indexing
│   ├── retriever.py            # Similarity search, doc filtering & citation builder
│   └── chatbot.py              # LLM factory, memory reformulation & hallucination prompt
│
├── data/
│   └── documents/              # Sample or uploaded raw files
│       └── sample_ai_handbook.txt
│
├── tests/                      # Automated pipeline tests
│   └── test_rag_pipeline.py
│
└── vector_db/                  # Persisted FAISS index files (index.faiss, index.pkl)
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### 2. Setup Virtual Environment
```powershell
# Clone or navigate to the project directory
cd "DocuMind AI"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure API Keys
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Add your free API key for Google Gemini or Groq:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
# Or:
GROQ_API_KEY=your_groq_api_key_here
```
*(Note: You can also enter the API key directly into the Streamlit sidebar at runtime).*

### 5. Run the Application
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser!

---

## 🧪 Running Automated Tests

Run the unit and integration tests covering loader, splitter, embeddings, FAISS, and retriever:
```powershell
python -m unittest tests/test_rag_pipeline.py -v
```

---

## 🎯 Technical Deep-Dive: Common Interview Questions

### 1. Why do we chunk documents with overlap?
- **Without overlap**: If a crucial sentence or definition is split at character boundary 1000, half the sentence lands in Chunk $A$ and the other half in Chunk $B$. Both chunks lose the complete thought, resulting in poor retrieval scores.
- **With overlap** (e.g., 200 characters): The boundary context is preserved in both contiguous chunks, ensuring complete semantic integrity.

### 2. How does FAISS perform vector similarity search?
- FAISS (Facebook AI Similarity Search) maps dense vector embeddings into an indexing structure.
- In `IndexFlatL2`, it calculates Euclidean distance:
  $$d(u, v) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$$
- When embeddings are normalized ($\|u\| = 1$), Euclidean distance is monotonically related to **Cosine Similarity**:
  $$\text{Cosine Similarity} = \frac{u \cdot v}{\|u\| \|v\|}$$
- FAISS enables sub-millisecond retrieval even over millions of vectors.

### 3. How does conversational memory work in RAG?
- Naive RAG fails on follow-up questions like *"Explain the second one"* because the vector search looks for the words *"Explain the second one"*, which matches irrelevant text.
- **DocuMind AI's approach**: We use a two-step RAG chain:
  1. **Query Reformulation**: The LLM takes the recent conversation history + the latest question and produces a standalone search query (e.g., *"Explain Support Vector Machines in supervised learning"*).
  2. **Retrieval & Answer**: Vector retrieval searches the standalone query, guaranteeing that relevant context is fetched.

### 4. How do you mitigate hallucinations?
- **Context Grounding**: System prompt strictly restricts the LLM to facts explicitly present in the retrieved context.
- **Strict Fallback Prompting**: If facts are missing, the prompt forces the model to respond *"I couldn't find this information in the uploaded documents."*
- **Low Temperature**: Default temperature is set to $0.1 - 0.2$ to minimize stochastic creative completion.
- **Source Auditing**: All responses are paired with visible source citations and page numbers so users can verify facts.

### 5. When should you use RAG vs Fine-Tuning?
| Feature | RAG | Fine-Tuning |
| :--- | :--- | :--- |
| **Primary Purpose** | Knowledge access & fact retrieval | Style, tone, format, domain syntax |
| **Data Freshness** | Instant (add file to vector DB) | High latency (requires retraining) |
| **Cost** | Low (cheap vector storage) | High (GPU compute required) |
| **Hallucination** | Low (answers anchored in context) | Moderate-High (can memorize false facts) |
| **Citations** | Direct page & document citations | Black-box weights (no citations) |

---

## 💼 Resume & LinkedIn Highlights

### Resume Description
```text
DocuMind AI — AI-Powered Multi-Document RAG Chatbot
Python | LangChain | FAISS | NLP | Streamlit | Google Gemini / Groq

- Architected and built an end-to-end Retrieval-Augmented Generation (RAG) chatbot enabling conversational QA across multi-format documents (PDF, DOCX, TXT).
- Designed an automated ingestion pipeline using RecursiveCharacterTextSplitter with sliding overlap, HuggingFace embeddings (all-MiniLM-L6-v2), and FAISS vector indexing with disk persistence.
- Implemented context-aware conversational memory using LLM-based query reformulation, resolving ambiguous pronouns across multi-turn chats.
- Engineered anti-hallucination guardrails and source citation tracking, returning document names, page numbers, and similarity metrics for every response.
- Delivered an interactive Streamlit UI featuring dynamic document filtering, real-time chunk statistics, and JSON transcript exports.
```

### LinkedIn Post / Project Section
```text
🚀 Built DocuMind AI: An Enterprise Multi-Document RAG Chatbot!

Rather than relying on generic LLM knowledge, DocuMind AI allows users to chat directly with private course notes, research papers, and company documentation.

Key Architectural Highlights:
🔹 Multi-Format Ingestion: Parses PDFs, Word docs, and text files with page-level metadata tracking.
🔹 Semantic Search: Uses FAISS vector indexing with dense embeddings to find semantic meaning beyond keyword matching.
🔹 Conversational Memory: Automatically reformulates follow-up queries using chat history context.
🔹 Anti-Hallucination Guardrails: Prevents false answers when information is absent from documents.
🔹 Transparent Citations: Displays exact document sources, page numbers, and similarity scores.

Tech Stack: Python, LangChain, FAISS, Sentence-Transformers, Streamlit, Google Gemini / Groq / OpenAI.
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
