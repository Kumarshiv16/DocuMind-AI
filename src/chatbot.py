"""
Chatbot Module (RAG Orchestration Engine)
Implements current Google GenAI SDK (google-genai Client & client.models.generate_content)
alongside LangChain / Groq / OpenAI fallbacks.
Includes conversational memory, strict anti-hallucination guardrails, and citation tracking.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from src.retriever import ContextRetriever

# Strict Anti-Hallucination System Instruction
GROUNDED_SYSTEM_INSTRUCTION = """You are DocuMind AI, an intelligent, professional research assistant that answers questions based STRICTLY on the provided document context.

Guidelines:
1. Answer the user's question using ONLY the factual information provided in the Context below.
2. If the answer cannot be found or directly deduced from the Context, you MUST respond EXACTLY with:
   "I couldn't find this information in the uploaded documents."
3. Do NOT speculate, extrapolate, or use outside knowledge.
4. When stating facts, cite the source document name and page number where available (e.g., [DocName, Page X]).
5. Be concise, precise, structured, and helpful.
"""

RELAXED_SYSTEM_INSTRUCTION = """You are DocuMind AI, a helpful research assistant.
Answer the user question primarily using the provided document context. If some details are missing from the context, you may provide helpful context while explicitly noting what comes from the document vs general knowledge.
"""

REPHRASE_PROMPT_TEMPLATE = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{chat_history}

Follow-up Question: {question}
Standalone Question:"""


class GeminiGenAIClient:
    """Wrapper around the modern google-genai SDK (client.models.generate_content)."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, system_instruction: str, temperature: float = 0.2) -> str:
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return response.text or ""


class RAGChatbot:
    """Orchestrates RAG generation with conversational memory, analytics, and citations."""

    def __init__(
        self,
        retriever: ContextRetriever,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        strict_grounded: bool = True
    ):
        self.retriever = retriever
        self.provider = (provider or "gemini").lower()
        self.temperature = temperature
        self.strict_grounded = strict_grounded

        # Check API Keys
        self.gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.groq_key = api_key or os.getenv("GROQ_API_KEY")
        self.openai_key = api_key or os.getenv("OPENAI_API_KEY")

        # Select Model
        if self.provider in ["gemini", "google"]:
            self.model_name = model_name or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
            if not self.gemini_key:
                raise ValueError("Gemini API Key is required. Set GEMINI_API_KEY in .env or enter it in the sidebar.")
            self.gemini_client = GeminiGenAIClient(api_key=self.gemini_key, model_name=self.model_name)
        elif self.provider == "groq":
            from langchain_groq import ChatGroq
            if not self.groq_key:
                raise ValueError("Groq API Key is required. Set GROQ_API_KEY in .env or enter it in the sidebar.")
            self.model_name = model_name or "llama-3.1-8b-instant"
            self.llm = ChatGroq(model_name=self.model_name, groq_api_key=self.groq_key, temperature=temperature)
        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            if not self.openai_key:
                raise ValueError("OpenAI API Key is required. Set OPENAI_API_KEY in .env or enter it in the sidebar.")
            self.model_name = model_name or "gpt-4o-mini"
            self.llm = ChatOpenAI(model_name=self.model_name, openai_api_key=self.openai_key, temperature=temperature)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def reformulate_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """Reformulates queries with pronouns into standalone search queries using conversational context."""
        if not chat_history:
            return query

        history_lines = []
        for msg in chat_history[-6:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.get('content', '')}")
        history_text = "\n".join(history_lines)

        rephrase_prompt = REPHRASE_PROMPT_TEMPLATE.format(chat_history=history_text, question=query)

        try:
            if self.provider in ["gemini", "google"]:
                reformulated = self.gemini_client.generate(
                    prompt=rephrase_prompt,
                    system_instruction="You reformulate conversational follow-up questions into standalone search queries.",
                    temperature=0.0
                ).strip()
            else:
                resp = self.llm.invoke(rephrase_prompt)
                reformulated = resp.content.strip()

            return reformulated if reformulated else query
        except Exception as e:
            print(f"Error reformulating query: {e}. Using raw query.")
            return query

    def answer_question(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        k: Optional[int] = None,
        filter_doc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline execution with performance metrics.
        """
        start_time = time.time()
        history = chat_history or []

        # 1. Conversational Query Reformulation
        standalone_query = self.reformulate_query(query, history)

        # 2. Semantic Retrieval
        retrieval_start = time.time()
        retrieved_items = self.retriever.retrieve(
            query=standalone_query,
            k=k,
            filter_doc=filter_doc
        )
        retrieval_latency = round((time.time() - retrieval_start) * 1000, 1)

        if not retrieved_items:
            total_latency = round((time.time() - start_time) * 1000, 1)
            return {
                "answer": "I couldn't find this information in the uploaded documents.",
                "sources": [],
                "standalone_query": standalone_query,
                "is_grounded": False,
                "analytics": {
                    "total_latency_ms": total_latency,
                    "retrieval_latency_ms": retrieval_latency,
                    "chunks_retrieved": 0,
                    "top_similarity_score": 0.0,
                    "estimated_context_tokens": 0
                }
            }

        # 3. Context & Source Formatting
        context_str = self.retriever.format_context_string(retrieved_items)
        sources = self.retriever.extract_sources(retrieved_items)
        top_score = sources[0]["score"] if sources else 0.0
        context_tokens = len(context_str) // 4  # rough estimate

        # 4. Construct Prompt & Generate Answer
        system_instruction = GROUNDED_SYSTEM_INSTRUCTION if self.strict_grounded else RELAXED_SYSTEM_INSTRUCTION
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        try:
            if self.provider in ["gemini", "google"]:
                answer_text = self.gemini_client.generate(
                    prompt=user_prompt,
                    system_instruction=system_instruction,
                    temperature=self.temperature
                ).strip()
            else:
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_instruction),
                    ("human", "{question}")
                ])
                resp = self.llm.invoke(prompt.format_messages(question=user_prompt))
                answer_text = resp.content.strip()

        except Exception as e:
            total_latency = round((time.time() - start_time) * 1000, 1)
            return {
                "answer": f"⚠️ Error generating response: {str(e)}",
                "sources": sources,
                "standalone_query": standalone_query,
                "is_grounded": False,
                "analytics": {
                    "total_latency_ms": total_latency,
                    "retrieval_latency_ms": retrieval_latency,
                    "chunks_retrieved": len(retrieved_items),
                    "top_similarity_score": top_score,
                    "estimated_context_tokens": context_tokens
                }
            }

        # 5. Hallucination Detection
        not_found_phrases = [
            "couldn't find this information",
            "could not find this information",
            "not mentioned in the uploaded documents",
            "not found in the uploaded documents"
        ]
        is_grounded = not any(p in answer_text.lower() for p in not_found_phrases)

        total_latency = round((time.time() - start_time) * 1000, 1)

        return {
            "answer": answer_text,
            "sources": sources if is_grounded else [],
            "standalone_query": standalone_query,
            "is_grounded": is_grounded,
            "analytics": {
                "total_latency_ms": total_latency,
                "retrieval_latency_ms": retrieval_latency,
                "chunks_retrieved": len(retrieved_items),
                "top_similarity_score": top_score,
                "estimated_context_tokens": context_tokens
            }
        }
