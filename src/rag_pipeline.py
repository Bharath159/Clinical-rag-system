from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Literal

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_EMBEDDING_MODEL,
)
from .llm_providers import ExtractiveProvider, get_llm_provider
from .models import RetrievedChunk

# Expose RetrievedChunk for external imports
__all__ = ["ClinicalRAG", "RetrievedChunk"]


class ClinicalRAG:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_provider: Literal["openai", "ollama", "huggingface", "extractive"] = "openai",
        **llm_kwargs: Any,
    ) -> None:
        load_dotenv()
        self.data_dir = Path(data_dir)
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize LLM provider
        try:
            self.llm = get_llm_provider(provider=llm_provider, **llm_kwargs)
        except Exception as e:
            print(f"Warning: Could not initialize {llm_provider} provider ({e}), falling back to extractive.")
            self.llm = ExtractiveProvider()

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
        overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    ) -> list[str]:
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        step = max(1, chunk_size_words - overlap_words)
        for start_idx in range(0, len(words), step):
            piece = words[start_idx : start_idx + chunk_size_words]
            if not piece:
                continue
            chunks.append(" ".join(piece))
            if start_idx + chunk_size_words >= len(words):
                break
        return chunks

    def _build_docs(
        self,
        chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
        overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    ) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []

        for file_path in sorted(self.data_dir.glob("*.txt")):
            text = file_path.read_text(encoding="utf8")
            chunks = self.chunk_text(
                text=text,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
            )
            for chunk_idx, chunk in enumerate(chunks):
                docs.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                        "meta": {"source": file_path.name, "chunk": chunk_idx},
                    }
                )

        return docs

    def ingest(
        self,
        clear_existing: bool = False,
        chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
        overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
        batch_size: int = 64,
    ) -> dict[str, int]:
        if clear_existing:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)

        docs = self._build_docs(
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

        if not docs:
            return {"files": 0, "chunks": 0}

        texts = [doc["text"] for doc in docs]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

        for start in range(0, len(docs), batch_size):
            stop = start + batch_size
            window = docs[start:stop]
            self.collection.add(
                documents=[d["text"] for d in window],
                metadatas=[d["meta"] for d in window],
                ids=[d["id"] for d in window],
                embeddings=embeddings[start:stop].tolist(),
            )

        return {
            "files": len(list(self.data_dir.glob("*.txt"))),
            "chunks": len(docs),
        }

    def has_index(self) -> bool:
        return self.collection.count() > 0

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query to improve retrieval for medical concepts.
        Expands with related terms and provides more context.
        """
        # Convert to lowercase for pattern matching
        query_lower = query.lower()
        
        # Medical terminology expansions
        expansions = {
            "insulin secretion": "insulin secretion physiology pancreatic beta cells islets",
            "dka": "diabetic ketoacidosis DKA",
            "hypertension": "hypertension blood pressure HTN",
            "sepsis": "sepsis infection SIRS systemic inflammatory",
            "pneumonia": "pneumonia CAP community-acquired",
            "anticoagulation": "anticoagulation warfarin DOACs blood thinners",
        }
        
        # Check if query contains any expandable terms
        for term, expansion in expansions.items():
            if term in query_lower:
                return f"{query} {expansion}"
        
        return query

    def _resolve_conversation_context(self, query: str, chat_history: list[dict[str, str]] | None = None) -> str:
        """
        Resolve pronouns and vague references using chat history.
        Handles cases like: "what are the risks of it?" → "what are the risks of hypertension?"
        
        Args:
            query: Current user question
            chat_history: List of previous messages with 'role' and 'content' keys
            
        Returns:
            Enhanced query with pronouns resolved
        """
        if not chat_history or len(chat_history) == 0:
            return query
        
        query_lower = query.lower()
        
        # List of pronouns and vague references to resolve
        pronoun_patterns = [
            "it ", "its ", "that ", "this ", "these ", "those ",
            "the risks", "the benefits", "the management", "the treatment",
            "the diagnosis", "the causes", "the symptoms", "it?"
        ]
        
        # Check if current query contains any pronouns/vague references
        has_pronoun = any(pattern in query_lower for pattern in pronoun_patterns)
        
        if not has_pronoun:
            return query
        
        # Extract the assistant's last response (should be the topic)
        last_assistant_msg = None
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg.get("content", "")
                break
        
        if not last_assistant_msg:
            return query
        
        # Extract main topic from last assistant message (first meaningful noun/word)
        # Look for common medical terms in the response
        medical_terms = [
            "hypertension", "insulin", "sepsis", "pneumonia", "diabetes", "dka",
            "blood pressure", "anticoagulation", "tuberculosis", "covid", "resuscitation"
        ]
        
        topic = None
        for term in medical_terms:
            if term.lower() in last_assistant_msg.lower():
                topic = term
                break
        
        # If no specific medical term found, try to extract from first sentence
        if not topic:
            first_sentence = last_assistant_msg.split('.')[0] if '.' in last_assistant_msg else last_assistant_msg[:100]
            # Extract the main subject (usually after "is" or "refers to")
            if " is " in first_sentence:
                parts = first_sentence.split(" is ", 1)
                topic = parts[1].split(" a " if " a " in parts[1] else ",")[0].strip()
        
        # Replace pronouns with the resolved topic
        if topic:
            enhanced_query = query
            for pattern in pronoun_patterns:
                if pattern.lower() in enhanced_query.lower():
                    enhanced_query = enhanced_query.replace(pattern, f"{topic} ")
                    enhanced_query = enhanced_query.replace(pattern.lower(), f"{topic} ")
            return enhanced_query
        
        return query

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # Preprocess query for better matching
        processed_query = self._preprocess_query(query)
        query_embedding = self.embedding_model.encode([processed_query], convert_to_numpy=True)[0].tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for document, meta, distance in zip(docs, metas, distances):
            source = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
            chunk_index = int(meta.get("chunk", -1)) if isinstance(meta, dict) else -1
            retrieved.append(
                RetrievedChunk(
                    text=document,
                    source=source,
                    chunk_index=chunk_index,
                    score=float(distance),
                )
            )

        return retrieved

    def answer_question(self, question: str, top_k: int = 5, chat_history: list[dict[str, str]] | None = None) -> tuple[str, list[RetrievedChunk]]:
        # Resolve pronouns from chat context
        enhanced_question = self._resolve_conversation_context(question, chat_history)
        
        # Retrieve with enhanced question
        contexts = self.retrieve(query=enhanced_question, top_k=top_k)
        try:
            answer = self.llm.generate_answer(question=question, contexts=contexts)
        except Exception as e:
            fallback_provider = ExtractiveProvider()
            answer = fallback_provider.generate_answer(question=question, contexts=contexts)
            print(f"LLM generation failed ({e}), using extractive fallback.")
        return answer, contexts
