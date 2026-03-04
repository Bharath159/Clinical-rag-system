from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Literal

import requests
from openai import OpenAI

from .models import RetrievedChunk


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate_answer(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """Generate an answer based on question and retrieved contexts."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI-based answer generation using Chat Completions API."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided and not found in environment")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key)

    def generate_answer(self, question: str, contexts: list[RetrievedChunk]) -> str:
        context_block = "\n\n".join(
            [
                f"[Source: {chunk.source} | Chunk: {chunk.chunk_index}]\n{chunk.text}"
                for chunk in contexts
            ]
        )

        prompt = (
            "You are a clinical QA assistant specializing in evidence-based medical guidelines.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer ONLY using information from the retrieved context below\n"
            "2. If the context contains relevant information, provide a comprehensive, well-structured answer\n"
            "3. If the answer is not in the context, clearly state: 'The provided context does not contain information about [topic]'\n"
            "4. Use clinical terminology appropriately and explain complex concepts clearly\n"
            "5. Cite specific sources when possible (e.g., 'According to guideline_insulin.txt...')\n"
            "6. This is for educational purposes only - not for diagnosing individual patients\n\n"
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED CONTEXT:\n{context_block}\n\n"
            "ANSWER:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()


class OllamaProvider(LLMProvider):
    """Ollama-based local LLM generation (for open source models like Llama2, Mistral, etc)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral") -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "mistral")

    def generate_answer(self, question: str, contexts: list[RetrievedChunk]) -> str:
        context_block = "\n\n".join(
            [
                f"[Source: {chunk.source} | Chunk: {chunk.chunk_index}]\n{chunk.text}"
                for chunk in contexts
            ]
        )

        prompt = (
            "You are a clinical QA assistant specializing in evidence-based medical guidelines.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer ONLY using information from the retrieved context below\n"
            "2. If the context contains relevant information, provide a comprehensive, well-structured answer\n"
            "3. If the answer is not in the context, clearly state: 'The provided context does not contain information about [topic]'\n"
            "4. Use clinical terminology appropriately and explain complex concepts clearly\n"
            "5. Cite specific sources when possible (e.g., 'According to guideline_insulin.txt...')\n"
            "6. This is for educational purposes only - not for diagnosing individual patients\n\n"
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED CONTEXT:\n{context_block}\n\n"
            "ANSWER:"
        )

        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API provider for open source models."""

    def __init__(self, api_key: str | None = None, model: str = "mistralai/Mistral-7B-Instruct-v0.1") -> None:
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY not provided and not found in environment")
        self.model = model or os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

    def generate_answer(self, question: str, contexts: list[RetrievedChunk]) -> str:
        context_block = "\n\n".join(
            [
                f"[Source: {chunk.source} | Chunk: {chunk.chunk_index}]\n{chunk.text}"
                for chunk in contexts
            ]
        )

        prompt = (
            "You are a clinical QA assistant specializing in evidence-based medical guidelines.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer ONLY using information from the retrieved context below\n"
            "2. If the context contains relevant information, provide a comprehensive, well-structured answer\n"
            "3. If the answer is not in the context, clearly state: 'The provided context does not contain information about [topic]'\n"
            "4. Use clinical terminology appropriately and explain complex concepts clearly\n"
            "5. Cite specific sources when possible (e.g., 'According to guideline_insulin.txt...')\n"
            "6. This is for educational purposes only - not for diagnosing individual patients\n\n"
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED CONTEXT:\n{context_block}\n\n"
            "ANSWER:"
        )

        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": prompt, "parameters": {"temperature": 0.1, "max_length": 512}}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            return str(result).strip()
        except Exception as e:
            raise RuntimeError(f"HuggingFace Inference API request failed: {e}")


class ExtractiveProvider(LLMProvider):
    """Fallback extractive provider (no LLM, just returns top context)."""

    @staticmethod
    def generate_answer(question: str, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return "I couldn't find relevant context in the indexed documents."

        top = contexts[0]
        excerpt = top.text[:1200]
        return (
            "No LLM provider configured. Here's an extractive answer from the most relevant chunk:\n\n"
            f"Question: {question}\n\n"
            f"Best matching source: {top.source} (chunk {top.chunk_index})\n"
            f"Relevant excerpt:\n{excerpt}"
        )


def get_llm_provider(
    provider: Literal["openai", "ollama", "huggingface", "extractive"] = "openai",
    **kwargs,
) -> LLMProvider:
    """
    Factory function to get an LLM provider by name.
    
    Args:
        provider: One of "openai", "ollama", "huggingface", "extractive"
        **kwargs: Provider-specific arguments (api_key, model, base_url, etc.)
    
    Returns:
        LLMProvider instance
    
    Raises:
        ValueError: If provider is unknown or required credentials are missing
    """
    if provider == "openai":
        return OpenAIProvider(**kwargs)
    elif provider == "ollama":
        return OllamaProvider(**kwargs)
    elif provider == "huggingface":
        return HuggingFaceProvider(**kwargs)
    elif provider == "extractive":
        return ExtractiveProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
