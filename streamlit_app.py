from __future__ import annotations

import traceback

import streamlit as st

from src.config import DEFAULT_LLM_PROVIDER, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, DEFAULT_TOP_K
from src.rag_pipeline import ClinicalRAG, RetrievedChunk

st.set_page_config(page_title="Clinical RAG Chatbot", page_icon="🩺", layout="wide")


@st.cache_resource
def get_rag(
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    **llm_kwargs,
) -> ClinicalRAG:
    """Get cached RAG instance with specified LLM provider."""
    return ClinicalRAG(llm_provider=llm_provider, **llm_kwargs)


def _format_sources(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No source chunks found."
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        # Calculate relevance (lower distance = more relevant)
        relevance_pct = max(0, 100 - (chunk.score * 100))
        preview = chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
        lines.append(
            f"{idx}. [{chunk.source}] Chunk {chunk.chunk_index} | "
            f"Relevance: {relevance_pct:.1f}% (distance: {chunk.score:.4f})\n"
            f"   Preview: {preview}\n"
        )
    return "\n".join(lines)


def main() -> None:
    st.title("Clinical RAG QA Chatbot")
    st.caption(
        "Asks questions over local clinical guideline text files using retrieval + LLM answer synthesis."
    )

    with st.sidebar:
        st.subheader("⚙️ LLM Configuration")
        provider_options = ["openai", "ollama", "huggingface", "extractive"]
        default_index = provider_options.index(DEFAULT_LLM_PROVIDER) if DEFAULT_LLM_PROVIDER in provider_options else 0
        llm_provider = st.radio(
            "Select LLM Provider:",
            options=provider_options,
            index=default_index,
            help="Choose which LLM to use for answer generation",
        )

        llm_kwargs = {}

        if llm_provider == "openai":
            st.markdown("**OpenAI Settings**")
            openai_key = st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key")
            openai_model = st.selectbox(
                "Model",
                options=["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
                index=0,
            )
            if openai_key:
                llm_kwargs["api_key"] = openai_key
            if openai_model:
                llm_kwargs["model"] = openai_model

        elif llm_provider == "ollama":
            st.markdown("**Ollama Settings** (Local open source models)")
            ollama_url = st.text_input(
                "Ollama Base URL",
                value=DEFAULT_OLLAMA_BASE_URL,
                help="Default: http://localhost:11434",
            )
            ollama_model = st.text_input(
                "Model Name",
                value=DEFAULT_OLLAMA_MODEL,
                help="e.g., mistral, llama2, neural-chat",
            )
            st.caption("ℹ️ Make sure Ollama is running: `ollama serve`")
            if ollama_url:
                llm_kwargs["base_url"] = ollama_url
            if ollama_model:
                llm_kwargs["model"] = ollama_model

        elif llm_provider == "huggingface":
            st.markdown("**HuggingFace Inference API Settings**")
            hf_key = st.text_input("HuggingFace API Key", type="password")
            hf_model = st.text_input(
                "Model ID",
                value="mistralai/Mistral-7B-Instruct-v0.1",
                help="From huggingface.co/models",
            )
            if hf_key:
                llm_kwargs["api_key"] = hf_key
            if hf_model:
                llm_kwargs["model"] = hf_model

        st.divider()
        st.subheader("🔍 Index Controls")
        top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=DEFAULT_TOP_K)

        rag = get_rag(llm_provider=llm_provider, **llm_kwargs)
        indexed_count = rag.collection.count()
        st.write(f"Indexed chunks: {indexed_count}")

        if st.button("Build / Rebuild Index", use_container_width=True):
            with st.spinner("Ingesting text files and building vector index..."):
                stats = rag.ingest(clear_existing=True)
            st.success(
                f"Indexed {stats['chunks']} chunks from {stats['files']} files into Chroma."
            )
            st.rerun()

    rag = get_rag(llm_provider=llm_provider, **llm_kwargs)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me clinical guideline questions. Use the sidebar to configure LLM and build the index.",
                "sources": [],
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    st.text(_format_sources(message["sources"]))

    prompt = st.chat_input("Ask a question about your guidelines...")

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                try:
                    if not rag.has_index():
                        auto_stats = rag.ingest(clear_existing=False)
                        st.info(
                            f"No index found. Auto-ingested {auto_stats['chunks']} chunks from {auto_stats['files']} files."
                        )

                    # Pass chat history for context resolution (resolve pronouns like "it" to previous topic)
                    answer, sources = rag.answer_question(
                        question=prompt,
                        top_k=top_k,
                        chat_history=st.session_state.messages
                    )
                    st.markdown(answer)
                    with st.expander("Sources"):
                        st.text(_format_sources(sources))

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as exc:
                    err = f"Error: {exc}\n\n{traceback.format_exc()}"
                    st.error(err)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "I hit an error while processing your request.",
                            "sources": [],
                        }
                    )


if __name__ == "__main__":
    main()
