# Clinical RAG QA Chatbot

A production-ready Retrieval-Augmented Generation (RAG) pipeline for clinical guideline text files, with a Streamlit chat UI supporting multiple LLM providers.

## ✨ Features

- **🤖 Flexible LLM Support**: OpenAI, Ollama (local open-source), HuggingFace Inference API, or extractive fallback
- **💾 Persistent Vector Store**: Chroma database with semantic search
- **🧠 Smart Embeddings**: Sentence-Transformers for document encoding
- **💬 Conversational Context**: Resolves pronouns from chat history
- **📊 Source Attribution**: Shows relevance scores and document previews
- **⚙️ Configurable**: Adjustable chunking, retrieval parameters, and LLM settings

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd clinical-rag

# Install uv (if not already installed)
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys (only needed for OpenAI/HuggingFace providers)
```

### Run the Application

```bash
# Launch Streamlit UI
uv run streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser! 🎉

## 📖 Usage

### Streamlit Web Interface

1. **Select LLM Provider** in the sidebar:
   - **OpenAI**: Best quality (requires API key)
   - **Ollama**: Free local LLMs (requires Ollama installation)
   - **HuggingFace**: Hosted open-source models (requires API key)
   - **Extractive**: No LLM needed (returns relevant chunks)

2. **Build Index**: Click "Build / Rebuild Index" on first run

3. **Ask Questions**: Type your question and get evidence-based answers with sources

### Command-Line Interface

```bash
# Ingest documents and build vector index
uv run python -m src.cli ingest --clear

# Custom chunking parameters
uv run python -m src.cli ingest --clear --chunk-size 220 --overlap 40

# Query from command line
uv run python -m src.cli query "What is the management of diabetic ketoacidosis?"

# Query with more context
uv run python -m src.cli query "Management of septic shock" --top-k 8
```

## 🤖 LLM Provider Setup

### OpenAI (Default)
**Best for: Production-quality answers**

1. Get API key from https://platform.openai.com/api-keys
2. Add to `.env`:
   ```
   OPENAI_API_KEY=sk-proj-...
   OPENAI_MODEL=gpt-4o-mini
   ```

### Ollama (Local, Free)
**Best for: Privacy, offline usage, no API costs**

1. Install Ollama: https://ollama.ai
2. Download a model:
   ```bash
   ollama pull mistral
   # or: llama2, neural-chat, etc.
   ```
3. Start Ollama server:
   ```bash
   ollama serve
   ```
4. Select "ollama" in Streamlit sidebar (no API key needed!)

### HuggingFace Inference API
**Best for: Experimentation with various open-source models**

1. Get API key from https://huggingface.co/settings/tokens
2. Add to `.env`:
   ```
   HUGGINGFACE_API_KEY=hf_...
   HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.1
   ```

### Extractive (No LLM)
**Best for: Testing, debugging retrieval quality**

No setup needed - just select "extractive" in sidebar.

## 📁 Project Structure

```
clinical-rag/
├── src/
│   ├── config.py              # Configuration and defaults
│   ├── models.py              # Data models (RetrievedChunk)
│   ├── rag_pipeline.py        # Core RAG engine
│   ├── llm_providers.py       # LLM provider implementations
│   └── cli.py                 # Command-line interface
├── data/
│   ├── sample_guidelines/     # Source clinical documents (.txt)
│   └── chroma/                # Vector database (auto-generated, gitignored)
├── streamlit_app.py           # Streamlit web interface
├── pyproject.toml             # Project dependencies
├── uv.lock                    # Locked dependency versions
├── .env.example               # Environment variable template
└── README.md                  # This file
```

## 🔧 Advanced Configuration

### Adjusting Retrieval
- **Top-K**: Number of chunks to retrieve (default: 8)
- **Chunk Size**: Words per chunk (default: 220)
- **Chunk Overlap**: Overlapping words between chunks (default: 40)

### Query Preprocessing
The system automatically expands medical terminology:
- "insulin secretion" → adds "physiology pancreatic beta cells islets"
- "dka" → adds "diabetic ketoacidosis"
- Handles pronoun resolution from conversation context

### Custom Documents
Add your own clinical guidelines to `data/sample_guidelines/` as `.txt` files, then rebuild the index.

## 🛠️ Development

```bash
# Install development dependencies
uv sync --all-extras

# Run tests (if you add them)
uv run pytest

# Format code
uv run black src/ streamlit_app.py

# Type checking
uv run mypy src/
```

## 📊 Technical Details

- **Embedding Model**: `sentence-transformers/paraphrase-MiniLM-L6-v2` (384 dimensions)
- **Vector Database**: Chroma (persistent, local)
- **Retrieval**: Semantic similarity search with cosine distance
- **Answer Generation**: Context-augmented prompting with temperature=0.1
- **Conversational**: Pronoun resolution and context tracking

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add new LLM providers
- Improve prompt engineering
- Add advanced retrieval methods (hybrid search, reranking)
- Enhance UI features

## 📄 License

MIT License - feel free to use for your own projects!

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [OpenAI](https://openai.com/)
- [Ollama](https://ollama.ai/)
- [uv](https://github.com/astral-sh/uv)
