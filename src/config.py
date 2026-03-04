from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "sample_guidelines"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "clinical_docs"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
DEFAULT_TOP_K = 8  # Increased from 5 to retrieve more context
DEFAULT_CHUNK_SIZE_WORDS = 220
DEFAULT_CHUNK_OVERLAP_WORDS = 40

# LLM Provider defaults
DEFAULT_LLM_PROVIDER = "openai"  # Options: "openai", "ollama", "huggingface", "extractive"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "mistral"
DEFAULT_HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

