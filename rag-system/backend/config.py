"""
Centralized configuration for the RAG system.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
THREADS_DIR = DATA_DIR / "threads"

for d in (UPLOAD_DIR, CHROMA_DIR, THREADS_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    # ---- LLM provider ----
    # Options: 'groq' | 'nvidia'
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").lower())

    # Groq
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_rewrite_model: str = field(default_factory=lambda: os.getenv("GROQ_REWRITE_MODEL", "llama-3.1-8b-instant"))

    # NVIDIA NIM
    nvidia_api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    nvidia_model: str = field(default_factory=lambda: os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"))
    nvidia_rewrite_model: str = field(default_factory=lambda: os.getenv("NVIDIA_REWRITE_MODEL", "meta/llama-3.1-8b-instruct"))

    llm_temperature: float = 0.1

    # ---- Embeddings (in-process via sentence-transformers; no external service) ----
    # Any HuggingFace sentence-transformers model id works.
    # Defaults to BAAI/bge-small-en-v1.5 — ~130MB, strong quality, fast on CPU.
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # ---- Reranker ----
    reranker_model: str = "BAAI/bge-reranker-base"
    use_reranker: bool = True

    # ---- Vector store ----
    chroma_collection: str = "rag_documents"
    chroma_dir: str = str(CHROMA_DIR)

    # ---- Chunking ----
    chunk_size: int = 1800
    chunk_overlap: int = 150
    use_semantic_chunking: bool = False

    # ---- Retrieval ----
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_rerank: int = 8
    use_hybrid: bool = True
    use_multi_query: bool = True
    multi_query_n: int = 3
    rrf_k: int = 60

    # ---- Compression ----
    use_compression: bool = True
    max_context_chars: int = 6000

    # ---- Verification ----
    use_verification: bool = True

    # ---- Memory ----
    memory_window: int = 6

    # ---- Security ----
    # Set API_KEY env var to require X-API-Key header on all API endpoints.
    # Leave empty to run without auth (fine for local / HF Spaces personal use).
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))

    # ---- Upload limits ----
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "50")))

    # ---- Thread retention ----
    max_threads: int = field(default_factory=lambda: int(os.getenv("MAX_THREADS", "200")))

    # ---- Rate limiting ----
    rate_limit: str = field(default_factory=lambda: os.getenv("RATE_LIMIT", "60/minute"))

    # ---- Convenience: returns the active model name based on provider ----
    @property
    def llm_model(self) -> str:
        return self.groq_model if self.llm_provider == "groq" else self.nvidia_model

    @property
    def rewrite_model(self) -> str:
        return self.groq_rewrite_model if self.llm_provider == "groq" else self.nvidia_rewrite_model


SETTINGS = Settings()