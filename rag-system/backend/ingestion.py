"""
Document ingestion pipeline.

- Loads PDFs (PyMuPDF for speed + better text extraction than PyPDF)
- Performs semantic chunking when possible, falls back to recursive splitter
- Persists chunks into Chroma with metadata (source, page, chunk_id)
- Maintains a parallel BM25 index (rebuilt on each sync)
- Uses FileLock to prevent concurrent writes corrupting the manifest / BM25
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Iterable

from filelock import FileLock
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from .config import SETTINGS, UPLOAD_DIR, CHROMA_DIR
from .logging_config import setup_logging  # noqa: F401 – ensures logging is initialised

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"
BM25_PICKLE = Path(SETTINGS.chroma_dir) / "bm25.pkl"
MANIFEST_PATH = Path(SETTINGS.chroma_dir) / "manifest.json"

# One lock per shared file — held only during the actual read/write, not the whole ingest.
_MANIFEST_LOCK = FileLock(str(MANIFEST_PATH) + ".lock", timeout=30)
_BM25_LOCK = FileLock(str(BM25_PICKLE) + ".lock", timeout=60)


# ---------- Embeddings (singleton, in-process via sentence-transformers) ----------
_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        log.info("Loading embedding model %s", SETTINGS.embedding_model)
        _embeddings = HuggingFaceEmbeddings(
            model_name=SETTINGS.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 64,
            },
            show_progress=True,
        )
        log.info("Embedding model loaded")
    return _embeddings


# ---------- Vector store ----------
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=SETTINGS.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=SETTINGS.chroma_dir,
    )


# ---------- Manifest ----------
def load_manifest() -> dict:
    with _MANIFEST_LOCK:
        if MANIFEST_PATH.exists():
            try:
                return json.loads(MANIFEST_PATH.read_text())
            except Exception:
                log.warning("Corrupt manifest; starting fresh")
    return {"files": {}}


def save_manifest(m: dict) -> None:
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    with _MANIFEST_LOCK:
        tmp.write_text(json.dumps(m, indent=2))
        tmp.replace(MANIFEST_PATH)  # atomic on POSIX; best-effort on Windows


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------- File validation ----------
def validate_upload(path: Path) -> None:
    """Raise ValueError if the file is too large or not a real PDF."""
    size_mb = path.stat().st_size / (1024 * 1024)
    limit_mb = SETTINGS.max_upload_mb
    if size_mb > limit_mb:
        raise ValueError(f"File too large: {size_mb:.1f} MB (limit {limit_mb} MB)")
    with open(path, "rb") as f:
        header = f.read(4)
    if header != PDF_MAGIC:
        raise ValueError("File does not appear to be a valid PDF")


# ---------- Loading & chunking ----------
def load_pdf(path: Path) -> list[Document]:
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = path.name
        d.metadata["source_path"] = str(path)
        d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
    return docs


def make_splitter():
    if SETTINGS.use_semantic_chunking:
        try:
            return SemanticChunker(
                embeddings=get_embeddings(),
                breakpoint_threshold_type="percentile",
            )
        except Exception:
            pass
    return RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = make_splitter()
    chunks: list[Document] = []
    for i, doc in enumerate(splitter.split_documents(docs)):
        cid = f"{doc.metadata.get('source','?')}::p{doc.metadata.get('page','?')}::c{i}"
        doc.metadata["chunk_id"] = cid
        chunks.append(doc)
    return chunks


# ---------- BM25 (sparse) ----------
def rebuild_bm25(all_docs: Iterable[Document]) -> None:
    docs = list(all_docs)
    with _BM25_LOCK:
        if not docs:
            if BM25_PICKLE.exists():
                BM25_PICKLE.unlink()
            return
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = SETTINGS.top_k_sparse
        tmp = BM25_PICKLE.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(docs, f)
        tmp.replace(BM25_PICKLE)


def load_bm25() -> BM25Retriever | None:
    with _BM25_LOCK:
        if not BM25_PICKLE.exists():
            return None
        try:
            with open(BM25_PICKLE, "rb") as f:
                docs = pickle.load(f)
        except Exception as e:
            log.warning("BM25 pickle corrupt, skipping sparse retrieval: %s", e)
            return None
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = SETTINGS.top_k_sparse
    return bm25


# ---------- Public API ----------
def ingest_file(path: Path, progress_cb=None) -> dict:
    """Ingest a single PDF. Skips if already ingested (by hash).

    progress_cb: optional callable(str) for UI status updates.
    """
    def _p(msg: str) -> None:
        log.info(msg.replace("📄", "").replace("✂️", "").replace("🧠", "").replace("🔍", "").strip())
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    try:
        validate_upload(path)
    except ValueError as e:
        log.error("Upload rejected for %s: %s", path.name, e)
        raise

    manifest = load_manifest()
    fhash = file_hash(path)
    if path.name in manifest["files"] and manifest["files"][path.name]["hash"] == fhash:
        log.info("Skipping %s (already ingested)", path.name)
        return {"status": "skipped", "file": path.name, "chunks": manifest["files"][path.name]["chunks"]}

    _p(f"📄 Loading {path.name}…")
    docs = load_pdf(path)

    _p(f"✂️ Chunking {len(docs)} pages…")
    chunks = chunk_documents(docs)

    vs = get_vectorstore()
    try:
        vs.delete(where={"source": path.name})
    except Exception:
        pass

    _p(f"🧠 Embedding {len(chunks)} chunks (slow step on CPU)…")
    vs.add_documents(chunks)

    manifest["files"][path.name] = {
        "hash": fhash,
        "chunks": len(chunks),
        "pages": len({d.metadata.get("page") for d in docs}),
    }
    save_manifest(manifest)

    _p("🔍 Rebuilding BM25 sparse index…")
    all_docs = vs.get()
    rebuilt = [
        Document(page_content=t, metadata=m)
        for t, m in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    rebuild_bm25(rebuilt)

    log.info("Ingested %s: %d chunks", path.name, len(chunks))
    return {"status": "ingested", "file": path.name, "chunks": len(chunks)}


def ingest_all_uploads(progress_cb=None) -> list[dict]:
    results = []
    for p in sorted(UPLOAD_DIR.glob("*.pdf")):
        try:
            results.append(ingest_file(p, progress_cb=progress_cb))
        except Exception as e:
            log.error("Failed to ingest %s: %s", p.name, e)
            results.append({"status": "error", "file": p.name, "error": str(e)})
    return results


def list_indexed_files() -> list[dict]:
    m = load_manifest()
    return [{"name": k, **v} for k, v in m["files"].items()]


def remove_file(name: str) -> None:
    vs = get_vectorstore()
    try:
        vs.delete(where={"source": name})
    except Exception:
        pass
    m = load_manifest()
    m["files"].pop(name, None)
    save_manifest(m)
    p = UPLOAD_DIR / name
    if p.exists():
        p.unlink()
    all_docs = vs.get()
    rebuilt = [
        Document(page_content=t, metadata=m_)
        for t, m_ in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    rebuild_bm25(rebuilt)
    log.info("Removed %s from knowledge base", name)


def clear_index() -> None:
    vs = get_vectorstore()
    try:
        vs.delete_collection()
    except Exception:
        pass
    with _BM25_LOCK:
        if BM25_PICKLE.exists():
            BM25_PICKLE.unlink()
    with _MANIFEST_LOCK:
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()
    log.info("Knowledge base cleared")
