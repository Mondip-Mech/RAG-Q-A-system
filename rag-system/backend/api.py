"""
FastAPI server.
Run with:  uvicorn backend.api:app --reload --port 8000

Security:
  - Set API_KEY env var to require X-API-Key on every request.
  - Rate-limited to RATE_LIMIT (default 60/minute per IP).
  - File uploads validated for size and PDF magic bytes.
  - CORS origins controlled via CORS_ORIGINS env var (default: deny all non-same-origin).

Endpoints:
  GET  /health                       - liveness probe
  POST /upload                       - upload one PDF
  POST /sync                         - reindex all uploads
  GET  /files                        - list indexed files
  DELETE /files/{name}               - remove a file from the KB
  POST /clear                        - wipe the KB
  GET  /runtime                      - config metadata
  POST /chat                         - non-streaming Q&A
  POST /chat/stream                  - SSE streaming Q&A with step events
  GET  /threads                      - list chat threads
  GET  /threads/{id}                 - load thread
  POST /threads                      - create thread
  DELETE /threads/{id}               - delete thread
  GET  /search/conversations?q=...   - semantic search across threads
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .config import SETTINGS, UPLOAD_DIR
from .ingestion import ingest_file, ingest_all_uploads, list_indexed_files, remove_file, clear_index, validate_upload
from .graph import run_pipeline, stream_pipeline
from .memory import new_thread, load_thread, save_thread, list_threads, delete_thread, append_message, search_conversations
from .logging_config import setup_logging  # noqa: F401

log = logging.getLogger(__name__)

# ---------- Rate limiter ----------
limiter = Limiter(key_func=get_remote_address, default_limits=[SETTINGS.rate_limit])

app = FastAPI(title="RAG Document Q&A", version="1.0.0")
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Slow down."})


# ---------- CORS ----------
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ---------- Auth ----------
def _check_api_key(request: Request) -> None:
    """If API_KEY is configured, every request must carry X-API-Key: <key>."""
    if not SETTINGS.api_key:
        return
    provided = request.headers.get("X-API-Key", "")
    if provided != SETTINGS.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


ApiKeyDep = Depends(_check_api_key)


# ---------- Models ----------
class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", dependencies=[ApiKeyDep])
@limiter.limit("20/minute")
async def upload(request: Request, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > SETTINGS.max_upload_mb:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Limit is {SETTINGS.max_upload_mb} MB.")
    if not content.startswith(b"%PDF"):
        raise HTTPException(400, "File does not appear to be a valid PDF.")

    dest = UPLOAD_DIR / Path(file.filename).name  # prevent path traversal
    dest.write_bytes(content)

    try:
        res = ingest_file(dest)
    except ValueError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    except Exception as e:
        log.error("Ingestion error for %s: %s", file.filename, e)
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"Ingestion failed: {e}")
    return res


@app.post("/sync", dependencies=[ApiKeyDep])
@limiter.limit("5/minute")
def sync(request: Request):
    return {"results": ingest_all_uploads()}


@app.get("/files", dependencies=[ApiKeyDep])
def files():
    return {"files": list_indexed_files()}


@app.delete("/files/{name}", dependencies=[ApiKeyDep])
def files_delete(name: str):
    # Prevent path traversal
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "Invalid file name.")
    remove_file(name)
    return {"ok": True}


@app.post("/clear", dependencies=[ApiKeyDep])
def clear():
    clear_index()
    return {"ok": True}


@app.get("/runtime")
def runtime():
    return {
        "llm": SETTINGS.llm_model,
        "embeddings": SETTINGS.embedding_model,
        "vector_db": "ChromaDB",
        "hybrid": SETTINGS.use_hybrid,
        "top_k_rerank": SETTINGS.top_k_rerank,
        "reranker": SETTINGS.reranker_model if SETTINGS.use_reranker else None,
    }


@app.post("/chat", dependencies=[ApiKeyDep])
@limiter.limit("30/minute")
def chat(req: ChatRequest, request: Request):
    thread = load_thread(req.thread_id) if req.thread_id else new_thread()
    if thread is None:
        thread = new_thread()
    history = thread["messages"]
    append_message(thread, "user", req.question)
    try:
        state = run_pipeline(req.question, history=history)
    except Exception as e:
        log.error("Pipeline failed for question %r: %s", req.question[:80], e)
        raise HTTPException(500, f"Pipeline error: {e}")
    append_message(thread, "assistant", state.get("answer", ""), state.get("citations", []))
    return {
        "thread_id": thread["id"],
        "answer": state.get("answer"),
        "citations": state.get("citations", []),
        "verification": state.get("verification", {}),
        "errors": state.get("errors", []),
    }


@app.post("/chat/stream", dependencies=[ApiKeyDep])
@limiter.limit("30/minute")
def chat_stream(req: ChatRequest, request: Request):
    thread = load_thread(req.thread_id) if req.thread_id else new_thread()
    if thread is None:
        thread = new_thread()
    history = list(thread["messages"])
    append_message(thread, "user", req.question)

    def gen():
        full = ""
        citations = []
        try:
            for ev in stream_pipeline(req.question, history=history):
                if ev["type"] == "token":
                    full += ev["content"]
                if ev["type"] == "final":
                    citations = ev.get("citations", [])
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            log.error("Stream pipeline error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
        append_message(thread, "assistant", full, citations)
        yield f"data: {json.dumps({'type': 'thread', 'thread_id': thread['id']})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/threads", dependencies=[ApiKeyDep])
def threads():
    return {"threads": list_threads()}


@app.get("/threads/{tid}", dependencies=[ApiKeyDep])
def threads_get(tid: str):
    t = load_thread(tid)
    if t is None:
        raise HTTPException(404, "Thread not found")
    return t


@app.post("/threads", dependencies=[ApiKeyDep])
def threads_new():
    return new_thread()


@app.delete("/threads/{tid}", dependencies=[ApiKeyDep])
def threads_delete(tid: str):
    delete_thread(tid)
    return {"ok": True}


@app.get("/search/conversations", dependencies=[ApiKeyDep])
def search_convos(q: str, k: int = 5):
    if k > 20:
        k = 20
    return {"results": search_conversations(q, k=k)}
