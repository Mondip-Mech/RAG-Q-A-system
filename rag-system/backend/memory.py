"""
Chat thread storage + semantic search over past conversations.

- Threads are stored as JSON files under data/threads/<thread_id>.json
- Writes are atomic (write to .tmp then rename) and protected by per-file locks
- Old threads are pruned when the total count exceeds MAX_THREADS
- A separate Chroma collection 'thread_messages' enables semantic search
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from filelock import FileLock
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import SETTINGS, THREADS_DIR
from .ingestion import get_embeddings
from .logging_config import setup_logging  # noqa: F401

log = logging.getLogger(__name__)

THREAD_COLLECTION = "thread_messages"


def _thread_vs() -> Chroma:
    return Chroma(
        collection_name=THREAD_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=SETTINGS.chroma_dir,
    )


def _lock(tid: str) -> FileLock:
    return FileLock(str(THREADS_DIR / f"{tid}.lock"), timeout=10)


def thread_path(tid: str) -> Path:
    return THREADS_DIR / f"{tid}.json"


# ---------- Pruning ----------
def _prune_old_threads() -> None:
    """Delete the oldest threads when the count exceeds MAX_THREADS."""
    files = sorted(THREADS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    excess = len(files) - SETTINGS.max_threads
    if excess <= 0:
        return
    for p in files[:excess]:
        tid = p.stem
        try:
            with _lock(tid):
                p.unlink(missing_ok=True)
            _thread_vs().delete(where={"thread_id": tid})
            log.info("Pruned old thread %s", tid)
        except Exception as e:
            log.warning("Could not prune thread %s: %s", tid, e)


# ---------- Public API ----------
def new_thread(title: str = "New chat") -> dict:
    tid = uuid.uuid4().hex[:12]
    thread = {
        "id": tid,
        "title": title,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "messages": [],
    }
    save_thread(thread)
    _prune_old_threads()
    return thread


def save_thread(thread: dict) -> None:
    thread["updated_at"] = datetime.utcnow().isoformat()
    p = thread_path(thread["id"])
    tmp = p.with_suffix(".tmp")
    with _lock(thread["id"]):
        tmp.write_text(json.dumps(thread, indent=2))
        tmp.replace(p)


def load_thread(tid: str) -> dict | None:
    p = thread_path(tid)
    if not p.exists():
        return None
    with _lock(tid):
        try:
            return json.loads(p.read_text())
        except Exception as e:
            log.warning("Could not load thread %s: %s", tid, e)
            return None


def list_threads() -> list[dict]:
    out = []
    for p in sorted(THREADS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            t = json.loads(p.read_text())
            out.append({
                "id": t["id"],
                "title": t.get("title", "Untitled"),
                "updated_at": t.get("updated_at"),
                "n_messages": len(t.get("messages", [])),
            })
        except Exception:
            continue
    return out


def delete_thread(tid: str) -> None:
    p = thread_path(tid)
    with _lock(tid):
        p.unlink(missing_ok=True)
    try:
        _thread_vs().delete(where={"thread_id": tid})
    except Exception:
        pass
    log.info("Deleted thread %s", tid)


def append_message(thread: dict, role: str, content: str, citations: list[dict] | None = None) -> None:
    msg = {
        "role": role,
        "content": content,
        "ts": datetime.utcnow().isoformat(),
        "citations": citations or [],
    }
    thread["messages"].append(msg)
    if role == "user" and thread.get("title", "New chat") == "New chat":
        thread["title"] = content[:60]
    save_thread(thread)

    if content.strip():
        try:
            _thread_vs().add_documents([
                Document(
                    page_content=content,
                    metadata={
                        "thread_id": thread["id"],
                        "thread_title": thread["title"],
                        "role": role,
                        "ts": msg["ts"],
                    },
                )
            ])
        except Exception as e:
            log.warning("Failed to index message in thread %s: %s", thread["id"], e)


def search_conversations(query: str, k: int = 5) -> list[dict]:
    try:
        results = _thread_vs().similarity_search(query, k=k)
    except Exception as e:
        log.warning("Conversation search failed: %s", e)
        return []
    return [
        {
            "thread_id": r.metadata.get("thread_id"),
            "thread_title": r.metadata.get("thread_title"),
            "role": r.metadata.get("role"),
            "snippet": r.page_content[:240],
            "ts": r.metadata.get("ts"),
        }
        for r in results
    ]
