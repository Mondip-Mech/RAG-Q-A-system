"""
Tests for thread storage (no Chroma / embeddings needed — we mock them out).
"""
import json
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def isolated_threads_dir(tmp_path, monkeypatch):
    """Redirect THREADS_DIR to a temp directory for every test."""
    import backend.memory as mem
    import backend.config as cfg
    monkeypatch.setattr(cfg, "THREADS_DIR", tmp_path)
    monkeypatch.setattr(mem, "THREADS_DIR", tmp_path)
    # Stub out Chroma-backed thread vector store
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    monkeypatch.setattr(mem, "_thread_vs", lambda: mock_vs)
    yield tmp_path


def test_new_thread_creates_file(isolated_threads_dir):
    from backend.memory import new_thread, thread_path
    t = new_thread("Test thread")
    assert thread_path(t["id"]).exists()
    assert t["title"] == "Test thread"
    assert t["messages"] == []


def test_save_and_load_thread(isolated_threads_dir):
    from backend.memory import new_thread, load_thread
    t = new_thread()
    t["title"] = "Updated title"
    from backend.memory import save_thread
    save_thread(t)
    loaded = load_thread(t["id"])
    assert loaded["title"] == "Updated title"


def test_load_nonexistent_thread_returns_none(isolated_threads_dir):
    from backend.memory import load_thread
    assert load_thread("doesnotexist") is None


def test_append_message_persists(isolated_threads_dir):
    from backend.memory import new_thread, append_message, load_thread
    t = new_thread()
    append_message(t, "user", "Hello world")
    reloaded = load_thread(t["id"])
    assert len(reloaded["messages"]) == 1
    assert reloaded["messages"][0]["content"] == "Hello world"


def test_auto_title_from_first_user_message(isolated_threads_dir):
    from backend.memory import new_thread, append_message, load_thread
    t = new_thread()
    append_message(t, "user", "What is attention?")
    reloaded = load_thread(t["id"])
    assert reloaded["title"] == "What is attention?"


def test_delete_thread(isolated_threads_dir):
    from backend.memory import new_thread, delete_thread, load_thread, thread_path
    t = new_thread()
    tid = t["id"]
    delete_thread(tid)
    assert not thread_path(tid).exists()
    assert load_thread(tid) is None


def test_list_threads(isolated_threads_dir):
    from backend.memory import new_thread, list_threads
    new_thread("A")
    new_thread("B")
    threads = list_threads()
    assert len(threads) == 2


def test_thread_pruning(isolated_threads_dir, monkeypatch):
    from backend import config, memory
    monkeypatch.setattr(config.SETTINGS, "max_threads", 2)
    memory.new_thread("First")
    memory.new_thread("Second")
    memory.new_thread("Third")  # should trigger pruning
    remaining = memory.list_threads()
    assert len(remaining) <= 2
