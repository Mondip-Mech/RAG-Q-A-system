"""
FastAPI endpoint tests using TestClient.
Mocks out all backend calls so tests run without GPU / API keys.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("backend.ingestion.get_embeddings"), \
         patch("backend.ingestion.get_vectorstore"), \
         patch("backend.memory._thread_vs", return_value=MagicMock()):
        from backend.api import app
        yield TestClient(app, raise_server_exceptions=False)


# ---------- Health ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------- Auth ----------

def test_auth_rejected_when_key_configured(client, monkeypatch):
    from backend import config
    monkeypatch.setattr(config.SETTINGS, "api_key", "secret123")
    r = client.get("/files")
    assert r.status_code == 401


def test_auth_accepted_with_correct_key(client, monkeypatch):
    from backend import config
    monkeypatch.setattr(config.SETTINGS, "api_key", "secret123")
    with patch("backend.ingestion.list_indexed_files", return_value=[]):
        r = client.get("/files", headers={"X-API-Key": "secret123"})
    assert r.status_code == 200


# ---------- Upload validation ----------

def test_upload_rejects_non_pdf(client):
    r = client.post("/upload", files={"file": ("test.txt", b"not a pdf", "text/plain")})
    assert r.status_code == 400


def test_upload_rejects_fake_pdf_extension(client):
    # .pdf extension but ZIP magic bytes
    r = client.post("/upload", files={"file": ("test.pdf", b"PK\x03\x04 zip", "application/pdf")})
    assert r.status_code == 400


def test_upload_rejects_oversized_file(client, monkeypatch):
    from backend import config
    monkeypatch.setattr(config.SETTINGS, "max_upload_mb", 0)
    r = client.post("/upload", files={"file": ("big.pdf", b"%PDF " + b"x" * 100, "application/pdf")})
    assert r.status_code == 413


# ---------- File name safety ----------

def test_delete_path_traversal_rejected(client):
    # FastAPI normalises /files/../secret to /secret before routing, so use
    # percent-encoded traversal which reaches our validation code.
    r = client.delete("/files/..%2Fsecret")
    assert r.status_code in (400, 404, 422)


# ---------- Runtime ----------

def test_runtime_returns_config(client):
    r = client.get("/runtime")
    assert r.status_code == 200
    data = r.json()
    assert "llm" in data
    assert "embeddings" in data


# ---------- Verifier ----------

def test_verifier_parses_clean_json():
    from backend.verifier import _extract_json, VerificationResult
    raw = '{"groundedness": 0.9, "relevance": 0.8, "issues": []}'
    data = _extract_json(raw)
    result = VerificationResult.model_validate(data)
    assert result.groundedness == pytest.approx(0.9)
    assert result.relevance == pytest.approx(0.8)


def test_verifier_strips_markdown_fences():
    from backend.verifier import _extract_json
    raw = '```json\n{"groundedness": 0.5, "relevance": 0.5, "issues": ["x"]}\n```'
    data = _extract_json(raw)
    assert data["groundedness"] == 0.5


def test_verifier_clamps_out_of_range_scores():
    from backend.verifier import VerificationResult
    result = VerificationResult.model_validate({"groundedness": 1.5, "relevance": -0.1, "issues": []})
    assert result.groundedness == pytest.approx(1.0)
    assert result.relevance == pytest.approx(0.0)
