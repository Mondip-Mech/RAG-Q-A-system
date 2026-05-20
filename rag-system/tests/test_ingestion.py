"""
Tests for ingestion helpers that don't require a GPU or live vector store.
"""
import hashlib
import io
import tempfile
from pathlib import Path

import pytest

from backend.ingestion import file_hash, validate_upload


# ---------- file_hash ----------

def test_file_hash_deterministic(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF some content")
    h1 = file_hash(p)
    h2 = file_hash(p)
    assert h1 == h2


def test_file_hash_differs_on_content(tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"%PDF content-a")
    b.write_bytes(b"%PDF content-b")
    assert file_hash(a) != file_hash(b)


def test_file_hash_length(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF hello")
    assert len(file_hash(p)) == 16


# ---------- validate_upload ----------

def test_validate_upload_rejects_non_pdf(tmp_path):
    p = tmp_path / "not_a_pdf.pdf"
    p.write_bytes(b"PK\x03\x04 zip archive")  # ZIP magic bytes
    with pytest.raises(ValueError, match="valid PDF"):
        validate_upload(p)


def test_validate_upload_accepts_valid_pdf(tmp_path):
    p = tmp_path / "valid.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    # Should not raise; uses default 50 MB limit
    validate_upload(p)


def test_validate_upload_rejects_oversized(tmp_path, monkeypatch):
    from backend import config
    monkeypatch.setattr(config.SETTINGS, "max_upload_mb", 0)
    p = tmp_path / "big.pdf"
    p.write_bytes(b"%PDF-1.4 " + b"x" * 100)
    with pytest.raises(ValueError, match="too large"):
        validate_upload(p)
