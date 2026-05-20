"""
Self-RAG style verification — scores answer groundedness and relevance.

Uses Pydantic to validate and coerce the LLM's JSON output rather than brittle
regex parsing. Falls back gracefully if the model produces malformed output.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import SETTINGS
from .llm import get_chat_llm
from .logging_config import setup_logging  # noqa: F401

log = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    groundedness: Optional[float] = Field(None, ge=0.0, le=1.0)
    relevance: Optional[float] = Field(None, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    skipped: bool = False

    @field_validator("groundedness", "relevance", mode="before")
    @classmethod
    def coerce_score(cls, v):
        if v is None:
            return None
        try:
            f = float(v)
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            return None


_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_chat_llm(temperature=0.0)
    return _llm


VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are an answer auditor. Given the question, the proposed answer, and
the sources used, return a JSON object with this exact schema:
{{
  "groundedness": <float 0.0-1.0>,
  "relevance":    <float 0.0-1.0>,
  "issues": ["<short string>", ...]
}}

Return ONLY valid JSON, nothing else. No markdown, no explanation.

Question: {question}

Answer:
\"\"\"
{answer}
\"\"\"

Sources:
{sources}
"""
)


def _extract_json(raw: str) -> dict:
    """Try json.loads, then strip markdown fences, then regex-extract first object."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip ```json ... ``` fences
    fenced = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    # Last resort: extract first {...} block
    m = re.search(r"\{[^{}]*\}", fenced, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from verifier output: {raw[:200]!r}")


def verify(question: str, answer: str, docs: list[Document]) -> dict:
    if not SETTINGS.use_verification or not docs:
        return VerificationResult(groundedness=1.0, relevance=1.0, skipped=True).model_dump()

    sources = "\n".join(f"[S{i+1}] {d.page_content[:600]}" for i, d in enumerate(docs))
    msg = VERIFY_PROMPT.format_messages(question=question, answer=answer, sources=sources)
    try:
        raw = _get_llm().invoke(msg).content
        data = _extract_json(raw)
        result = VerificationResult.model_validate(data)
        return result.model_dump()
    except Exception as e:
        log.warning("Verifier failed: %s", e)
        return VerificationResult(
            issues=[f"verifier error: {e}"],
        ).model_dump()
