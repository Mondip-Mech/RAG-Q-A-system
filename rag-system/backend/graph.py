"""
LangGraph pipeline:

  retrieve  ->  rerank  ->  compress  ->  generate  ->  verify

Each node has error handling: if a step fails the pipeline continues with a
graceful fallback rather than crashing. Failures are logged for observability.

Two public entry points:
  - run_pipeline(question, history)  -> final state dict (non-streaming)
  - stream_pipeline(question, history) -> iterator of step events + tokens
"""
from __future__ import annotations

import logging
from typing import TypedDict, Iterator

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from .retriever import hybrid_retrieve, rewrite_query
from .reranker import rerank
from .compressor import compress_documents
from .generator import synthesize, stream_answer, format_sources
from .verifier import verify
from .config import SETTINGS
from .logging_config import setup_logging  # noqa: F401

log = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict]
    rewrites: list[str]
    candidates: list[Document]
    reranked: list[Document]
    compressed: list[Document]
    answer: str
    citations: list[dict]
    verification: dict
    errors: list[str]


# ---------- Nodes (each catches its own errors) ----------

def n_retrieve(state: RAGState) -> RAGState:
    q = state["question"]
    errors = list(state.get("errors") or [])
    try:
        rewrites = rewrite_query(q, SETTINGS.multi_query_n) if SETTINGS.use_multi_query else [q]
    except Exception as e:
        log.warning("Query rewrite failed, using original question: %s", e)
        rewrites = [q]
        errors.append(f"rewrite: {e}")
    try:
        cands = hybrid_retrieve(q)
    except Exception as e:
        log.error("Retrieval failed: %s", e)
        cands = []
        errors.append(f"retrieve: {e}")
    return {"rewrites": rewrites, "candidates": cands, "errors": errors}


def n_rerank(state: RAGState) -> RAGState:
    errors = list(state.get("errors") or [])
    cands = state.get("candidates", [])
    try:
        reranked = rerank(state["question"], cands)
    except Exception as e:
        log.warning("Reranking failed, using retrieval order: %s", e)
        reranked = cands[: SETTINGS.top_k_rerank]
        errors.append(f"rerank: {e}")
    return {"reranked": reranked, "errors": errors}


def n_compress(state: RAGState) -> RAGState:
    errors = list(state.get("errors") or [])
    reranked = state.get("reranked", [])
    try:
        compressed = compress_documents(state["question"], reranked)
    except Exception as e:
        log.warning("Compression failed, passing docs uncompressed: %s", e)
        compressed = reranked
        errors.append(f"compress: {e}")
    return {"compressed": compressed, "errors": errors}


def n_generate(state: RAGState) -> RAGState:
    errors = list(state.get("errors") or [])
    try:
        answer, citations = synthesize(
            state["question"], state.get("compressed", []), state.get("history", [])
        )
    except Exception as e:
        log.error("Generation failed: %s", e)
        answer = (
            "### Answer\nI encountered an error generating a response. "
            "Please try again in a moment.\n\n"
            f"### Key Points\n- Error: {e}"
        )
        citations = []
        errors.append(f"generate: {e}")
    return {"answer": answer, "citations": citations, "errors": errors}


def n_verify(state: RAGState) -> RAGState:
    errors = list(state.get("errors") or [])
    try:
        v = verify(state["question"], state.get("answer", ""), state.get("compressed", []))
    except Exception as e:
        log.warning("Verification failed, skipping: %s", e)
        v = {"groundedness": None, "relevance": None, "issues": [], "skipped": True}
        errors.append(f"verify: {e}")
    return {"verification": v, "errors": errors}


# ---------- Graph ----------

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", n_retrieve)
    g.add_node("rerank", n_rerank)
    g.add_node("compress", n_compress)
    g.add_node("generate", n_generate)
    g.add_node("verify", n_verify)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "compress")
    g.add_edge("compress", "generate")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    return g.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------- Public API ----------

def run_pipeline(question: str, history: list[dict] | None = None) -> dict:
    return get_graph().invoke({"question": question, "history": history or []})


def stream_pipeline(question: str, history: list[dict] | None = None) -> Iterator[dict]:
    """Yields events:
       {'type': 'step',  'name': 'retrieve', 'detail': '...'}
       {'type': 'token', 'content': '...'}
       {'type': 'final', 'answer': ..., 'citations': ..., 'verification': ...}
    """
    history = history or []

    yield {"type": "step", "name": "rewrite", "detail": "Rewriting query into multi-query variants…"}
    try:
        rewrites = rewrite_query(question, SETTINGS.multi_query_n) if SETTINGS.use_multi_query else [question]
    except Exception as e:
        log.warning("Rewrite failed in stream: %s", e)
        rewrites = [question]
    yield {"type": "step", "name": "rewrite", "detail": f"{len(rewrites)} variants"}

    yield {"type": "step", "name": "retrieve", "detail": "Hybrid retrieval (dense + BM25)…"}
    try:
        cands = hybrid_retrieve(question)
    except Exception as e:
        log.error("Retrieval failed in stream: %s", e)
        cands = []
    yield {"type": "step", "name": "retrieve", "detail": f"{len(cands)} candidates"}

    yield {"type": "step", "name": "rerank", "detail": "Cross-encoder reranking…"}
    try:
        ranked = rerank(question, cands)
    except Exception as e:
        log.warning("Rerank failed in stream: %s", e)
        ranked = cands[: SETTINGS.top_k_rerank]
    yield {"type": "step", "name": "rerank", "detail": f"top {len(ranked)} retained"}

    yield {"type": "step", "name": "compress", "detail": "Extracting only relevant passages…"}
    try:
        compressed = compress_documents(question, ranked)
    except Exception as e:
        log.warning("Compression failed in stream: %s", e)
        compressed = ranked
    yield {"type": "step", "name": "compress", "detail": f"{len(compressed)} passages kept"}

    yield {"type": "step", "name": "generate", "detail": "Synthesizing grounded answer…"}
    full_text = ""
    citations: list[dict] = []
    try:
        for token, cits in stream_answer(question, compressed, history):
            citations = cits
            full_text += token
            yield {"type": "token", "content": token}
    except Exception as e:
        log.error("Generation streaming failed: %s", e)
        full_text = f"### Answer\nAn error occurred during generation: {e}"
        yield {"type": "token", "content": full_text}

    yield {"type": "step", "name": "verify", "detail": "Self-RAG verification…"}
    try:
        v = verify(question, full_text, compressed)
    except Exception as e:
        log.warning("Verification failed in stream: %s", e)
        v = {"groundedness": None, "relevance": None, "issues": [], "skipped": True}

    yield {
        "type": "final",
        "answer": full_text,
        "citations": citations,
        "verification": v,
        "rewrites": rewrites,
    }
