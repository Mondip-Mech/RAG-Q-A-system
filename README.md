---
title: Rag Qa System
emoji: 🤖
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
---

# RAG Q&A system

A modern, hybrid-cloud RAG system with a polished Streamlit UI. Cloud LLMs (Groq Llama 3.3 70B or NVIDIA NIM) handle generation; embeddings stay local via Ollama. Built with LangChain, LangGraph, and ChromaDB.

## Pipeline

```
User Query
  → Multi-query rewrite
  → Hybrid retrieval (Chroma dense + BM25 sparse, RRF fused)
  → Cross-encoder rerank (BAAI/bge-reranker-base)
  → LLM-based contextual compression
  → Grounded synthesis with structured output
  → Self-RAG verification (groundedness + relevance)
```

## Features

### Retrieval & generation
- **Hybrid retrieval** — dense + BM25 + RRF fusion + multi-query
- **Cross-encoder reranking** for relevance
- **Contextual compression** to keep context tight
- **Self-RAG verification** — auto-scores groundedness and relevance
- **Persistent KB** — Chroma + a manifest, dedup by content hash
- **Chat threads** — saved as JSON, with semantic search across all past conversations
- **Streaming UI** — token-by-token answers, live pipeline step indicators

### Rich answer layout
Every assistant turn renders as a structured, educational answer:
- **TL;DR callout** — one-sentence core takeaway in a highlighted card
- **💡 Answer** — thorough 4–8 sentence explanation
- **📊 Inline diagrams** — Graphviz DOT preferred, Mermaid supported, with a 4-tier fallback chain (sanitize → native render → LLM repair → cross-format conversion → source view)
- **🎯 Key Points** — bulleted takeaways
- **🌍 Real-world Example** — concrete example or analogy in a green-accent callout
- **💡 Continue learning** — clickable follow-up question buttons that auto-submit
- **Meta row** — reading time, word count, source count
- **📋 Copy** — copy the raw answer markdown
- **📚 Sources** — collapsible citation cards (file, page, snippet)
- **Verification badge** — color-coded confidence dot

## Setup

### 1. Install Ollama and pull the embedding model
```bash
# https://ollama.com
ollama pull nomic-embed-text       # embeddings (local)
```

### 2. Install Python deps
```bash
python -m venv .venv && source .venv/bin/activate   # macOS/Linux
# or:  python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Configure your LLM provider
```bash
cp .env.example .env
```

Then edit `.env`:
- Set `LLM_PROVIDER=groq` or `LLM_PROVIDER=nvidia`
- Paste a key from https://console.groq.com/keys or https://build.nvidia.com
- Defaults use Llama 3.3 70B for generation and Llama 3.1 8B for query rewriting

> **⚠️ Never commit `.env`.** It is already in `.gitignore`.

### 4. Run the UI
```bash
streamlit run frontend/app.py
```

Open http://localhost:8501.

### 5. (Alternative) Run the FastAPI backend
```bash
uvicorn backend.api:app --reload --port 8000
```

### 6. Docker
```bash
docker compose up --build
docker exec rag-ollama ollama pull nomic-embed-text
```

## Project structure

```
backend/
  config.py        - Centralized settings
  llm.py           - Provider-agnostic Groq / NVIDIA LLM factory
  ingestion.py     - PDF loading, semantic chunking, Chroma + BM25 build
  retriever.py     - Hybrid retrieval, multi-query, RRF fusion
  reranker.py      - BGE cross-encoder reranking
  compressor.py    - LLM-based contextual compression
  generator.py     - Structured synthesis (TL;DR / Answer / Key Points / Example / Follow-ups)
  verifier.py      - Self-RAG groundedness/relevance scoring
  graph.py         - LangGraph orchestration
  memory.py        - Chat threads + cross-thread semantic search
  api.py           - Optional FastAPI server (use with React/Next.js)
frontend/
  app.py           - Streamlit UI with structured answer rendering
  diagrams.py      - Multi-tier diagram rendering (Mermaid + Graphviz)
  styles.py        - Custom CSS
data/
  chroma_db/       - Persistent vector store + BM25 + manifest (gitignored)
  uploads/         - User-uploaded PDFs (gitignored)
  threads/         - Chat thread JSON files (gitignored)
```

## Tuning

All knobs live in `backend/config.py`. The most impactful:

- `chunk_size` / `chunk_overlap` — tradeoff between retrieval precision and context completeness
- `top_k_dense` / `top_k_sparse` / `top_k_rerank` — recall vs latency
- `rrf_k` — RRF smoothing constant (60 is the canonical default)
- `multi_query_n` — paraphrase count, more = better recall but slower
- `use_compression` — turn off for faster responses if your LLM has a large context

## Switching to a React frontend

Run the FastAPI server instead of Streamlit:
```bash
uvicorn backend.api:app --reload --port 8000
```
The `/chat/stream` endpoint emits Server-Sent Events with the same step+token+final structure the Streamlit UI consumes — easy to wire into a React app with `EventSource`.

## License

MIT
