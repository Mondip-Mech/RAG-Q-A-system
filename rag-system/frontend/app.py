"""
Streamlit UI for the RAG Document Q&A system.

Layout:
  Sidebar:
    - PDF uploader
    - Sync / Clear buttons
    - Runtime panel (Provider, LLM, embeddings, vector DB, hybrid, top-k)
    - Indexed files
    - Chat threads list
    - Conversation search
  Main:
    - Hero header with feature badges
    - Chat thread (user + assistant turns)
    - Inline citation cards under each assistant answer
    - Sticky bottom chat input with streaming
"""
from __future__ import annotations

import re
import sys
import html as html_lib
from pathlib import Path

import streamlit as st

# Make sibling backend importable when running `streamlit run frontend/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import SETTINGS, UPLOAD_DIR
from backend.ingestion import (
    ingest_file,
    ingest_all_uploads,
    list_indexed_files,
    remove_file,
    clear_index,
)
from backend.graph import stream_pipeline
from backend.memory import (
    new_thread,
    load_thread,
    save_thread,
    list_threads,
    delete_thread,
    append_message,
    search_conversations,
)
from frontend.styles import CUSTOM_CSS

from frontend.diagrams import render_diagrams


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Pre-warm the reranker so the first question doesn't hang on model download
@st.cache_resource
def _warmup_reranker():
    if SETTINGS.use_reranker:
        from backend.reranker import _load_model
        _load_model()
    return True


# Pre-warm the embedder so the first ingestion doesn't pay the model load cost
@st.cache_resource
def _warmup_embedder():
    from backend.ingestion import get_embeddings
    emb = get_embeddings()
    # Force the underlying SentenceTransformer to load now (it's lazy by default)
    emb.embed_query("warmup")
    return True


if not st.session_state.get("_models_ready"):
    with st.spinner("Loading models (reranker + embedder)…"):
        _warmup_reranker()
        _warmup_embedder()
    st.session_state["_models_ready"] = True
else:
    # Already loaded — just touch the cache entries (no-op, instant)
    _warmup_reranker()
    _warmup_embedder()


# ============================================================
# Session state
# ============================================================
def _init_state():
    if "thread" not in st.session_state:
        threads = list_threads()
        if threads:
            st.session_state.thread = load_thread(threads[0]["id"])
        else:
            st.session_state.thread = new_thread()


_init_state()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")

    # ---- Uploads ----
    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    # Guard: only ingest files that haven't been processed in this browser session.
    # Without this, every Streamlit rerun (e.g. from widget interaction) would
    # re-trigger ingest_file() for the same files the uploader still holds,
    # producing an infinite rerun loop and "Skipping…" log spam.
    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()

    if uploaded:
        new_files = [f for f in uploaded if f.name not in st.session_state.processed_uploads]
        if new_files:
            with st.status("Ingesting…", expanded=True) as status:
                for f in new_files:
                    status.update(label=f"Ingesting {f.name}…")
                    dest = UPLOAD_DIR / f.name
                    dest.write_bytes(f.getbuffer())
                    res = ingest_file(dest, progress_cb=lambda m: status.write(m))
                    status.write(f"✅ {res['status']}: {res['file']} ({res.get('chunks', 0)} chunks)")
                    st.session_state.processed_uploads.add(f.name)
                status.update(label="Done.", state="complete")
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Sync KB", use_container_width=True):
            with st.status("Reindexing…", expanded=True) as status:
                ingest_all_uploads(progress_cb=lambda m: status.write(m))
                status.update(label="Knowledge base synced.", state="complete")
            st.rerun()
    with col2:
        if st.button("🗑️ Clear KB", use_container_width=True):
            clear_index()
            st.toast("KB cleared")
            st.rerun()

    # ---- Runtime panel ----
    provider_color = "ok" if SETTINGS.llm_provider in ("groq", "nvidia") else ""
    # Show only the basename (after the last "/") so HF model ids like
    # "sentence-transformers/all-MiniLM-L6-v2" don't overflow the row.
    embed_short = SETTINGS.embedding_model.rsplit("/", 1)[-1]
    llm_short = SETTINGS.llm_model.rsplit("/", 1)[-1]
    st.markdown(
        f"""
        <div class="sidebar-section">
          <div class="label">Runtime</div>
          <div class="runtime-row"><span class="k">Provider</span><span class="v {provider_color}">{SETTINGS.llm_provider.upper()}</span></div>
          <div class="runtime-row"><span class="k">LLM</span><span class="v" title="{SETTINGS.llm_model}">{llm_short}</span></div>
          <div class="runtime-row"><span class="k">Embeddings</span><span class="v" title="{SETTINGS.embedding_model}">{embed_short}</span></div>
          <div class="runtime-row"><span class="k">Vector DB</span><span class="v">ChromaDB</span></div>
          <div class="runtime-row"><span class="k">Hybrid search</span><span class="v ok">{'on' if SETTINGS.use_hybrid else 'off'}</span></div>
          <div class="runtime-row"><span class="k">Reranker</span><span class="v ok">{'on' if SETTINGS.use_reranker else 'off'}</span></div>
          <div class="runtime-row"><span class="k">Mode</span><span class="v ok">Learning</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Indexed files ----
    files = list_indexed_files()
    st.markdown('<div class="sidebar-section"><div class="label">Indexed files</div></div>', unsafe_allow_html=True)
    if not files:
        st.caption("No documents indexed yet.")
    else:
        for f in files:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f'<div class="file-pill"><div><div class="name">📄 {html_lib.escape(f["name"])[:32]}</div>'
                    f'<div class="meta">{f["chunks"]} chunks · {f.get("pages","?")} pages</div></div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("✕", key=f"del_{f['name']}", help="Remove"):
                    remove_file(f["name"])
                    st.rerun()

    # ---- Index status ----
    total_chunks = sum(f["chunks"] for f in files)
    st.markdown(
        f"""
        <div class="sidebar-section">
          <div class="label">Index status</div>
          <div class="runtime-row"><span class="k">Files</span><span class="v">{len(files)}</span></div>
          <div class="runtime-row"><span class="k">Chunks</span><span class="v">{total_chunks}</span></div>
          <div class="runtime-row"><span class="k">Status</span><span class="v ok">{'ready' if total_chunks else 'empty'}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Chat threads ----
    st.markdown("### 💬 Chats")
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.thread = new_thread()
        st.rerun()
    if st.button("🧹 Clear current chat", use_container_width=True):
        delete_thread(st.session_state.thread["id"])
        st.session_state.thread = new_thread()
        st.rerun()

    threads = list_threads()
    for t in threads[:25]:
        active = t["id"] == st.session_state.thread["id"]
        label = ("● " if active else "  ") + t["title"][:36]
        if st.button(label, key=f"th_{t['id']}", use_container_width=True):
            st.session_state.thread = load_thread(t["id"])
            st.rerun()

    # ---- Semantic search across conversations ----
    st.markdown("### 🔍 Search past chats")
    q = st.text_input("Search", label_visibility="collapsed", placeholder="search conversations…")
    if q:
        results = search_conversations(q, k=5)
        for r in results:
            if st.button(
                f"📜 {r['thread_title'][:32]} — {r['snippet'][:60]}",
                key=f"srch_{r['thread_id']}_{r.get('ts','')}",
                use_container_width=True,
            ):
                st.session_state.thread = load_thread(r["thread_id"])
                st.rerun()


# ============================================================
# Main area
# ============================================================

# Hero — badge text adapts to provider
provider_label = SETTINGS.llm_provider.upper()
st.markdown(
    f"""
    <div class="hero">
      <h1>RAG Q&A system</h1>
      <p>Ask questions across your documents and get clear, educational explanations. Powered by hybrid retrieval and a fast cloud LLM.</p>
      <div class="badges">
        <span class="badge"><span class="dot"></span>{provider_label} · Llama 3.3</span>
        <span class="badge"><span class="dot"></span>Hybrid Retrieval</span>
        <span class="badge"><span class="dot"></span>Visual Diagrams</span>
        <span class="badge"><span class="dot"></span>Learning Mode</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Render helpers ----------
CITE_RE = re.compile(r"\[(S\d+)\]")


def render_answer_html(text: str) -> str:
    """Strip [S#] tags and convert markdown to HTML for clean rendering."""
    import markdown as md
    # Strip any leaked citation tags
    text = CITE_RE.sub("", text)
    # Convert markdown (### headers, **bold**, lists, etc.) to HTML
    return md.markdown(text, extensions=["extra", "sane_lists"])

def sanitize_mermaid(code: str) -> str:
    """Clean up common Mermaid syntax issues from LLM output:
    - Replace smart quotes with straight quotes
    - Replace problematic chars in node labels
    - Strip empty lines that confuse the parser
    """
    # Replace smart/curly quotes
    code = (code.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))

    # Inside square-bracket labels, replace problematic characters
    def clean_label(match):
        label = match.group(1)
        # Replace chars that break Mermaid parsing
        label = label.replace("<", "lt").replace(">", "gt")
        label = label.replace("(", "").replace(")", "")
        label = label.replace("&", "and")
        label = label.replace('"', "").replace("'", "")
        return f"[{label}]"

    code = re.sub(r"\[([^\[\]]+)\]", clean_label, code)

    # Strip blank lines inside the diagram (Mermaid prefers compact)
    lines = [ln for ln in code.splitlines() if ln.strip()]
    return "\n".join(lines)


SECTION_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


def parse_sections(content: str) -> dict[str, str]:
    """Split a structured answer into its named sections.

    Returns a dict like {"tl;dr": "...", "answer": "...", "key points": "...",
    "real-world example": "...", "want to learn more?": "..."} (lowercased keys).
    Anything before the first ### heading is stored under the "_preamble" key.
    """
    sections: dict[str, str] = {}
    matches = list(SECTION_RE.finditer(content))
    if not matches:
        sections["_preamble"] = content.strip()
        return sections

    if matches[0].start() > 0:
        sections["_preamble"] = content[: matches[0].start()].strip()

    for i, m in enumerate(matches):
        name = m.group(1).strip().lower()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections[name] = content[body_start:body_end].strip()
    return sections


def parse_followups(text: str) -> list[str]:
    """Extract bullet items from the 'Want to learn more?' section."""
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("-") or line.startswith("*"):
            q = line.lstrip("-* ").strip()
            if q:
                items.append(q)
    return items


def render_message(role: str, content: str, citations: list[dict] | None = None,
                   verification: dict | None = None, msg_key: str = "static",
                   is_latest: bool = False):
    """Render one chat message with rich, structured layout."""
    cls = "user" if role == "user" else "bot"
    role_label = "You" if role == "user" else "Assistant"
    avatar_letter = "U" if role == "user" else "AI"

    # ---- User: simple bubble ----
    if role == "user":
        body_html = render_answer_html(content)
        st.markdown(
            f'<div class="chat-msg {cls}"><div class="avatar {cls}">{avatar_letter}</div>'
            f'<div class="msg-body"><div class="msg-role">{role_label}</div>'
            f'<div class="msg-content">{body_html}</div></div></div>',
            unsafe_allow_html=True,
        )
        return

    # ---- Assistant: structured rendering ----
    from frontend.diagrams import MERMAID_RE, DOT_RE

    sections = parse_sections(content)
    tldr = sections.get("tl;dr") or sections.get("tldr") or ""
    answer = sections.get("answer") or sections.get("_preamble", "")
    key_points = sections.get("key points", "")
    example = sections.get("real-world example") or sections.get("example", "")
    followups_text = sections.get("want to learn more?") or sections.get("follow-ups", "")
    followups = parse_followups(followups_text)

    # Strip diagram blocks from any prose so they only render once via render_diagrams
    def clean(t: str) -> str:
        t = MERMAID_RE.sub("", t)
        t = DOT_RE.sub("", t)
        return t.strip()

    answer_clean = clean(answer)
    key_points_clean = clean(key_points)
    example_clean = clean(example)

    # Reading time estimate (200 wpm, min 1 minute)
    word_count = len(re.findall(r"\w+", content))
    read_min = max(1, round(word_count / 200))

    # Open the chat-msg shell and the body via raw HTML
    st.markdown(
        f'<div class="chat-msg {cls}"><div class="avatar {cls}">{avatar_letter}</div>'
        f'<div class="msg-body"><div class="msg-role">{role_label}</div>',
        unsafe_allow_html=True,
    )

    # Meta row
    meta_pills = [
        f'<span class="pill">⏱ {read_min} min read</span>',
        f'<span class="pill">📝 {word_count} words</span>',
    ]
    if citations:
        meta_pills.append(f'<span class="pill">📚 {len(citations)} sources</span>')
    st.markdown(
        f'<div class="msg-meta">{"".join(meta_pills)}</div>',
        unsafe_allow_html=True,
    )

    # TL;DR card
    if tldr:
        tldr_html = render_answer_html(tldr).replace("<p>", "").replace("</p>", "")
        st.markdown(
            f'<div class="tldr-card"><span class="tldr-tag">TL;DR</span>'
            f'<div>{tldr_html}</div></div>',
            unsafe_allow_html=True,
        )

    # Answer body (under "💡 Answer" header)
    if answer_clean:
        body_html = render_answer_html(f"### 💡 Answer\n\n{answer_clean}")
        st.markdown(f'<div class="msg-content">{body_html}</div>', unsafe_allow_html=True)

    # Diagrams with fallback chain
    render_diagrams(content)

    # Key Points
    if key_points_clean:
        kp_html = render_answer_html(f"### 🎯 Key Points\n\n{key_points_clean}")
        st.markdown(f'<div class="msg-content">{kp_html}</div>', unsafe_allow_html=True)

    # Real-world Example as a green-accent callout
    if example_clean:
        ex_html = render_answer_html(example_clean)
        st.markdown(
            f'<div class="msg-content"><h3>🌍 Real-world Example</h3></div>'
            f'<div class="example-card">{ex_html}</div>',
            unsafe_allow_html=True,
        )

    # Verification badge
    if verification:
        score = verification.get("score") or verification.get("confidence")
        label = verification.get("label") or verification.get("status") or "verified"
        if score is not None:
            try:
                score_f = float(score)
                tier = "ok" if score_f >= 0.75 else ("mid" if score_f >= 0.5 else "bad")
                st.markdown(
                    f'<div class="verify"><span class="{tier}">●</span> '
                    f'<span>Verification: <b>{label}</b> · score {score_f:.2f}</span></div>',
                    unsafe_allow_html=True,
                )
            except (TypeError, ValueError):
                pass

    # Close .msg-body and .chat-msg
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Follow-up clickable questions (outside the bubble for visual separation)
    if followups:
        st.markdown(
            '<div class="followup-label">💡 Continue learning</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="followup-row">', unsafe_allow_html=True)
        cols = st.columns(min(len(followups), 3))
        for i, q in enumerate(followups[:3]):
            if cols[i].button(q, key=f"fu_{msg_key}_{i}", use_container_width=True):
                st.session_state["queued_prompt"] = q
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Action row (copy answer) — only on the latest assistant turn
    if is_latest:
        st.markdown('<div class="action-row">', unsafe_allow_html=True)
        a_cols = st.columns([1, 7])
        if a_cols[0].button("📋 Copy", key=f"copy_{msg_key}"):
            st.session_state["copy_target"] = content
        st.markdown('</div>', unsafe_allow_html=True)

        # Show copy buffer in a code block when requested
        if st.session_state.get("copy_target") == content:
            with st.expander("📋 Copy this answer", expanded=True):
                st.code(content, language="markdown")

    # Citation cards (collapsed by default to keep the answer clean)
    if citations:
        with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
            cards = []
            for c in citations:
                tag = html_lib.escape(str(c.get("tag", "")))
                src = html_lib.escape(str(c.get("source", "?")))
                page = html_lib.escape(str(c.get("page", "?")))
                snippet = html_lib.escape((c.get("snippet") or "")[:240])
                cards.append(
                    f'<div class="cite-card">'
                    f'  <div class="cite-head">'
                    f'    <span class="cite-tag">{tag}</span>'
                    f'    <span class="cite-source" title="{src}">{src}</span>'
                    f'    <span class="cite-page">p.{page}</span>'
                    f'  </div>'
                    f'  <div class="cite-snippet">{snippet}…</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div class="cite-grid">{"".join(cards)}</div>',
                unsafe_allow_html=True,
            )




# ---------- Render chat history ----------
thread = st.session_state.thread
# Index of the last assistant message — only that one shows action buttons (copy/regenerate)
last_asst_idx = max(
    (i for i, m in enumerate(thread["messages"]) if m["role"] == "assistant"),
    default=-1,
)
for idx, msg in enumerate(thread["messages"]):
    render_message(
        msg["role"],
        msg["content"],
        msg.get("citations"),
        msg.get("verification"),
        msg_key=f"hist_{idx}",
        is_latest=(idx == last_asst_idx),
    )


# ============================================================
# Chat input + streaming
# ============================================================
chat_prompt = st.chat_input("Ask anything about your documents…")

# A click on a follow-up question feeds back through here
queued = st.session_state.pop("queued_prompt", None)
prompt = chat_prompt or queued

if prompt:
    # Persist user message
    append_message(thread, "user", prompt)
    render_message("user", prompt, msg_key=f"live_user_{len(thread['messages'])}")

    # Slots for the streaming response
    steps_placeholder = st.empty()
    placeholder = st.empty()

    # Track pipeline step state for the live step chips
    step_state = {
        "rewrite": "pending",
        "retrieve": "pending",
        "rerank": "pending",
        "compress": "pending",
        "generate": "pending",
        "verify": "pending",
    }

    STEP_ORDER = ["rewrite", "retrieve", "rerank", "compress", "generate", "verify"]
    STEP_LABELS = {
        "rewrite": "🔁 Rewrite",
        "retrieve": "🔎 Hybrid retrieval",
        "rerank": "🎯 Rerank",
        "compress": "📦 Compress",
        "generate": "✍️ Generate",
        "verify": "✅ Verify",
    }

    def steps_html():
        chips = "".join(
            f'<span class="step-chip {step_state[s]}">{STEP_LABELS[s]}</span>'
            for s in STEP_ORDER
        )
        return f'<div class="steps">{chips}</div>'

    full_text = ""
    citations: list[dict] = []
    verification: dict = {}
    history_for_pipeline = thread["messages"][:-1]  # exclude the just-added user msg

    try:
        for ev in stream_pipeline(prompt, history=history_for_pipeline):
            et = ev["type"]

            if et == "step":
                name = ev["name"]
                if name in STEP_ORDER:
                    i = STEP_ORDER.index(name)
                    for j, s in enumerate(STEP_ORDER):
                        if j < i:
                            step_state[s] = "done"
                        elif j == i:
                            step_state[s] = "active"
                    steps_placeholder.markdown(steps_html(), unsafe_allow_html=True)

            elif et == "token":
                full_text += ev["content"]
                # Strip BOTH mermaid and dot blocks from the live preview
                from frontend.diagrams import MERMAID_RE, DOT_RE
                preview_text = MERMAID_RE.sub(
                    '<div class="step-chip active">📊 diagram pending…</div>',
                    full_text,
                )
                preview_text = DOT_RE.sub(
                    '<div class="step-chip active">📊 diagram pending…</div>',
                    preview_text,
                )
                placeholder.markdown(
                    f'<div class="chat-msg bot"><div class="avatar bot">AI</div>'
                    f'<div class="msg-body"><div class="msg-role">Assistant</div>'
                    f'<div class="msg-content">{render_answer_html(preview_text)}'
                    f'<span class="cite-marker">▍</span></div></div></div>',
                    unsafe_allow_html=True,
                )

            elif et == "final":
                citations = ev.get("citations", [])
                verification = ev.get("verification", {})
                for s in step_state:
                    step_state[s] = "done"
                steps_placeholder.markdown(steps_html(), unsafe_allow_html=True)

    except Exception as e:
        import traceback
        placeholder.empty()
        st.error(f"Pipeline error: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

    # Clear the streaming placeholder, then render the final message cleanly
    placeholder.empty()
    render_message(
        "assistant", full_text, citations, verification,
        msg_key=f"live_asst_{len(thread['messages'])}",
        is_latest=True,
    )

    # Persist assistant message to thread (include verification so it survives reload)
    append_message(thread, "assistant", full_text, citations)
    if verification:
        thread["messages"][-1]["verification"] = verification
    save_thread(thread)