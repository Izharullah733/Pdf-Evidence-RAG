from html import escape

import streamlit as st

from rag.runtime import get_runtime

runtime = get_runtime()
Settings = runtime.config.Settings
Embedder = runtime.embeddings.Embedder
Generator = runtime.generator.Generator
RagError = runtime.models.RagError
RagService = runtime.service.RagService
VectorStore = runtime.vector_store.VectorStore

st.set_page_config(page_title="PDF Evidence | Pinecone RAG", page_icon="📘", layout="wide")


def apply_dashboard_style():
    """Give the local Streamlit app a polished, responsive dashboard surface."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

        :root {
            --ink: #12233d;
            --muted: #6d7b91;
            --line: #e6ebf3;
            --blue: #2e6cf6;
            --navy: #102b5f;
            --surface: #ffffff;
            --mist: #f6f8fc;
            --mint: #e9fbf3;
        }
        .stApp { background: #f7f9fc; color: var(--ink); }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        [data-testid="stHeader"] { background: rgba(247,249,252,.88); border-bottom: 1px solid #e9edf4; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8edf4; }
        [data-testid="stSidebar"] > div:first-child { padding: 1.45rem 1rem 2rem; }
        [data-testid="stSidebar"] h2 { font-family: 'Manrope', sans-serif; font-size: .96rem; letter-spacing: -.01em; margin: 1.3rem 0 .65rem; }
        [data-testid="stSidebar"] .stCaption { color: #728096; line-height: 1.55; }
        .block-container { max-width: 1360px; padding-top: 2.15rem; padding-bottom: 2.5rem; }
        .dashboard-hero { position: relative; overflow: hidden; padding: 1.65rem 1.75rem; border-radius: 20px;
            background: linear-gradient(120deg, #0e2860 0%, #194ca5 58%, #386ff2 100%); color: white; box-shadow: 0 18px 34px rgba(23,66,147,.18); margin: 0 0 1.35rem; }
        .dashboard-hero:after { content: ''; position: absolute; width: 320px; height: 320px; border-radius: 50%;
            right: -105px; top: -190px; background: rgba(255,255,255,.10); box-shadow: -85px 155px 0 rgba(255,255,255,.055); }
        .eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; opacity: .76; margin-bottom: .45rem; }
        .hero-title { font-family: 'Manrope', sans-serif; font-size: clamp(1.8rem, 3vw, 2.55rem); font-weight: 800; line-height: 1.1; letter-spacing: -.055em; margin: 0; }
        .hero-copy { position: relative; z-index: 1; max-width: 700px; color: #dce8ff; margin: .6rem 0 0; font-size: 1rem; }
        .hero-badge { position: relative; z-index: 1; display: inline-flex; align-items: center; gap: .38rem; margin-top: 1.05rem; padding: .34rem .63rem; border: 1px solid rgba(255,255,255,.22); border-radius: 999px; background: rgba(0,0,0,.12); font-size: .76rem; color: #eff5ff; }
        .status-dot { width: 7px; height: 7px; display: inline-block; border-radius: 50%; background: #75e5bb; box-shadow: 0 0 0 3px rgba(117,229,187,.17); }
        .section-label { color: #758299; font-size: .76rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin: 1.55rem 0 .65rem; }
        div[data-testid="stMetric"] { padding: 1.1rem 1.15rem; border: 1px solid var(--line); background: #ffffff; border-radius: 14px; box-shadow: 0 5px 14px rgba(20,42,77,.035); }
        div[data-testid="stMetric"] label { color: #738197 !important; font-size: .78rem !important; font-weight: 600 !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; font-size: 1.8rem; color: var(--ink); }
        .stButton > button, .stFormSubmitButton > button { border-radius: 10px; font-weight: 700; border: 1px solid transparent; min-height: 2.65rem; transition: transform .18s ease, box-shadow .18s ease; }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: linear-gradient(135deg, #2d6cf5, #2458d4); box-shadow: 0 8px 16px rgba(44,101,226,.20); }
        .stButton > button:hover, .stFormSubmitButton > button:hover { transform: translateY(-1px); }
        .stTextArea textarea { border: 1px solid #dce4f0 !important; border-radius: 13px !important; background: #fbfcfe !important; font-size: 1rem !important; padding: .9rem !important; }
        .stTextArea textarea:focus { border-color: #4c7df3 !important; box-shadow: 0 0 0 3px rgba(46,108,246,.12) !important; }
        [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: #fff; }
        .stAlert { border-radius: 12px; }
        .answer-card { border: 1px solid #dce6fb; border-left: 4px solid #2e6cf6; border-radius: 14px; padding: 1.05rem 1.15rem; background: linear-gradient(120deg, #ffffff, #f7faff); margin: .35rem 0 .9rem; }
        .answer-claim { font-size: 1rem; line-height: 1.6; color: #1d2e4b; padding: .2rem 0; }
        .source-card { border: 1px solid #e4eaf3; background: #fff; border-radius: 11px; padding: .85rem .95rem; margin: .55rem 0; }
        .source-meta { color: #66768f; font-size: .78rem; margin-top: .42rem; }
        .quick-tip { border: 1px solid #dce9ff; background: #f4f8ff; color: #365273; border-radius: 12px; padding: .8rem 1rem; font-size: .88rem; line-height: 1.55; margin: .85rem 0 1rem; }
        .collection-chip { display: inline-flex; align-items: center; gap: .42rem; padding: .42rem .7rem; border-radius: 999px; background: var(--mint); color: #21765a; font-weight: 700; font-size: .78rem; margin: .15rem 0 .7rem; }
        .collection-chip:before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: #23a779; }
        @media (max-width: 720px) { .block-container { padding: 1rem .85rem 2rem; } .dashboard-hero { padding: 1.35rem; border-radius: 16px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def embedding_model():
    return Embedder()


def service():
    if "service" not in st.session_state:
        settings = Settings.from_env()
        settings.validate()
        st.session_state.service = RagService(
            embedding_model(), VectorStore(settings), Generator(settings.groq_key, settings.model)
        )
    else:
        # Refresh Python implementations after Streamlit hot reload without losing the uploaded collection.
        previous = st.session_state.service
        if previous.__class__.__module__ == "rag.service":
            settings = Settings.from_env()
            settings.validate()
            current = RagService(
                previous.embedder, previous.store, Generator(settings.groq_key, settings.model)
            )
            current.pending_cleanup = previous.pending_cleanup
            st.session_state.service = current
    return st.session_state.service


def render_answer(answer):
    if not answer.evidence:
        st.info(answer.text)
    else:
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        if getattr(answer, "claims", None):
            for claim in answer.claims:
                citations = " ".join(f"<b>[{i}]</b>" for i in claim.citations)
                st.markdown(
                    f'<div class="answer-claim">{escape(claim.text)} {citations}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<small>Supporting PDF excerpts</small>", unsafe_allow_html=True)
        for i, evidence in enumerate(answer.evidence, 1):
            score = evidence.source.score
            meta = (
                f"{escape(evidence.source.chunk.document_name)} &middot; Page {evidence.source.chunk.page} "
                + (f"&middot; Similarity {score:.3f}" if score is not None else "&middot; Document text")
            )
            st.markdown(
                f'<div class="source-card"><b>[{i}]</b> {escape(evidence.quote)}'
                f'<div class="source-meta">{meta}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    st.caption(
        f"{len(answer.retrieved)} source passages considered · {answer.elapsed_seconds:.2f}s "
        f"· {getattr(answer, 'mode', 'semantic search')} "
        "· Similarity measures relevance, not the probability that an answer is correct."
    )
    if getattr(answer, "note", ""):
        st.caption(answer.note)
    if answer.retrieved:
        with st.expander("Inspect retrieved passages"):
            for hit in answer.retrieved:
                st.caption(
                    f"{hit.chunk.document_name} · Page {hit.chunk.page} "
                    + (f"· {hit.score:.3f} " if hit.score is not None else "· Document text ")
                    + f"· Chunk {hit.chunk.id}"
                )
                st.text(hit.chunk.text)


apply_dashboard_style()
st.session_state.setdefault("history", [])

collection = st.session_state.get("collection")
collection_status = "Collection ready" if collection else "Ready for upload"
st.markdown(
    f"""
    <div class="dashboard-hero">
      <div class="eyebrow">Pinecone-powered document intelligence</div>
      <div class="hero-title">Ask better questions.<br>See the evidence.</div>
      <div class="hero-copy">Upload PDFs, get concise answers and summaries, and verify every important detail with page-level source references.</div>
      <div class="hero-badge"><span class="status-dot"></span>{collection_status}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Knowledge base")
    st.caption("Create a private, searchable collection from your PDFs.")
    uploads = st.file_uploader(
        "PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Up to 20 MB per PDF. Text-based PDFs; OCR is not included.",
    )
    with st.expander("Indexing options", expanded=False):
        size = st.slider("Chunk size (model tokens)", 64, 240, 180, 4)
        overlap = st.slider("Chunk overlap (tokens)", 0, min(60, size - 1), min(30, size - 1))
        st.caption("Chunk settings apply next time you index documents.")
    st.caption(
        "Indexing sends extracted text to Pinecone. Questions and retrieved passages are sent to Groq."
    )
    if st.button("Index selected PDFs", type="primary", disabled=not uploads, use_container_width=True):
        try:
            with st.spinner("Extracting, embedding, and indexing PDFs…"):
                backend = service()
                new_collection = backend.ingest(
                    [(file.name, file.getvalue()) for file in uploads], size, overlap
                )
                previous = st.session_state.get("collection")
                st.session_state.collection = new_collection
                st.session_state.history = []
                if previous:
                    backend.pending_cleanup.add(previous.namespace)
            st.success(
                f"Ready: {new_collection.chunk_count} chunks in {new_collection.ingestion_seconds:.1f}s"
            )
            try:
                backend.cleanup()
            except RagError as exc:
                st.warning(str(exc))
        except RagError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Indexing failed unexpectedly. Check the installed dependencies and try again.")

    st.divider()
    st.markdown("## Search settings")
    search_mode = st.radio(
        "Search behavior",
        ["Automatic", "Strict similarity"],
        help="Automatic also reads small documents in full and broadens weak searches. Strict uses only Pinecone matches above your threshold.",
    )
    top_k = st.slider("Top-k passages", 1, 10, 5)
    threshold = st.slider("Minimum cosine similarity", 0.0, 1.0, 0.35, 0.01)
    collection = st.session_state.get("collection")
    selected = []
    page = None
    if collection:
        labels = {d.id: f"{d.name} ({d.id[:6]})" for d in collection.documents}
        selected = st.multiselect(
            "Search documents",
            list(labels),
            default=list(labels),
            format_func=labels.get,
            key=f"docs_{collection.namespace}",
        )
        if st.checkbox("Filter by PDF page"):
            page = st.number_input(
                "Page number (1-based)",
                min_value=1,
                max_value=max(d.total_pages for d in collection.documents),
                step=1,
            )
        st.caption(f"Namespace: {collection.namespace}")
    backend = st.session_state.get("service")
    if backend and (collection or backend.pending_cleanup):
        if st.button("Delete this session's indexed data", use_container_width=True):
            if collection:
                backend.pending_cleanup.add(collection.namespace)
            try:
                backend.cleanup()
                st.session_state.pop("collection", None)
                st.session_state.history = []
                st.rerun()
            except RagError as exc:
                # Do not allow queries against a collection whose deletion was attempted.
                st.session_state.pop("collection", None)
                st.session_state.history = []
                collection = None
                st.error(str(exc))

if not collection:
    st.markdown('<div class="section-label">Get started</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="quick-tip"><b>1.</b> Upload one or more text-based PDFs in the left panel. '
        '<b>2.</b> Click <b>Index selected PDFs</b>. <b>3.</b> Ask a question, or generate a summary. '
        'Your sources will appear with each answer.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("First-time setup"):
        st.write(
            "Copy .env.example to .env and set PINECONE_API_KEY and GROQ_API_KEY. "
            "The app creates a 384-dimensional cosine index automatically. "
            "The first indexing run downloads the embedding model."
        )
else:
    st.markdown('<div class="section-label">Collection overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="collection-chip">Collection indexed and ready to search</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Documents", len(collection.documents))
    b.metric("Text pages", sum(len(d.pages) for d in collection.documents))
    c.metric("Indexed chunks", collection.chunk_count)
    d.metric("Questions this session", len(st.session_state.history))
    for document in collection.documents:
        skipped = document.total_pages - len(document.pages)
        if skipped:
            st.warning(f"{document.name}: {skipped} pages had no extractable text and were skipped.")
    st.markdown('<div class="section-label">Explore your documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="quick-tip"><b>Ask naturally.</b> Use English or Roman Urdu. '
        'Try a quick overview first, then ask about a person, rule, topic, date, requirement, or section.</div>',
        unsafe_allow_html=True,
    )
    summary_col, topics_col = st.columns(2)
    summarize_clicked = summary_col.button(
        "✦ Summarize documents", disabled=not selected, use_container_width=True
    )
    topics_clicked = topics_col.button("☰ Show main topics", disabled=not selected, use_container_width=True)
    st.caption(
        "Examples: 'Summarize this document' · 'Explain the main concepts' · 'Is document ka khulasa batao'. "
        "For a specific answer, mention the topic, person, rule, or section you mean."
    )
    st.markdown('<div class="section-label">Ask a question</div>', unsafe_allow_html=True)
    with st.form("question_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            placeholder="Summarize this document, or ask about something inside it…",
            max_chars=2000,
        )
        submitted = st.form_submit_button("Find answer", type="primary", disabled=not selected)
    if not selected:
        st.info("Select at least one document to search.")
    if submitted or summarize_clicked or topics_clicked:
        mode = "auto" if search_mode == "Automatic" else "strict"
        if summarize_clicked or topics_clicked:
            question = (
                "Summarize the selected documents in clear, concise points."
                if summarize_clicked
                else "What are the main topics covered in the selected documents?"
            )
            mode = "summary"
        try:
            with st.spinner("Reading the documents and checking the answer against sources…"):
                answer = service().ask(collection, question, top_k, threshold, selected, page, mode=mode)
            st.session_state.history.append({"question": question.strip(), "answer": answer})
            st.session_state.history = st.session_state.history[-20:]
        except RagError as exc:
            st.error(str(exc))
    if st.session_state.history:
        st.markdown('<div class="section-label">Latest answer</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="quick-tip"><b>Question:</b> {escape(st.session_state.history[-1]["question"])}</div>',
            unsafe_allow_html=True,
        )
        render_answer(st.session_state.history[-1]["answer"])
        with st.expander(f"Session history ({len(st.session_state.history)} questions)"):
            st.caption("Each question is answered independently; prior answers are never used as evidence.")
            for entry in reversed(st.session_state.history[:-1]):
                st.text(entry["question"])
                render_answer(entry["answer"])
                st.divider()
        if st.button("Clear question history"):
            st.session_state.history = []
            st.rerun()

st.divider()
st.caption(
    "Session collections remain in Pinecone after the browser closes. Use the delete button before "
    "leaving, or remove the displayed namespace in the Pinecone console."
)
