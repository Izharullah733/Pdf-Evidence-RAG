import io
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from pypdf import PdfWriter
from reportlab.pdfgen.canvas import Canvas

from rag.chunker import chunk_document
from rag.config import DIMENSION, Settings
from rag.generator import Generator, validate_claims, validate_evidence
from rag.loader import MAX_PDF_BYTES, clean_text, load_pdf
from rag.models import FALLBACK, Answer, Chunk, Document, Hit, Page, RagError
from rag.service import Collection, RagService, is_overview
from rag.vector_store import VectorStore


class Tokenizer:
    def __call__(self, text, **kwargs):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"offset_mapping": offsets, "input_ids": list(range(len(offsets)))}


def pdf_bytes(pages):
    buffer = io.BytesIO()
    canvas = Canvas(buffer)
    for text in pages:
        canvas.drawString(50, 750, text)
        canvas.showPage()
    canvas.save()
    return buffer.getvalue()


@pytest.fixture
def hit():
    return Hit(
        Chunk("chunk-1", "doc-1", "manual.pdf", 2, "The support desk opens at 09:00. It closes at 17:00."),
        0.83,
    )


def test_pdf_pages_and_cleanup():
    doc = load_pdf(pdf_bytes(["First page", "", "Third page"]), "sample.pdf")
    assert [p.number for p in doc.pages] == [1, 3]
    assert doc.total_pages == 3
    assert clean_text("hy-\nphen\x00  test\n text") == "hyphen test text"


@pytest.mark.parametrize("data,name", [(b"hello", "x.pdf"), (b"%PDF-invalid", "x.pdf"), (b"%PDF-", "x.txt")])
def test_invalid_pdf(data, name):
    with pytest.raises(RagError):
        load_pdf(data, name)


def test_oversize_pdf():
    with pytest.raises(RagError, match="20 MB"):
        load_pdf(b"%PDF-" + b"0" * MAX_PDF_BYTES, "large.pdf")


def test_empty_and_encrypted_pdf():
    with pytest.raises(RagError, match="no extractable text"):
        load_pdf(pdf_bytes([""]), "blank.pdf")
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(RagError, match="password-protected"):
        load_pdf(buffer.getvalue(), "encrypted.pdf")


def test_chunk_coverage_overlap_and_page_boundaries():
    text = " ".join(f"word{i}." if i % 19 == 0 else f"word{i}" for i in range(130))
    doc = Document("id", "doc.pdf", [Page(1, text), Page(2, "Another page.")], 2)
    chunks = chunk_document(doc, Tokenizer(), size=40, overlap=8)
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks == chunk_document(doc, Tokenizer(), size=40, overlap=8)
    page_one = [c for c in chunks if c.page == 1]
    assert all(len(c.text.split()) <= 40 for c in chunks)
    assert set(text.split()) == {word for c in page_one for word in c.text.split()}
    assert all(a.text.split()[-8:] == b.text.split()[:8] for a, b in zip(page_one, page_one[1:]))
    assert chunks[-1].text == "Another page."


def test_bad_chunk_settings():
    with pytest.raises(RagError):
        chunk_document(Document("id", "d", [], 0), Tokenizer(), 32, 32)


def test_evidence_validation(hit):
    data = {"answerable": True, "evidence": [{"source_id": "S1", "quote": "It closes at 17:00."}]}
    evidence = validate_evidence(data, [hit])
    assert Answer(evidence).text == "It closes at 17:00. [1]"
    assert evidence[0].source.chunk.page == 2


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"answerable": False},
        {"answerable": True, "evidence": []},
        {"answerable": True, "evidence": [{"source_id": "S2", "quote": "It closes at 17:00."}]},
        {"answerable": True, "evidence": [{"source_id": "S1", "quote": "It closes at 18:00."}]},
        {"answerable": True, "evidence": [{"source_id": [], "quote": "x"}]},
        {"answerable": True, "evidence": [{"source_id": "S1", "quote": " "}]},
    ],
)
def test_unsupported_evidence_abstains(hit, payload):
    assert Answer(validate_evidence(payload, [hit])).text == FALLBACK


def test_one_fabricated_quote_rejects_entire_response(hit):
    assert not validate_evidence(
        {
            "answerable": True,
            "evidence": [
                {"source_id": "S1", "quote": "It closes at 17:00."},
                {"source_id": "S1", "quote": "Open every Sunday."},
            ],
        },
        [hit],
    )


def test_abbreviated_quotes_expand_to_original_text(hit):
    evidence = validate_evidence(
        {
            "answerable": True,
            "evidence": [{"source_id": "S1", "quote": "The support desk...It closes at 17:00."}],
        },
        [hit],
    )
    assert evidence[0].quote == hit.chunk.text
    assert not validate_evidence(
        {
            "answerable": True,
            "evidence": [{"source_id": "S1", "quote": "It closes at 17:00....The support desk"}],
        },
        [hit],
    )
    assert not validate_evidence(
        {
            "answerable": True,
            "evidence": [{"source_id": "S1", "quote": "The support desk...It closes at 18:00."}],
        },
        [hit],
    )


def store_with_mock(**kwargs):
    client = Mock()
    client.has_index.return_value = False
    client.describe_index.return_value = SimpleNamespace(
        dimension=DIMENSION, metric="cosine", status={"ready": True}, host="host"
    )
    store = VectorStore(Settings("key", "key"), client=client, **kwargs)
    return store, client


def test_index_creation_and_metadata_filters(hit):
    store, client = store_with_mock()
    assert client.create_index.call_args.kwargs["metric"] == "cosine"
    assert client.create_index.call_args.kwargs["dimension"] == 384
    store.upsert("test-ns", [hit.chunk], [[0.0] * DIMENSION])
    sent = store.index.upsert.call_args.kwargs
    assert sent["namespace"] == "test-ns"
    assert sent["vectors"][0]["metadata"]["page"] == 2
    store.index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(score=0.83, metadata=hit.chunk.metadata()),
            SimpleNamespace(score=0.2, metadata=hit.chunk.metadata()),
        ]
    )
    assert store.query("test-ns", [0.0] * DIMENSION, 4, 0.5, ["doc-1"], 2) == [hit]
    args = store.index.query.call_args.kwargs
    assert args["top_k"] == 4 and args["include_metadata"]
    assert args["filter"]["$and"][0] == {"document_id": {"$in": ["doc-1"]}}
    assert args["filter"]["$and"][-1] == {"page": {"$eq": 2}}
    assert store.query("test-ns", [], 4, 0.5, []) == []


def test_index_dimension_mismatch():
    client = Mock()
    client.describe_index.return_value = SimpleNamespace(dimension=1536, metric="cosine")
    with pytest.raises(RagError, match="dimension 384"):
        VectorStore(Settings("key", "key"), client=client)


def test_real_pinecone_sdk_request_contract(hit):
    """Build actual SDK requests with transport mocked; catch invalid SDK kwargs."""
    from pinecone import Pinecone
    from pinecone.core.openapi.db_data.models import QueryResponse, ScoredVector

    store, _ = store_with_mock()
    index = Pinecone(api_key="test-key").Index(host="https://example.invalid")
    transport = Mock()
    transport.query_vectors.return_value = QueryResponse(
        matches=[ScoredVector(id=hit.chunk.id, score=hit.score, metadata=hit.chunk.metadata())]
    )
    index._vector_api = transport
    store.index = index
    try:
        store.upsert("sdk-contract", [hit.chunk], [[0.0] * DIMENSION])
        assert transport.upsert_vectors.call_args.kwargs["_request_timeout"] == 30
        hits = store.query("sdk-contract", [0.0] * DIMENSION, 5, 0.3, ["doc-1"], 2)
        assert hits == [hit]
        request = transport.query_vectors.call_args.args[0]
        assert request.namespace == "sdk-contract"
        assert "timeout" not in request.to_dict()
    finally:
        index.close()


def test_pinecone_connection_failure():
    client = Mock()
    client.has_index.side_effect = RuntimeError("private service error")
    with pytest.raises(RagError, match="connection failed"):
        VectorStore(Settings("key", "key"), client=client)


def test_visibility_timeout_and_query_failure():
    store, _ = store_with_mock()
    store.index.describe_index_stats.return_value = SimpleNamespace(namespaces={})
    with pytest.raises(RagError, match="not made the full collection visible"):
        store.wait_visible("ns", 10, timeout=0)
    store.index.query.side_effect = RuntimeError("network")
    with pytest.raises(RagError, match="retrieval failed"):
        store.query("ns", [], 5, 0.3, ["doc"])


def backend():
    embedder = Mock(tokenizer=Tokenizer())
    embedder.encode.side_effect = lambda texts: [[0.0] * DIMENSION for _ in texts]
    return RagService(embedder, Mock(), Mock())


def test_ingestion_dedup_and_failed_staging():
    service = backend()
    data = pdf_bytes(["A document for testing."])
    collection = service.ingest([("a.pdf", data), ("alias.pdf", data)])
    assert len(collection.documents) == 1
    assert collection.chunk_count == 1
    assert not service.pending_cleanup
    service.store.upsert.side_effect = RagError("failed")
    with pytest.raises(RagError):
        service.ingest([("a.pdf", data)])
    assert collection.namespace not in service.pending_cleanup
    assert len(service.pending_cleanup) == 1
    service.cleanup()
    assert not service.pending_cleanup


def test_empty_question_never_calls_api():
    service = backend()
    collection = Collection("ns", [], 0, 0)
    with pytest.raises(RagError, match="Enter a question"):
        service.ask(collection, "   ")
    service.store.query.assert_not_called()


def test_no_hits_skips_llm(monkeypatch):
    request = Mock()
    monkeypatch.setattr(httpx, "post", request)
    assert Generator("key", "model").generate("unknown", []) == []
    request.assert_not_called()


def test_service_answer_and_document_scope(hit):
    service = backend()
    collection = Collection("ns", [Document("doc-1", "manual.pdf", [Page(2, hit.chunk.text)], 2)], 1, 0)
    service.store.query.return_value = [hit]
    service.generator.answer.return_value = Answer(
        validate_evidence(
            {"answerable": True, "evidence": [{"source_id": "S1", "quote": "It closes at 17:00."}]}, [hit]
        )
    )
    answer = service.ask(collection, "When does it close?", document_ids=["doc-1"], page=2, mode="strict")
    assert answer.text == "It closes at 17:00. [1]"
    assert answer.retrieved == [hit]
    with pytest.raises(RagError, match="active collection"):
        service.ask(collection, "When?", document_ids=["other-document"])


def test_successful_llm_request(monkeypatch, hit):
    import json

    response = Mock()
    response.json.return_value = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "answerable": True,
                            "evidence": [{"source_id": "S1", "quote": "It closes at 17:00."}],
                        }
                    )
                },
            }
        ]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(httpx, "post", post)
    result = Generator("key", "model").generate("When does it close?", [hit])
    assert result[0].quote == "It closes at 17:00."
    request = post.call_args.kwargs["json"]
    assert request["response_format"] == {"type": "json_object"}
    context = json.loads(request["messages"][1]["content"])
    assert context["sources"][0]["text"] == hit.chunk.text


@pytest.mark.parametrize("content,reason", [("not json", "stop"), ("{}", "length")])
def test_bad_llm_output_abstains(monkeypatch, hit, content, reason):
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": content}, "finish_reason": reason}]}
    monkeypatch.setattr(httpx, "post", Mock(return_value=response))
    assert Generator("key", "model").generate("When?", [hit]) == []


def test_llm_network_error_is_not_missing_evidence(monkeypatch, hit):
    monkeypatch.setattr(httpx, "post", Mock(side_effect=httpx.ConnectError("offline")))
    with pytest.raises(RagError, match="Groq request failed"):
        Generator("key", "model").generate("When?", [hit])


def test_settings_missing_secrets():
    with pytest.raises(RagError, match="PINECONE_API_KEY, GROQ_API_KEY"):
        Settings("", "").validate()


def test_streamlit_initial_screen():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception
    assert any("Ask better questions" in item.value for item in app.markdown)
    assert app.button[0].disabled


def test_streamlit_question_flow(hit):
    from streamlit.testing.v1 import AppTest

    evidence = validate_evidence(
        {"answerable": True, "evidence": [{"source_id": "S1", "quote": "It closes at 17:00."}]}, [hit]
    )
    service = Mock()
    service.pending_cleanup = set()
    from rag.models import Claim

    service.ask.return_value = Answer(evidence, [hit], 0.1, claims=[Claim("The desk closes at 5 pm.", [1])])
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.session_state["service"] = service
    app.session_state["collection"] = Collection(
        "test-session", [Document("doc-1", "manual.pdf", [Page(2, hit.chunk.text)], 2)], 1, 0
    )
    app.run(timeout=20)
    app.text_area[0].input("When does the support desk close?")
    next(button for button in app.button if button.label == "Find answer").click().run()
    assert not app.exception
    assert any("It closes at 17:00." in item.value for item in app.text)
    assert any("The desk closes at 5 pm." in item.value for item in app.markdown)
    assert any("Page 2" in item.value for item in app.caption)
    assert len(app.session_state["history"]) == 1


@pytest.mark.parametrize(
    "question",
    [
        "what is in the document , just summarize this",
        "Summarize this document",
        "What is this PDF about?",
        "What are the main topics?",
        "Is document ka khulasa batao",
    ],
)
def test_overview_routing(question):
    assert is_overview(question)


def test_specific_question_is_not_overview():
    assert not is_overview("What is the support desk closing time?")


def test_summary_reads_all_selected_pages_and_respects_scope(hit):
    service = backend()
    documents = [
        Document("doc-1", "manual.pdf", [Page(1, "First topic."), Page(2, hit.chunk.text)], 2),
        Document("doc-2", "private.pdf", [Page(1, "Unselected document.")], 1),
    ]
    collection = Collection("ns", documents, 3, 0)
    service.generator.answer.return_value = Answer()
    result = service.ask(collection, "Summarize this document", threshold=1.0, document_ids=["doc-1"])
    sent = service.generator.answer.call_args.args[1]
    assert {h.chunk.page for h in sent} == {1, 2}
    assert {h.chunk.document_id for h in sent} == {"doc-1"}
    assert all(h.score is None for h in sent)
    assert result.mode == "document overview"
    service.store.query.assert_not_called()
    service.ask(collection, "Summarize this document", document_ids=["doc-1"], page=2)
    assert {h.chunk.page for h in service.generator.answer.call_args.args[1]} == {2}


def test_automatic_reads_short_document_when_search_has_no_hits(hit):
    service = backend()
    service.store.query.return_value = []
    service.generator.answer.return_value = Answer()
    collection = Collection("ns", [Document("doc-1", "manual.pdf", [Page(2, hit.chunk.text)], 2)], 1, 0)
    result = service.ask(collection, "Support desk kab band hota hai?")
    assert result.mode == "full selected text"
    assert service.generator.answer.call_args.args[1][0].chunk.text == hit.chunk.text
    service.store.query.assert_called_once()
    service.generator.search_query.assert_not_called()


def test_strict_search_does_not_expand(hit):
    service = backend()
    service.store.query.return_value = []
    service.generator.answer.return_value = Answer()
    collection = Collection("ns", [Document("doc-1", "manual.pdf", [Page(2, hit.chunk.text)], 2)], 1, 0)
    result = service.ask(collection, "When does it close?", mode="strict")
    assert result.text == FALLBACK and result.retrieved == []
    assert service.generator.answer.call_args.args[1] == []
    service.generator.search_query.assert_not_called()


def test_claims_require_real_source_quotes(hit):
    payload = {
        "answerable": True,
        "claims": [
            {
                "text": "The desk closes at 5 pm.",
                "evidence": [{"source_id": "S1", "quote": "It closes at 17:00."}],
            }
        ],
    }
    answer = validate_claims(payload, [hit])
    assert answer.text == "The desk closes at 5 pm. [1]"
    payload["claims"][0]["evidence"][0]["quote"] = "It closes at 18:00."
    assert validate_claims(payload, [hit]).text == FALLBACK


@pytest.mark.parametrize(
    "supported,expected",
    [([True], True), ([False], False), ([True, True], False), (["true"], False), (None, False)],
)
def test_generated_claims_must_pass_support_check(hit, supported, expected):
    generator = Generator("key", "model")
    generator._request = Mock(
        side_effect=[
            {
                "answerable": True,
                "claims": [
                    {
                        "text": "The desk closes at 5 pm.",
                        "evidence": [{"source_id": "S1", "quote": "It closes at 17:00."}],
                    }
                ],
            },
            {"supported": supported},
        ]
    )
    result = generator.answer("When does the desk close?", [hit])
    assert bool(result.evidence) is expected
    assert generator._request.call_count == 2


def test_large_summary_covers_all_batches(hit):
    service = backend()
    pages = [Page(i, (f"Topic{i} important statement. " * 100)) for i in range(1, 13)]
    collection = Collection("ns", [Document("doc-1", "large.pdf", pages, 12)], 24, 0)
    seen_pages = set()

    def select(question, batch, overview):
        from rag.models import Evidence

        seen_pages.update(h.chunk.page for h in batch)
        return [Evidence(batch[0], batch[0].chunk.text.split(".")[0] + ".")]

    service.generator.generate.side_effect = select
    service.generator.answer.return_value = Answer()
    result = service.ask(collection, "Summarize this document")
    assert seen_pages == set(range(1, 13))
    assert "all 12 selected text pages" in result.note
    assert service.generator.generate.call_count > 1


def test_ui_summary_button_calls_summary_mode():
    from streamlit.testing.v1 import AppTest

    service = Mock()
    service.pending_cleanup = set()
    service.ask.return_value = Answer(mode="document overview")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.session_state["service"] = service
    app.session_state["collection"] = Collection(
        "summary-session", [Document("doc-1", "manual.pdf", [Page(1, "A topic.")], 1)], 1, 0
    )
    app.run(timeout=20)
    next(button for button in app.button if button.label == "✦ Summarize documents").click().run()
    assert not app.exception
    assert service.ask.call_args.kwargs["mode"] == "summary"


def test_automatic_blank_page_does_not_call_generator(hit):
    service = backend()
    service.store.query.return_value = []
    collection = Collection("ns", [Document("doc-1", "manual.pdf", [Page(2, hit.chunk.text)], 2)], 1, 0)
    answer = service.ask(collection, "What time does it close?", page=1)
    assert answer.text == FALLBACK
    assert "Remove the page filter" in answer.note
    service.generator.answer.assert_not_called()


def test_large_document_expands_weak_search(hit):
    service = backend()
    pages = [Page(i, "Unrelated material. " * 300) for i in range(1, 5)] + [Page(5, hit.chunk.text)]
    collection = Collection("ns", [Document("doc-1", "manual.pdf", pages, 5)], 15, 0)
    service.store.query.side_effect = [[], [hit]]
    service.generator.search_query.return_value = "support desk closing time"
    service.generator.answer.side_effect = [Answer(), Answer()]
    answer = service.ask(collection, "Support desk kab band hota hai?")
    assert answer.mode == "expanded semantic and keyword search"
    assert service.store.query.call_count == 2
    assert service.store.query.call_args.args[3] == 0.0
    assert any("17:00" in h.chunk.text for h in service.generator.answer.call_args.args[1])


@pytest.mark.parametrize("status,pattern", [(401, "API key"), (404, "unavailable"), (429, "usage limit")])
def test_provider_errors_are_actionable(monkeypatch, status, pattern):
    response = Mock(status_code=status, headers={"retry-after": "0"})
    monkeypatch.setattr(httpx, "post", Mock(return_value=response))
    with pytest.raises(RagError, match=pattern):
        Generator("key", "model")._request("Return JSON", {})
