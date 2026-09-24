import re
import time
import uuid
from dataclasses import dataclass, replace

from rag.chunker import chunk_document
from rag.loader import load_pdf
from rag.models import Answer, Document, Hit, RagError


def is_overview(question: str) -> bool:
    text = question.casefold()
    return bool(
        re.search(
            r"\b(summari[sz]e|summary|overview|main (points|topics|concepts)|key (points|topics|ideas)|"
            r"khulasa|khulassa)\b|what (is|are|does).{0,30}(document|pdf|file).{0,15}"
            r"(about|contain|cover)|what.s (in|inside) (this|the) (document|pdf)|"
            r"what is in (this|the) (document|pdf)|document.{0,25}(kis bare|kis baray|kya hai)|"
            r"explain (this|the) (document|pdf)|(is|es) (mein|main|mai) (kya|kia) (likha|hai)|"
            r"(document|pdf|file).{0,15}\b(kya|kia)\b.{0,12}\b(hai|likha)\b",
            text,
        )
    )


def context_batches(hits: list[Hit], tokenizer) -> list[list[Hit]]:
    """Keep requests small enough for common Groq account token limits."""
    batches, batch, tokens, chars = [], [], 0, 0
    for hit in hits:
        cost = len(tokenizer(hit.chunk.text, add_special_tokens=False)["input_ids"]) + 50
        length = len(hit.chunk.text) + 150
        if batch and (tokens + cost > 2400 or chars + length > 10000):
            batches.append(batch)
            batch, tokens, chars = [], 0, 0
        batch.append(hit)
        tokens += cost
        chars += length
    if batch:
        batches.append(batch)
    return batches


@dataclass
class Collection:
    namespace: str
    documents: list[Document]
    chunk_count: int
    ingestion_seconds: float


class RagService:
    def __init__(self, embedder, store, generator):
        self.embedder, self.store, self.generator = embedder, store, generator
        self.pending_cleanup: set[str] = set()

    def ingest(self, files: list[tuple[str, bytes]], size=180, overlap=30) -> Collection:
        started = time.perf_counter()
        if not files:
            raise RagError("Upload at least one PDF before indexing.")
        documents = {}
        for name, data in files:
            document = load_pdf(data, name)
            documents.setdefault(document.id, document)
        chunks = [
            chunk
            for document in documents.values()
            for chunk in chunk_document(document, self.embedder.tokenizer, size, overlap)
        ]
        if not chunks:
            raise RagError("No text chunks could be produced from these documents.")
        namespace = f"pdf-{uuid.uuid4().hex}"
        # Stage into a fresh namespace: failed uploads never corrupt the active collection.
        self.pending_cleanup.add(namespace)
        for offset in range(0, len(chunks), 64):
            batch = chunks[offset : offset + 64]
            self.store.upsert(namespace, batch, self.embedder.encode([c.text for c in batch]))
        self.store.wait_visible(namespace, len(chunks))
        self.pending_cleanup.discard(namespace)
        return Collection(namespace, list(documents.values()), len(chunks), time.perf_counter() - started)

    def cleanup(self):
        for namespace in list(self.pending_cleanup):
            self.store.delete_namespace(namespace)
            self.pending_cleanup.remove(namespace)

    def ask(
        self,
        collection: Collection,
        question: str,
        top_k=5,
        threshold=0.35,
        document_ids: list[str] | None = None,
        page: int | None = None,
        mode: str = "auto",
    ) -> Answer:
        question = question.strip()
        if not question:
            raise RagError("Enter a question before searching.")
        if len(question) > 2000:
            raise RagError("Please keep questions under 2,000 characters.")
        tokens = self.embedder.tokenizer(question, add_special_tokens=True)["input_ids"]
        if len(tokens) > 256:
            raise RagError("Please shorten the question to 256 model tokens or fewer.")
        if not 1 <= top_k <= 10 or not 0 <= threshold <= 1:
            raise RagError("Top-k must be 1–10 and the similarity threshold must be 0–1.")
        if page is not None and page < 1:
            raise RagError("Page numbers start at 1.")
        if mode not in ("auto", "summary", "strict"):
            raise RagError("Choose automatic search, summary, or strict search.")
        allowed = {d.id for d in collection.documents}
        selected = list(allowed) if document_ids is None else document_ids
        if not set(selected).issubset(allowed):
            raise RagError("Select documents from the active collection.")
        if not selected:
            raise RagError("Select at least one document to search.")
        started = time.perf_counter()
        if mode == "summary" or (mode == "auto" and is_overview(question)):
            answer = self.summarize(collection, question, selected, page)
            answer.elapsed_seconds = time.perf_counter() - started
            return answer
        hits = self.store.query(
            collection.namespace, self.embedder.encode([question])[0], top_k, threshold, selected, page
        )
        route = "Pinecone semantic search"
        note = ""
        if mode == "auto":
            full_context = self.document_context(collection, selected, page)
            if not full_context:
                return Answer(
                    elapsed_seconds=time.perf_counter() - started,
                    note="No extracted text exists for this document/page selection. Remove the page filter and retry.",
                )
            batches = context_batches(full_context, self.embedder.tokenizer)
            if len(batches) == 1:
                note = f"Pinecone found {len(hits)} matches above the threshold. All selected text was also read for completeness."
                hits, route = full_context, "full selected text"
        answer = self.generator.answer(question, hits)
        if mode == "auto" and route != "full selected text" and not answer.evidence:
            rewritten = self.generator.search_query(question)
            broader = self.store.query(
                collection.namespace, self.embedder.encode([rewritten])[0], max(top_k, 8), 0.0, selected, page
            )
            # Literal names/terms can be missed by the English semantic model.
            words = set(re.findall(r"\w{3,}", rewritten.casefold())) - {
                "the",
                "and",
                "what",
                "who",
                "how",
                "when",
                "where",
                "document",
                "pdf",
                "this",
                "that",
                "does",
                "are",
                "for",
                "from",
                "about",
                "with",
                "explain",
                "please",
            }
            ranked = sorted(
                full_context,
                key=lambda h: len(words & set(re.findall(r"\w{3,}", h.chunk.text.casefold()))),
                reverse=True,
            )
            literal = [h for h in ranked[:3] if words & set(re.findall(r"\w{3,}", h.chunk.text.casefold()))]
            # Include both literal and semantic candidates within the same bounded context.
            combined, seen = [], set()
            for hit in literal + broader:
                key = (hit.chunk.document_id, hit.chunk.page, hit.chunk.text)
                if key not in seen:
                    combined.append(hit)
                    seen.add(key)
            hits = context_batches(combined, self.embedder.tokenizer)
            hits = hits[0] if hits else []
            answer = self.generator.answer(question, hits)
            route = "expanded semantic and keyword search"
            note = "Automatic search expanded beyond the similarity threshold and rephrased the search. Source-support checks still apply."
        answer.retrieved = hits
        answer.mode = route
        answer.note = " ".join(part for part in (note, answer.note) if part)
        answer.elapsed_seconds = time.perf_counter() - started
        return answer

    def document_context(self, collection, selected, page=None):
        hits = []
        for document in collection.documents:
            if document.id not in selected:
                continue
            scoped = replace(document, pages=[p for p in document.pages if page is None or p.number == page])
            hits.extend(
                Hit(chunk, None)
                for chunk in chunk_document(scoped, self.embedder.tokenizer, size=240, overlap=0)
            )
        return hits

    def summarize(self, collection, question, selected, page=None):
        hits = self.document_context(collection, selected, page)
        page_count = len({(hit.chunk.document_id, hit.chunk.page) for hit in hits})
        if not hits:
            return Answer(
                mode="document overview", note="No extracted text exists for the selected documents/page."
            )
        batches = context_batches(hits, self.embedder.tokenizer)
        if len(batches) > 24:
            raise RagError(
                "This selection is too large for one overview. Select one document or a page, then summarize again."
            )
        rounds = 0
        while len(batches) > 1:
            compressed = []
            for batch in batches:
                pieces = self.generator.generate(question, batch, overview=True)
                if not pieces:
                    raise RagError(
                        "A document section could not be summarized reliably. Retry or select a smaller document/page."
                    )
                compressed.extend(Hit(replace(e.source.chunk, text=e.quote), None) for e in pieces)
            hits = compressed
            batches = context_batches(hits, self.embedder.tokenizer)
            rounds += 1
            if rounds >= 3 and len(batches) > 1:
                raise RagError("The overview is too long. Select fewer documents or a page and retry.")
        answer = self.generator.answer(question, hits)
        answer.retrieved = hits
        answer.mode = "document overview"
        answer.note = (
            f"Reviewed all {page_count} selected text pages. "
            + ("Long text was condensed in batches before composing this overview. " if rounds else "")
            + answer.note
        )
        return answer
