import time

from pinecone import Pinecone, ServerlessSpec

from rag.config import DIMENSION, EMBEDDING_MODEL, Settings
from rag.models import Chunk, Hit, RagError


class VectorStore:
    def __init__(self, settings: Settings, client=None, ready_timeout=90):
        try:
            self.client = client or Pinecone(api_key=settings.pinecone_key)
            if not self.client.has_index(settings.index):
                try:
                    self.client.create_index(
                        name=settings.index,
                        dimension=DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
                        timeout=-1,  # Readiness is polled below with a bounded deadline.
                    )
                except Exception as exc:
                    # Another session may have created the same index concurrently.
                    if getattr(exc, "status", None) != 409:
                        raise
            deadline = time.monotonic() + ready_timeout
            while True:
                description = self.client.describe_index(settings.index)
                if description.dimension != DIMENSION or description.metric != "cosine":
                    raise RagError(
                        "Pinecone index must have dimension 384 and cosine metric. Choose a new index name."
                    )
                if description.status["ready"]:
                    break
                if time.monotonic() >= deadline:
                    raise RagError("Pinecone index is still initializing. Retry shortly.")
                time.sleep(1)
            self.index = self.client.Index(host=description.host)
        except RagError:
            raise
        except Exception as exc:
            raise RagError("Pinecone connection failed. Check your key, index, region, and network.") from exc

    def upsert(self, namespace: str, chunks: list[Chunk], vectors: list[list[float]]):
        if len(chunks) != len(vectors) or any(len(v) != DIMENSION for v in vectors):
            raise RagError("Vector count or dimension does not match the document chunks.")
        try:
            for offset in range(0, len(chunks), 64):
                batch = [
                    {
                        "id": chunk.id,
                        "values": vector,
                        "metadata": {**chunk.metadata(), "embedding_model": EMBEDDING_MODEL},
                    }
                    for chunk, vector in zip(chunks[offset : offset + 64], vectors[offset : offset + 64])
                ]
                self.index.upsert(vectors=batch, namespace=namespace, _request_timeout=30)
        except Exception as exc:
            raise RagError(
                "Pinecone upload failed. This collection was not activated; retry indexing."
            ) from exc

    def wait_visible(self, namespace: str, expected: int, timeout=45):
        """Fresh namespaces contain only the current ingestion's unique chunks."""
        deadline = time.monotonic() + timeout
        try:
            while True:
                stats = self.index.describe_index_stats(_request_timeout=15)
                entry = stats.namespaces.get(namespace)
                count = entry.vector_count if entry else 0
                if count >= expected:
                    return
                if time.monotonic() >= deadline:
                    raise RagError(
                        "Pinecone has not made the full collection visible yet. Retry indexing shortly."
                    )
                time.sleep(1)
        except RagError:
            raise
        except Exception as exc:
            raise RagError("Could not verify Pinecone indexing. Retry shortly.") from exc

    def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int,
        threshold: float,
        document_ids: list[str],
        page: int | None = None,
    ) -> list[Hit]:
        if not document_ids:
            return []
        filters = [{"document_id": {"$in": document_ids}}, {"embedding_model": {"$eq": EMBEDDING_MODEL}}]
        if page is not None:
            filters.append({"page": {"$eq": page}})
        try:
            response = self.index.query(
                namespace=namespace,
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False,
                filter={"$and": filters},
                _request_timeout=30,
            )
            hits = []
            for match in response.matches:
                if float(match.score) < threshold:
                    continue
                m = match.metadata
                chunk = Chunk(m["chunk_id"], m["document_id"], m["document_name"], int(m["page"]), m["text"])
                hits.append(Hit(chunk, float(match.score)))
            return hits
        except Exception as exc:
            raise RagError("Pinecone retrieval failed. Check connectivity and retry your question.") from exc

    def delete_namespace(self, namespace: str):
        try:
            self.index.delete(delete_all=True, namespace=namespace, _request_timeout=30)
        except Exception as exc:
            raise RagError(
                "Could not delete this session's collection from Pinecone. Retry cleanup."
            ) from exc
