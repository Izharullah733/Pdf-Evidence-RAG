from rag.config import DIMENSION, EMBEDDING_MODEL
from rag.models import RagError


class Embedder:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            self.tokenizer = self.model.tokenizer
            if self.model.get_sentence_embedding_dimension() != DIMENSION:
                raise ValueError("Unexpected embedding dimension")
        except Exception as exc:
            raise RagError(
                "Could not load the embedding model. Check internet access and installed dependencies."
            ) from exc

    def encode(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.model.encode(
                texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
        except Exception as exc:
            raise RagError("Embedding generation failed. Try a smaller document batch.") from exc
