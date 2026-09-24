import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from rag.models import RagError

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384


@dataclass(frozen=True)
class Settings:
    pinecone_key: str = field(repr=False)
    groq_key: str = field(repr=False)
    index: str = "pdf-rag-minilm-v1"
    cloud: str = "aws"
    region: str = "us-east-1"
    model: str = "qwen/qwen3.8-27b"

    @classmethod
    def from_env(cls):
        load_dotenv(override=True)
        return cls(
            os.getenv("PINECONE_API_KEY", "").strip(),
            os.getenv("GROQ_API_KEY", "").strip(),
            os.getenv("PINECONE_INDEX", "pdf-rag-minilm-v1"),
            os.getenv("PINECONE_CLOUD", "aws"),
            os.getenv("PINECONE_REGION", "us-east-1"),
            os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        )

    def validate(self):
        missing = [
            name
            for name, value in (("PINECONE_API_KEY", self.pinecone_key), ("GROQ_API_KEY", self.groq_key))
            if not value
        ]
        if missing:
            raise RagError("Set " + ", ".join(missing) + " in your .env file, then retry.")
