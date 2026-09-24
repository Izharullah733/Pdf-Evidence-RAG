from dataclasses import dataclass, field

FALLBACK = "The answer is not available in the provided document."


class RagError(Exception):
    """An actionable error safe to display in the interface."""


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Document:
    id: str
    name: str
    pages: list[Page]
    total_pages: int


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    document_name: str
    page: int
    text: str

    def metadata(self) -> dict:
        return {
            "chunk_id": self.id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page": self.page,
            "text": self.text,
        }


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float | None


@dataclass(frozen=True)
class Evidence:
    source: Hit
    quote: str


@dataclass(frozen=True)
class Claim:
    text: str
    citations: list[int]


@dataclass
class Answer:
    evidence: list[Evidence] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    claims: list[Claim] = field(default_factory=list)
    mode: str = "semantic search"
    note: str = ""

    @property
    def text(self) -> str:
        if not self.evidence:
            return FALLBACK
        if self.claims:
            return "\n\n".join(
                f"{claim.text} " + " ".join(f"[{i}]" for i in claim.citations) for claim in self.claims
            )
        return "\n\n".join(f"{e.quote} [{i}]" for i, e in enumerate(self.evidence, 1))
