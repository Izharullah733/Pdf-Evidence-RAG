import hashlib
import re

from rag.models import Chunk, Document, RagError


def chunk_document(document: Document, tokenizer, size: int = 180, overlap: int = 30) -> list[Chunk]:
    """Split by tokenizer offsets, preferring sentence endings; never cross a page.

    Uses original text slices instead of decoding tokens, preserving source quotes.
    The UI caps the content budget below MiniLM's 256-token model limit.
    """
    if not 32 <= size <= 240 or not 0 <= overlap < size:
        raise RagError("Chunk size must be 32–240 tokens, with overlap smaller than size.")
    chunks = []
    for page in document.pages:
        offsets = tokenizer(
            page.text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )["offset_mapping"]
        start = 0
        while start < len(offsets):
            end = min(start + size, len(offsets))
            if end < len(offsets):
                # Keep at least half a window and enough room for forward progress.
                lower = start + max(size // 2, overlap + 1)
                for candidate in range(end, lower, -1):
                    if re.search(r"[.!?][\"')\]]?$", page.text[: offsets[candidate - 1][1]]):
                        end = candidate
                        break
            text = page.text[offsets[start][0] : offsets[end - 1][1]].strip()
            digest = hashlib.sha256(
                f"{document.id}:{page.number}:{start}:{size}:{overlap}".encode()
            ).hexdigest()[:32]
            chunks.append(Chunk(digest, document.id, document.name, page.number, text))
            if end == len(offsets):
                break
            start = end - overlap
    return chunks
