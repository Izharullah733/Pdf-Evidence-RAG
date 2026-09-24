"""Local real-model smoke check. Downloads MiniLM on first use; no API keys needed."""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunker import chunk_document  # noqa: E402
from rag.embeddings import Embedder  # noqa: E402
from rag.loader import load_pdf  # noqa: E402
from scripts.make_demo_pdf import create_demo  # noqa: E402


def main():
    started = time.perf_counter()
    embedder = Embedder()
    load_seconds = time.perf_counter() - started
    pdf = create_demo()
    document = load_pdf(pdf.read_bytes(), pdf.name)
    chunks = chunk_document(document, embedder.tokenizer)
    assert all(len(embedder.tokenizer(c.text)["input_ids"]) <= 256 for c in chunks)
    started = time.perf_counter()
    vectors = embedder.encode([c.text for c in chunks])
    assert all(len(v) == 384 and abs(math.sqrt(sum(x * x for x in v)) - 1) < 0.001 for v in vectors)
    query = embedder.encode(["Who is the course coordinator?"])[0]
    scores = [sum(a * b for a, b in zip(query, vector)) for vector in vectors]
    best = max(range(len(scores)), key=scores.__getitem__)
    assert chunks[best].page == 1, "Expected the coordinator question to retrieve page 1."
    result = {
        "model_load_seconds": load_seconds,
        "encoding_seconds": time.perf_counter() - started,
        "chunks": len(chunks),
        "dimension": len(query),
        "top_page": chunks[best].page,
        "top_cosine_similarity": scores[best],
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/embedding_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
