"""A small live regression check for summaries, direct questions, and abstention.

Uses only the included fictional PDF; never prints keys or provider error bodies.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import Settings  # noqa: E402
from rag.embeddings import Embedder  # noqa: E402
from rag.generator import Generator  # noqa: E402
from rag.models import RagError  # noqa: E402
from rag.service import RagService  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402


def main():
    config = Settings.from_env()
    config.validate()
    backend = RagService(Embedder(), VectorStore(config), Generator(config.groq_key, config.model))
    collection = None
    rows = []
    cases = [
        ("what is in the document , just summarize this", True),
        ("Who is the course coordinator?", True),
        ("Is course ki muddat kitni hai? Roman Urdu mein batao.", True),
        ("What is the lab Wi-Fi password?", False),
    ]
    try:
        path = Path("examples/course_guide.pdf")
        collection = backend.ingest([(path.name, path.read_bytes())])
        print("Sample PDF indexed in a temporary namespace.", flush=True)
        for question, should_answer in cases:
            answer = backend.ask(collection, question)
            row = {
                "question": question,
                "answer": answer.text,
                "mode": answer.mode,
                "source_count": len(answer.evidence),
                "seconds": round(answer.elapsed_seconds, 2),
                "passed": bool(answer.evidence) == should_answer,
                "note": answer.note,
            }
            if "coordinator" in question:
                row["passed"] = row["passed"] and "Sara Malik" in answer.text
            if "muddat" in question:
                row["passed"] = row["passed"] and any(
                    t in answer.text.casefold() for t in ("eight", "8", "aath", "ath", "aat")
                )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/live_answers.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not all(row["passed"] for row in rows):
            raise RagError("One or more live answer checks failed; inspect artifacts/live_answers.json.")
    finally:
        if collection:
            backend.pending_cleanup.add(collection.namespace)
        backend.cleanup()
        print("Temporary test vectors cleaned up.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RagError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
