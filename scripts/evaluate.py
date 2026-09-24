"""Live evaluation. Requires .env keys; never substitutes mock retrieval results."""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import Settings  # noqa: E402
from rag.embeddings import Embedder  # noqa: E402
from rag.generator import Generator  # noqa: E402
from rag.models import RagError  # noqa: E402
from rag.service import RagService  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402
from scripts.make_demo_pdf import create_demo  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--mode", choices=["strict", "auto"], default="strict")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.validate()
    service = RagService(Embedder(), VectorStore(settings), Generator(settings.groq_key, settings.model))
    cases = json.loads(Path("examples/questions.json").read_text(encoding="utf-8"))
    collection = None
    try:
        pdf = create_demo()
        collection = service.ingest([(pdf.name, pdf.read_bytes())], args.chunk_size)
        rows = []
        for case in cases:
            answer = service.ask(collection, case["question"], args.top_k, args.threshold, mode=args.mode)
            pages = {h.chunk.page for h in answer.retrieved}
            answerable = case["expected"] is not None
            correct = (
                (case["expected"].casefold() in answer.text.casefold()) if answerable else not answer.evidence
            )
            rows.append(
                {
                    **case,
                    "answer": answer.text,
                    "retrieved_pages": sorted(pages),
                    "retrieved_expected_page": bool(pages.intersection(case["pages"])),
                    "answer_check_passed": correct,
                    "abstained": not answer.evidence,
                    "latency_seconds": answer.elapsed_seconds,
                }
            )
        supported = [r for r in rows if r["expected"] is not None]
        unsupported = [r for r in rows if r["expected"] is None]
        output = {
            "configuration": {
                "top_k": args.top_k,
                "threshold": args.threshold,
                "chunk_size": args.chunk_size,
                "model": settings.model,
                "mode": args.mode,
            },
            "metrics": {
                "retrieval_page_recall_at_k": statistics.mean(
                    r["retrieved_expected_page"] for r in supported
                ),
                "supported_answer_substring_rate": statistics.mean(
                    r["answer_check_passed"] for r in supported
                ),
                "unsupported_abstention_rate": statistics.mean(r["abstained"] for r in unsupported),
                "mean_latency_seconds": statistics.mean(r["latency_seconds"] for r in rows),
                "ingestion_seconds": collection.ingestion_seconds,
            },
            "cases": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output["metrics"], indent=2))
        print(f"Full results: {args.output}")
    finally:
        if collection:
            service.pending_cleanup.add(collection.namespace)
        pending = sorted(service.pending_cleanup)
        try:
            service.cleanup()
        except RagError:
            print(f"Cleanup failed. Remove these namespaces in Pinecone: {pending}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
