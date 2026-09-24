# Evaluation procedure

The supplied ten-question fixture has six answerable questions, three missing-information questions, and one adversarial instruction. It is a smoke benchmark, not a representative accuracy study.

Run `python scripts/evaluate.py` after configuring credentials. The script generates the sample PDF, loads the actual embedding model, creates/validates the Pinecone index, indexes a temporary namespace, asks Groq for evidence, and attempts namespace deletion in a `finally` block. Failures abort evaluation rather than recording fabricated answers. A cleanup error prints the namespace IDs for manual removal.

For comparison, run separate outputs with `--top-k 3 --threshold 0.25`, `--top-k 5 --threshold 0.35`, and `--top-k 8 --threshold 0.50`. Vary `--chunk-size` between 96, 180, and 240. Keep documents and model fixed. Use `--output artifacts/run_name.json` to preserve each run.

| Metric | Definition | Interpretation |
|---|---|---|
| Retrieval page recall@k | Answerable questions retrieving at least one gold page / answerable questions | Page-level, not claim-level, recall |
| Supported answer substring rate | Answers containing expected answer text / answerable questions | Simple automated check; manually review surrounding qualifications |
| Unsupported abstention rate | Missing/adversarial questions receiving fallback / unsupported questions | Higher is better on this fixture |
| Mean query latency | Mean wall time for query embedding + search + generation + validation | Excludes model initialization and ingestion |
| Ingestion time | PDF extraction through vector visibility | Includes embedding; excludes model initialization |

Also manually check source page accuracy, whether quotes directly answer the question, and whether negations, numbers, units, and exceptions remain intact. A verbatim answer can still be misleading when context is omitted. Never interpret the cosine score as an answer-correctness percentage.

Extend the dataset with at least 30–50 questions from real course PDFs: direct facts, section-level explanations, tables, similarly worded but different facts, questions absent from documents, and embedded instructions. Label gold pages and expected answers before running the system. Report failure cases and both cold-start and warm-query timings. For longer tests include median and p95 latency; ten queries are too few for a stable tail estimate.

Offline tests use mocks for the external services. Their pass rate verifies local behavior and error contracts; it does not measure cloud connectivity, model quality, or live retrieval accuracy. Record real API runs separately and update the technical report with measured results before submission.
