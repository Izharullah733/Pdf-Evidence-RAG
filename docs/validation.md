# Recorded local verification

Environment: Windows, Python 3.11.9. Verification date: 2026-09-24.

| Check | Observed result |
|---|---|
| `python -m pytest -q` | 34 passed in 3.11 seconds |
| `python -m ruff check .` | All checks passed |
| `python -m pip check` | No broken requirements found |
| PDF report build | Exactly four pages |
| Real MiniLM embedding check | Three sample chunks; all vectors 384-dimensional and normalized |
| Local coordinator-question retrieval | Expected physical page 1 ranked first; cosine 0.4511 |
| Model loading | 61.55 seconds, including first-run setup/download work |
| Sample and query encoding | 0.124 seconds for this small fixture |

Key installed versions: Streamlit 1.64.0, Sentence Transformers 5.7.0, Pinecone 7.3.0, pypdf 6.19.0, httpx 0.28.1, PyTorch 2.14.0. The embedding check is reproducible with `python scripts/check_embeddings.py`; it writes a machine-readable result under `artifacts/`.

The SDK contract test constructs requests through the installed Pinecone SDK while mocking its network transport. Streamlit tests render the initial interface and simulate a cited-answer flow with a mocked backend. These checks do not use actual Pinecone or Groq credentials.

Neither API key was configured during verification. Live indexing, cloud retrieval, generation quality, latency, and cost are therefore unverified. Run `python scripts/evaluate.py` after configuring `.env`, inspect the case-by-case results, and add representative PDFs before reporting assignment accuracy.

GitHub publication and an actual 5–7 minute demo recording have not been performed. The source, diagram, report, sample PDF, evaluation script, and timed recording script are present locally.
