# Recorded local verification

Environment: Windows, Python 3.11.9. Verification date: 2026-09-24.

| Check | Observed result |
|---|---|
| `python -m pytest -q` | 57 passed |
| `python -m ruff check .` | All checks passed |
| `python -m pip check` | No broken requirements found |
| PDF report build | Exactly four pages |
| Real MiniLM embedding check | Three sample chunks; all vectors 384-dimensional and normalized |
| Local coordinator-question retrieval | Expected physical page 1 ranked first; cosine 0.4511 |
| Model loading | 61.55 seconds, including first-run setup/download work |
| Sample and query encoding | 0.124 seconds for this small fixture |

Key installed versions: Streamlit 1.64.0, Sentence Transformers 5.7.0, Pinecone 7.3.0, pypdf 6.19.0, httpx 0.28.1, and PyTorch 2.14.0. The embedding check is reproducible with `python scripts/check_embeddings.py`; it writes a machine-readable result under `artifacts/`.

The SDK contract test constructs requests through the installed Pinecone SDK while mocking its network transport. Streamlit tests render the initial interface and simulate a cited-answer flow with a mocked backend. These checks do not use actual Pinecone or Groq credentials.

## Live smoke test

With configured Pinecone and Groq credentials, `python scripts/check_live_answers.py` completed successfully against the included fictional course-guide fixture. The script created a temporary Pinecone namespace, removed it after the check, and recorded results in the ignored local file `artifacts/live_answers.json`.

| Scenario | Result |
|---|---|
| Document overview | Correctly summarized course duration, schedule, grading, report/video requirements, lab access, team size, and Pinecone requirement |
| Specific fact | Returned Dr. Sara Malik as course coordinator with a citation |
| Roman Urdu | Returned the eight-week duration in Roman Urdu with a citation |
| Unsupported fact | Returned the required abstention message for a lab Wi-Fi password |

The observed case timings ranged from 1.60 to 16.48 seconds; they are smoke-test observations, not a performance guarantee. The live smoke test demonstrates configured services and the main grounding routes, but it is not a general accuracy evaluation. Run `python scripts/evaluate.py` on representative documents and review every case before reporting retrieval accuracy, cost, or latency.

The source repository is published at [github.com/Izharullah733/Pdf-Evidence-RAG](https://github.com/Izharullah733/Pdf-Evidence-RAG). The required 5–7 minute demo video still needs to be recorded from the working application.
