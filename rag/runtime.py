"""Upgrade an already-running Streamlit session after the answer API changes."""

import importlib
import inspect
from types import SimpleNamespace


def get_runtime():
    service = importlib.import_module("rag.service")
    generator = importlib.import_module("rag.generator")
    models = importlib.import_module("rag.models")
    if (
        "mode" not in inspect.signature(service.RagService.ask).parameters
        or not hasattr(generator.Generator, "answer")
        or not hasattr(models, "Claim")
    ):
        # Reload dependants in order so errors and data classes share one identity.
        # Uploaded pages, Pinecone namespaces and the loaded embedding model live
        # in session state and are preserved by app.service().
        for name in (
            "models",
            "config",
            "loader",
            "chunker",
            "embeddings",
            "vector_store",
            "generator",
            "service",
        ):
            importlib.reload(importlib.import_module(f"rag.{name}"))
    return SimpleNamespace(
        **{
            name: importlib.import_module(f"rag.{name}")
            for name in ("config", "embeddings", "generator", "models", "service", "vector_store")
        }
    )
