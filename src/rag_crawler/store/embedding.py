"""Pluggable embedding interface with local and OpenAI backends."""

from __future__ import annotations

import logging
from typing import Optional

from rag_crawler.config import EmbeddingConfig

logger = logging.getLogger(__name__)

# Cache loaded local model to avoid reloading
_local_model = None
_local_model_name: Optional[str] = None


def _get_local_model(model_name: str):
    """Lazy-load a sentence-transformers model (cached)."""
    global _local_model, _local_model_name
    if _local_model is not None and _local_model_name == model_name:
        return _local_model

    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformers model: %s", model_name)
    _local_model = SentenceTransformer(model_name)
    _local_model_name = model_name
    return _local_model


def embed_texts(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    """Compute embeddings for a list of texts.

    Uses sentence-transformers locally or OpenAI API based on config.provider.

    Returns:
        List of embedding vectors, one per input text.
    """
    if not texts:
        return []

    if config.provider == "openai":
        return _embed_openai(texts, config)
    return _embed_local(texts, config)


def get_embedding_dim(config: EmbeddingConfig) -> int:
    """Return the embedding dimension for the configured model."""
    if config.provider == "openai":
        model = config.openai_model or "text-embedding-3-small"
        # Known dimensions for OpenAI models
        dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dims.get(model, 1536)

    # Local model: load and check
    model = _get_local_model(config.model)
    return model.get_sentence_embedding_dimension()


def _embed_local(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    """Embed using sentence-transformers."""
    model = _get_local_model(config.model)
    embeddings = model.encode(texts, show_progress_bar=False)
    return [vec.tolist() for vec in embeddings]


def _embed_openai(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    """Embed using OpenAI API."""
    import openai

    model = config.openai_model or "text-embedding-3-small"
    logger.info("Embedding %d texts via OpenAI (%s)", len(texts), model)

    client = openai.OpenAI()
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]
