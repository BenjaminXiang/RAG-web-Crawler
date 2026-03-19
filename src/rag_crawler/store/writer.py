"""Batch vector writing and incremental updates to Milvus."""

from __future__ import annotations

import logging
from typing import Optional

from pymilvus import MilvusClient

from rag_crawler.config import EmbeddingConfig, MilvusConfig, StoreConfig
from rag_crawler.processor.processor import ProcessedDocument
from rag_crawler.store.embedding import embed_texts, get_embedding_dim
from rag_crawler.store.milvus_client import ensure_collection, get_client

logger = logging.getLogger(__name__)

# Max batch size for Milvus insert
_INSERT_BATCH = 500


def store_documents(
    documents: list[ProcessedDocument],
    config: StoreConfig,
) -> int:
    """Embed and store all chunks from documents into Milvus.

    Performs incremental update: for each document, deletes existing records
    with the same source_url before inserting new ones.

    Returns:
        Total number of chunks stored.
    """
    if not documents:
        return 0

    client = get_client(config.milvus)
    dim = get_embedding_dim(config.embedding)
    collection = ensure_collection(client, config.milvus, embedding_dim=dim)

    total_stored = 0
    for doc in documents:
        if not doc.chunks:
            logger.warning("No chunks for %s, skipping", doc.url)
            continue

        # Incremental update: delete old records for this URL
        _delete_by_url(client, collection, doc.url)

        # Prepare texts and metadata
        texts = [chunk.text for chunk in doc.chunks]

        # Embed in batches
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _INSERT_BATCH):
            batch = texts[i : i + _INSERT_BATCH]
            all_embeddings.extend(embed_texts(batch, config.embedding))

        # Build rows for insert
        rows = []
        for i, chunk in enumerate(doc.chunks):
            rows.append({
                "embedding": all_embeddings[i],
                "text": chunk.text,
                "source_url": doc.url,
                "title": doc.title or "",
                "chunk_index": chunk.chunk_index,
                "crawled_at": doc.crawled_at or "",
            })

        # Insert in batches
        for i in range(0, len(rows), _INSERT_BATCH):
            batch = rows[i : i + _INSERT_BATCH]
            client.insert(collection_name=collection, data=batch)

        total_stored += len(rows)
        logger.info("Stored %d chunks for %s", len(rows), doc.url)

    logger.info("Total chunks stored: %d", total_stored)
    return total_stored


def _delete_by_url(client: MilvusClient, collection: str, url: str) -> None:
    """Delete all records matching the given source_url."""
    filter_expr = f'source_url == "{url}"'
    result = client.delete(collection_name=collection, filter=filter_expr)
    if result:
        logger.info("Deleted existing records for %s", url)
