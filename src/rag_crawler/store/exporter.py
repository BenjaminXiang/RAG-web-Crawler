"""JSONL export of stored chunks with optional embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pymilvus import MilvusClient

from rag_crawler.config import StoreConfig
from rag_crawler.store.milvus_client import get_client

logger = logging.getLogger(__name__)

_QUERY_BATCH = 1000


def export_jsonl(
    config: StoreConfig,
    output_path: Optional[str] = None,
    include_embedding: bool = True,
) -> str:
    """Export all records from Milvus collection to a JSONL file.

    Args:
        config: Store configuration.
        output_path: Override output file path. Defaults to config.export_dir/export.jsonl.
        include_embedding: Whether to include embedding vectors in the output.

    Returns:
        Path to the written JSONL file.
    """
    client = get_client(config.milvus)
    collection = config.milvus.collection_name

    # Determine output path
    if output_path is None:
        out_dir = Path(config.export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "full" if include_embedding else "no_embedding"
        out_file = out_dir / f"export_{suffix}.jsonl"
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    output_fields = ["text", "source_url", "title", "chunk_index", "crawled_at"]
    if include_embedding:
        output_fields.append("embedding")

    # Query all records using iteration
    results = client.query(
        collection_name=collection,
        filter="",
        output_fields=output_fields,
        limit=_QUERY_BATCH,
    )

    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for record in results:
            row: dict = {
                "text": record.get("text", ""),
                "metadata": {
                    "source_url": record.get("source_url", ""),
                    "title": record.get("title", ""),
                    "chunk_index": record.get("chunk_index", 0),
                    "crawled_at": record.get("crawled_at", ""),
                },
            }
            if include_embedding and "embedding" in record:
                row["embedding"] = record["embedding"]

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d records to %s", count, out_file)
    return str(out_file)
