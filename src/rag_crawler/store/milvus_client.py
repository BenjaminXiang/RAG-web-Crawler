"""Milvus connection management and collection initialization."""

from __future__ import annotations

import logging
from typing import Optional

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
)

from rag_crawler.config import MilvusConfig

logger = logging.getLogger(__name__)

# Default embedding dimension for all-MiniLM-L6-v2
DEFAULT_EMBEDDING_DIM = 384

_COLLECTION_FIELDS = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DEFAULT_EMBEDDING_DIM),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
    FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="chunk_index", dtype=DataType.INT64),
    FieldSchema(name="crawled_at", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
]

_BM25_FUNCTION = Function(
    name="text_bm25",
    input_field_names=["text"],
    output_field_names=["sparse_embedding"],
    function_type=FunctionType.BM25,
)


def get_client(config: MilvusConfig) -> MilvusClient:
    """Create and return a MilvusClient connected to the configured URI."""
    logger.info("Connecting to Milvus at %s", config.uri)
    return MilvusClient(uri=config.uri)


def ensure_collection(
    client: MilvusClient,
    config: MilvusConfig,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> str:
    """Ensure the collection exists with the correct schema.

    Creates the collection if it does not exist. Returns the collection name.
    """
    name = config.collection_name

    if client.has_collection(name):
        logger.info("Collection '%s' already exists", name)
        return name

    # Build schema with updated embedding dim if needed
    fields = list(_COLLECTION_FIELDS)
    if embedding_dim != DEFAULT_EMBEDDING_DIM:
        fields = [
            FieldSchema(name=f.name, dtype=f.dtype, dim=embedding_dim, **{
                k: v for k, v in {
                    "is_primary": f.is_primary if f.is_primary else None,
                    "auto_id": f.auto_id if f.auto_id else None,
                    "max_length": getattr(f, "max_length", None),
                    "enable_analyzer": getattr(f, "enable_analyzer", None),
                }.items() if v is not None
            }) if f.name == "embedding" else f
            for f in fields
        ]

    schema = CollectionSchema(fields=fields)
    schema.add_function(_BM25_FUNCTION)

    logger.info("Creating collection '%s' (dim=%d)", name, embedding_dim)
    client.create_collection(
        collection_name=name,
        schema=schema,
    )

    # Create indexes
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse_embedding", index_type="AUTOINDEX", metric_type="BM25")
    client.create_index(collection_name=name, index_params=index_params)

    logger.info("Collection '%s' created with indexes", name)
    return name


def drop_collection(client: MilvusClient, config: MilvusConfig) -> None:
    """Drop the collection if it exists."""
    name = config.collection_name
    if client.has_collection(name):
        client.drop_collection(name)
        logger.info("Dropped collection '%s'", name)
