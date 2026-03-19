"""Vector store module for embedding storage and retrieval."""

from rag_crawler.store.embedding import embed_texts, get_embedding_dim
from rag_crawler.store.exporter import export_jsonl
from rag_crawler.store.milvus_client import drop_collection, ensure_collection, get_client
from rag_crawler.store.searcher import SearchResult, search
from rag_crawler.store.writer import store_documents

__all__ = [
    "embed_texts",
    "export_jsonl",
    "drop_collection",
    "ensure_collection",
    "get_client",
    "get_embedding_dim",
    "search",
    "SearchResult",
    "store_documents",
]
