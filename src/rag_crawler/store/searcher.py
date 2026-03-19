"""Search functionality: hybrid, vector-only, keyword-only with scalar filters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from rag_crawler.config import StoreConfig
from rag_crawler.store.embedding import embed_texts
from rag_crawler.store.milvus_client import get_client

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    text: str
    score: float
    source_url: str = ""
    title: str = ""
    chunk_index: int = 0
    crawled_at: str = ""


def search(
    query: str,
    config: StoreConfig,
    top_k: int = 10,
    mode: str = "hybrid",
    filter_urls: Optional[list[str]] = None,
    crawled_after: Optional[str] = None,
    crawled_before: Optional[str] = None,
) -> list[SearchResult]:
    """Search the vector store.

    Args:
        query: Query text.
        config: Store configuration.
        top_k: Number of results to return.
        mode: "hybrid", "vector", or "keyword".
        filter_urls: Optional list of source_url values to restrict search to.
        crawled_after: Optional ISO timestamp lower bound for crawled_at.
        crawled_before: Optional ISO timestamp upper bound for crawled_at.

    Returns:
        List of SearchResult ordered by relevance.
    """
    client = get_client(config.milvus)
    collection = config.milvus.collection_name
    output_fields = ["text", "source_url", "title", "chunk_index", "crawled_at"]

    # Build scalar filter expression
    filter_expr = _build_filter(filter_urls, crawled_after, crawled_before)

    if mode == "vector":
        return _search_vector(client, collection, query, config, top_k, filter_expr, output_fields)
    elif mode == "keyword":
        return _search_keyword(client, collection, query, top_k, filter_expr, output_fields)
    else:
        return _search_hybrid(client, collection, query, config, top_k, filter_expr, output_fields)


def _build_filter(
    filter_urls: Optional[list[str]],
    crawled_after: Optional[str],
    crawled_before: Optional[str],
) -> str:
    """Build a Milvus filter expression from optional parameters."""
    parts: list[str] = []

    if filter_urls:
        url_list = ", ".join(f'"{u}"' for u in filter_urls)
        parts.append(f"source_url in [{url_list}]")

    if crawled_after:
        parts.append(f'crawled_at >= "{crawled_after}"')

    if crawled_before:
        parts.append(f'crawled_at <= "{crawled_before}"')

    return " and ".join(parts) if parts else ""


def _search_vector(
    client: MilvusClient,
    collection: str,
    query: str,
    config: StoreConfig,
    top_k: int,
    filter_expr: str,
    output_fields: list[str],
) -> list[SearchResult]:
    """Pure vector similarity search."""
    query_vec = embed_texts([query], config.embedding)[0]

    kwargs: dict = {
        "collection_name": collection,
        "data": [query_vec],
        "anns_field": "embedding",
        "limit": top_k,
        "output_fields": output_fields,
    }
    if filter_expr:
        kwargs["filter"] = filter_expr

    results = client.search(**kwargs)
    return _parse_results(results)


def _search_keyword(
    client: MilvusClient,
    collection: str,
    query: str,
    top_k: int,
    filter_expr: str,
    output_fields: list[str],
) -> list[SearchResult]:
    """Pure BM25 keyword search."""
    kwargs: dict = {
        "collection_name": collection,
        "data": [query],
        "anns_field": "sparse_embedding",
        "limit": top_k,
        "output_fields": output_fields,
    }
    if filter_expr:
        kwargs["filter"] = filter_expr

    results = client.search(**kwargs)
    return _parse_results(results)


def _search_hybrid(
    client: MilvusClient,
    collection: str,
    query: str,
    config: StoreConfig,
    top_k: int,
    filter_expr: str,
    output_fields: list[str],
) -> list[SearchResult]:
    """Hybrid search combining vector similarity and BM25 with RRF fusion."""
    query_vec = embed_texts([query], config.embedding)[0]

    vector_req = AnnSearchRequest(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=top_k,
    )
    keyword_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse_embedding",
        param={"metric_type": "BM25"},
        limit=top_k,
    )

    kwargs: dict = {
        "collection_name": collection,
        "reqs": [vector_req, keyword_req],
        "ranker": RRFRanker(),
        "limit": top_k,
        "output_fields": output_fields,
    }
    if filter_expr:
        kwargs["filter"] = filter_expr

    results = client.hybrid_search(**kwargs)
    return _parse_results(results)


def _parse_results(raw_results: list) -> list[SearchResult]:
    """Convert Milvus search results to SearchResult objects."""
    parsed: list[SearchResult] = []
    if not raw_results:
        return parsed

    # MilvusClient.search returns list of list of hits
    hits = raw_results[0] if raw_results else []
    for hit in hits:
        entity = hit.get("entity", hit) if isinstance(hit, dict) else hit
        # Handle both dict-style and object-style access
        if isinstance(entity, dict):
            get = entity.get
        else:
            get = lambda k, d=None: getattr(entity, k, d)

        parsed.append(SearchResult(
            text=get("text", ""),
            score=hit.get("distance", 0.0) if isinstance(hit, dict) else getattr(hit, "distance", 0.0),
            source_url=get("source_url", ""),
            title=get("title", ""),
            chunk_index=get("chunk_index", 0),
            crawled_at=get("crawled_at", ""),
        ))
    return parsed
