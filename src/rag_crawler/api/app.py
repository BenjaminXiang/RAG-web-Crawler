"""FastAPI application with query, crawl, and health endpoints."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from rag_crawler.config import load_config
from rag_crawler.store.searcher import search, SearchResult
from rag_crawler.api.llm_answer import generate_answer

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Web Crawler API", version="0.1.0")

# In-memory task store for crawl jobs
_crawl_tasks: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: str = "hybrid"
    filter_urls: Optional[list[str]] = None
    crawled_after: Optional[str] = None
    crawled_before: Optional[str] = None
    llm: bool = False


class QueryResultItem(BaseModel):
    text: str
    score: float
    source_url: str = ""
    title: str = ""
    chunk_index: int = 0
    crawled_at: str = ""


class QueryResponse(BaseModel):
    results: list[QueryResultItem]
    answer: Optional[str] = None


class CrawlRequest(BaseModel):
    urls: list[str]


class CrawlTaskResponse(BaseModel):
    task_id: str
    status: str


class CrawlStatusResponse(BaseModel):
    task_id: str
    status: str
    total_urls: int = 0
    succeeded: int = 0
    failed: int = 0
    documents: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Query the vector store for relevant text chunks."""
    config = load_config()

    results = search(
        query=req.query,
        config=config.store,
        top_k=req.top_k,
        mode=req.mode,
        filter_urls=req.filter_urls,
        crawled_after=req.crawled_after,
        crawled_before=req.crawled_before,
    )

    items = [
        QueryResultItem(
            text=r.text,
            score=r.score,
            source_url=r.source_url,
            title=r.title,
            chunk_index=r.chunk_index,
            crawled_at=r.crawled_at,
        )
        for r in results
    ]

    answer = None
    if req.llm and results and config.llm.provider != "none":
        answer = generate_answer(req.query, results, config.llm)

    return QueryResponse(results=items, answer=answer)


@app.post("/api/crawl", response_model=CrawlTaskResponse)
async def crawl_endpoint(req: CrawlRequest, background_tasks: BackgroundTasks):
    """Trigger an asynchronous crawl task."""
    task_id = str(uuid.uuid4())
    _crawl_tasks[task_id] = {
        "status": "pending",
        "total_urls": len(req.urls),
        "succeeded": 0,
        "failed": 0,
        "documents": 0,
        "error": None,
    }

    background_tasks.add_task(_run_crawl, task_id, req.urls)
    return CrawlTaskResponse(task_id=task_id, status="pending")


@app.get("/api/crawl/{task_id}", response_model=CrawlStatusResponse)
async def crawl_status(task_id: str):
    """Get crawl task status."""
    if task_id not in _crawl_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _crawl_tasks[task_id]
    return CrawlStatusResponse(task_id=task_id, **task)


# ---------------------------------------------------------------------------
# Background crawl runner
# ---------------------------------------------------------------------------

def _run_crawl(task_id: str, urls: list[str]) -> None:
    """Execute the full crawl pipeline in the background."""
    from rag_crawler.crawler import crawl_urls, parse_urls
    from rag_crawler.processor import process_results
    from rag_crawler.store import store_documents

    task = _crawl_tasks[task_id]
    task["status"] = "running"

    try:
        config = load_config()
        valid_urls = parse_urls(urls)

        if not valid_urls:
            task["status"] = "failed"
            task["error"] = "No valid URLs provided"
            return

        task["total_urls"] = len(valid_urls)

        results = asyncio.run(crawl_urls(valid_urls, config.crawler))
        succeeded = sum(1 for r in results if r.success)
        task["succeeded"] = succeeded
        task["failed"] = len(results) - succeeded

        documents = process_results(results, config.processor, llm_config=config.llm)
        task["documents"] = len(documents)

        if documents:
            store_documents(documents, config.store)

        task["status"] = "completed"
    except Exception as exc:
        logger.exception("Crawl task %s failed", task_id)
        task["status"] = "failed"
        task["error"] = str(exc)


def create_app() -> FastAPI:
    """Factory function for the FastAPI app."""
    return app
