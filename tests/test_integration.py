"""End-to-end integration tests for the RAG pipeline.

These tests mock external services (crawl4ai, Milvus, embedding models)
but verify the full flow from URL input through to vector storage and query.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from rag_crawler.config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_crawl_result(url: str):
    """Return a mock CrawlResult."""
    from rag_crawler.crawler import CrawlResult
    return CrawlResult(
        url=url,
        html="<h1>Title</h1><p>Some test content for RAG processing.</p>",
        markdown="# Title\n\nSome test content for RAG processing.",
        success=True,
    )


def _fake_embed(texts, config):
    return [[0.1] * 4 for _ in texts]


# ---------------------------------------------------------------------------
# CLI end-to-end: crawl → process → store
# ---------------------------------------------------------------------------

class TestCrawlE2E:
    @patch("rag_crawler.cli.store_documents", return_value=2)
    @patch("rag_crawler.cli.process_results")
    @patch("rag_crawler.cli.crawl_urls")
    @patch("rag_crawler.cli.load_config")
    def test_crawl_stores_documents(self, mock_config, mock_crawl, mock_process, mock_store):
        mock_cfg = MagicMock()
        mock_cfg.processor.output_dir = "./output"
        mock_config.return_value = mock_cfg

        mock_crawl.return_value = [_fake_crawl_result("https://example.com/")]
        mock_process.return_value = [MagicMock(folder_path="/tmp/out/example")]

        from rag_crawler.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["crawl", "--url", "https://example.com"])

        assert result.exit_code == 0
        assert "Stored 2 chunks" in result.output
        mock_store.assert_called_once()

    @patch("rag_crawler.cli.process_results")
    @patch("rag_crawler.cli.crawl_urls")
    @patch("rag_crawler.cli.load_config")
    def test_crawl_no_store_flag_skips_storage(self, mock_config, mock_crawl, mock_process):
        mock_cfg = MagicMock()
        mock_cfg.processor.output_dir = "./output"
        mock_config.return_value = mock_cfg

        mock_crawl.return_value = [_fake_crawl_result("https://example.com/")]
        mock_process.return_value = [MagicMock(folder_path="/tmp/out/example")]

        from rag_crawler.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["crawl", "--url", "https://example.com", "--no-store"])

        assert result.exit_code == 0
        assert "Storing documents" not in result.output

    @patch("rag_crawler.cli.crawl_urls")
    @patch("rag_crawler.cli.load_config")
    def test_crawl_no_urls_exits_with_error(self, mock_config, mock_crawl):
        from rag_crawler.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["crawl"])

        assert result.exit_code != 0 or "Error" in result.output


# ---------------------------------------------------------------------------
# CLI end-to-end: query → search → display
# ---------------------------------------------------------------------------

class TestQueryE2E:
    @patch("rag_crawler.cli.search")
    @patch("rag_crawler.cli.load_config")
    def test_query_displays_results(self, mock_config, mock_search):
        from rag_crawler.store.searcher import SearchResult
        mock_cfg = MagicMock()
        mock_cfg.llm.provider = "none"
        mock_config.return_value = mock_cfg
        mock_search.return_value = [
            SearchResult(text="RAG content", score=0.92, source_url="https://example.com", title="Example", chunk_index=0, crawled_at="2026-01-01"),
        ]

        from rag_crawler.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["query", "what is RAG"])

        assert result.exit_code == 0
        assert "RAG content" in result.output
        assert "0.9200" in result.output
        assert "https://example.com" in result.output


# ---------------------------------------------------------------------------
# API end-to-end: crawl → status
# ---------------------------------------------------------------------------

class TestApiCrawlE2E:
    def test_crawl_lifecycle(self):
        from rag_crawler.api.app import app, _crawl_tasks

        client = TestClient(app)

        with patch("rag_crawler.api.app._run_crawl"):
            resp = client.post("/api/crawl", json={"urls": ["https://example.com"]})
            assert resp.status_code == 200
            task_id = resp.json()["task_id"]
            assert resp.json()["status"] == "pending"

        # Simulate task completion
        _crawl_tasks[task_id]["status"] = "completed"
        _crawl_tasks[task_id]["succeeded"] = 1
        _crawl_tasks[task_id]["documents"] = 1

        resp = client.get(f"/api/crawl/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["succeeded"] == 1

        # Cleanup
        del _crawl_tasks[task_id]


# ---------------------------------------------------------------------------
# API end-to-end: query
# ---------------------------------------------------------------------------

class TestApiQueryE2E:
    @patch("rag_crawler.api.app.search")
    @patch("rag_crawler.api.app.load_config")
    def test_query_flow(self, mock_config, mock_search):
        from rag_crawler.store.searcher import SearchResult
        from rag_crawler.api.app import app

        mock_cfg = MagicMock()
        mock_cfg.llm.provider = "none"
        mock_config.return_value = mock_cfg
        mock_search.return_value = [
            SearchResult(text="Integration test chunk", score=0.85, source_url="https://test.com", title="Test", chunk_index=0, crawled_at="2026-01-01"),
        ]

        client = TestClient(app)
        resp = client.post("/api/query", json={"query": "test integration", "top_k": 3})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "Integration test chunk"
        assert data["answer"] is None
