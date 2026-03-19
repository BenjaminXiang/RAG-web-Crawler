"""Tests for rag_crawler.api and CLI query command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from rag_crawler.api.app import app, _crawl_tasks
from rag_crawler.store.searcher import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_results():
    return [
        SearchResult(text="chunk one", score=0.95, source_url="https://a.com", title="A", chunk_index=0, crawled_at="2026-01-01"),
        SearchResult(text="chunk two", score=0.88, source_url="https://b.com", title="B", chunk_index=1, crawled_at="2026-01-02"),
    ]


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    @patch("rag_crawler.api.app.search")
    @patch("rag_crawler.api.app.load_config")
    def test_query_returns_results(self, mock_config, mock_search, client, sample_results):
        mock_cfg = MagicMock()
        mock_cfg.llm.provider = "none"
        mock_config.return_value = mock_cfg
        mock_search.return_value = sample_results

        resp = client.post("/api/query", json={"query": "test", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["text"] == "chunk one"
        assert data["results"][0]["score"] == pytest.approx(0.95)
        assert data["answer"] is None

    @patch("rag_crawler.api.app.search")
    @patch("rag_crawler.api.app.load_config")
    def test_query_empty_results(self, mock_config, mock_search, client):
        mock_cfg = MagicMock()
        mock_cfg.llm.provider = "none"
        mock_config.return_value = mock_cfg
        mock_search.return_value = []

        resp = client.post("/api/query", json={"query": "nothing"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    @patch("rag_crawler.api.app.search")
    @patch("rag_crawler.api.app.load_config")
    def test_query_with_filters(self, mock_config, mock_search, client):
        mock_cfg = MagicMock()
        mock_cfg.llm.provider = "none"
        mock_config.return_value = mock_cfg
        mock_search.return_value = []

        resp = client.post("/api/query", json={
            "query": "test",
            "mode": "vector",
            "filter_urls": ["https://a.com"],
            "crawled_after": "2026-01-01",
        })
        assert resp.status_code == 200
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["mode"] == "vector"
        assert call_kwargs["filter_urls"] == ["https://a.com"]
        assert call_kwargs["crawled_after"] == "2026-01-01"

    @patch("rag_crawler.api.app.generate_answer", return_value="LLM says hello")
    @patch("rag_crawler.api.app.search")
    @patch("rag_crawler.api.app.load_config")
    def test_query_with_llm(self, mock_config, mock_search, mock_llm, client, sample_results):
        mock_cfg = MagicMock()
        mock_cfg.llm.provider = "local"
        mock_config.return_value = mock_cfg
        mock_search.return_value = sample_results

        resp = client.post("/api/query", json={"query": "test", "llm": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "LLM says hello"


# ---------------------------------------------------------------------------
# Crawl endpoint
# ---------------------------------------------------------------------------

class TestCrawlEndpoint:
    @patch("rag_crawler.api.app._run_crawl")
    def test_crawl_returns_task_id(self, mock_run, client):
        resp = client.post("/api/crawl", json={"urls": ["https://a.com"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    def test_crawl_status_not_found(self, client):
        resp = client.get("/api/crawl/nonexistent")
        assert resp.status_code == 404

    def test_crawl_status_returns_state(self, client):
        task_id = "test-task-123"
        _crawl_tasks[task_id] = {
            "status": "completed",
            "total_urls": 2,
            "succeeded": 2,
            "failed": 0,
            "documents": 2,
            "error": None,
        }

        resp = client.get(f"/api/crawl/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["succeeded"] == 2

        # Cleanup
        del _crawl_tasks[task_id]


# ---------------------------------------------------------------------------
# CLI query command
# ---------------------------------------------------------------------------

class TestCliQuery:
    @patch("rag_crawler.cli.search")
    @patch("rag_crawler.cli.load_config")
    def test_query_outputs_results(self, mock_config, mock_search, sample_results):
        mock_cfg = MagicMock()
        mock_cfg.llm.provider = "none"
        mock_config.return_value = mock_cfg
        mock_search.return_value = sample_results

        from rag_crawler.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["query", "test question"])

        assert result.exit_code == 0
        assert "chunk one" in result.output
        assert "https://a.com" in result.output
        assert "0.9500" in result.output

    @patch("rag_crawler.cli.search")
    @patch("rag_crawler.cli.load_config")
    def test_query_no_results(self, mock_config, mock_search):
        mock_cfg = MagicMock()
        mock_config.return_value = mock_cfg
        mock_search.return_value = []

        from rag_crawler.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["query", "nothing"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("rag_crawler.cli.search")
    @patch("rag_crawler.cli.load_config")
    def test_query_with_options(self, mock_config, mock_search):
        mock_cfg = MagicMock()
        mock_config.return_value = mock_cfg
        mock_search.return_value = []

        from rag_crawler.cli import main
        runner = CliRunner()
        result = runner.invoke(main, [
            "query", "test",
            "--top-k", "3",
            "--mode", "keyword",
            "--url-filter", "https://a.com",
        ])

        assert result.exit_code == 0
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["mode"] == "keyword"


# ---------------------------------------------------------------------------
# LLM answer generation
# ---------------------------------------------------------------------------

class TestLlmAnswer:
    def test_empty_results(self):
        from rag_crawler.api.llm_answer import generate_answer
        from rag_crawler.config import LlmConfig

        result = generate_answer("test", [], LlmConfig())
        assert "No relevant context" in result

    @patch("openai.OpenAI")
    def test_generates_answer_with_citations(self, MockOpenAI, sample_results):
        mock_choice = MagicMock()
        mock_choice.message.content = "Answer based on [1] and [2]."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        MockOpenAI.return_value.chat.completions.create.return_value = mock_response

        from rag_crawler.api.llm_answer import generate_answer
        from rag_crawler.config import LlmConfig

        config = LlmConfig(provider="local", model="test-model")
        result = generate_answer("question", sample_results, config)

        assert "Answer based on [1] and [2]." in result
        assert "Sources:" in result
        assert "https://a.com" in result
