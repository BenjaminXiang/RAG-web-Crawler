"""Unit tests for rag_crawler.crawler module."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_crawler.config import CrawlerConfig
from rag_crawler.crawler.crawler import (
    CrawlResult,
    _html_to_markdown_fallback,
    _is_allowed_by_robots,
    _robots_cache,
    crawl_urls,
)


@pytest.fixture
def crawler_config():
    return CrawlerConfig(
        rate_limit=0,
        concurrency=2,
        timeout=10,
        user_agent="TestBot/1.0",
        respect_robots_txt=False,
        headers={},
    )


@pytest.fixture
def robots_config():
    return CrawlerConfig(
        rate_limit=0,
        concurrency=1,
        timeout=10,
        user_agent="TestBot/1.0",
        respect_robots_txt=True,
        headers={},
    )


# ---------------------------------------------------------------------------
# CrawlResult dataclass
# ---------------------------------------------------------------------------

class TestCrawlResult:
    def test_defaults(self):
        r = CrawlResult(url="https://example.com")
        assert r.url == "https://example.com"
        assert r.html == ""
        assert r.markdown == ""
        assert r.success is False
        assert r.error is None

    def test_success_result(self):
        r = CrawlResult(url="https://a.com", html="<p>hi</p>", markdown="hi", success=True)
        assert r.success
        assert r.html == "<p>hi</p>"


# ---------------------------------------------------------------------------
# crawl_urls
# ---------------------------------------------------------------------------

class TestCrawlUrls:
    def test_empty_urls_returns_empty(self, crawler_config):
        results = asyncio.run(crawl_urls([], crawler_config))
        assert results == []

    @patch("rag_crawler.crawler.crawler.AsyncWebCrawler")
    def test_successful_crawl(self, MockCrawler, crawler_config):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = "<h1>Hello</h1>"
        mock_result.markdown = "# Hello"

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockCrawler.return_value = mock_instance

        results = asyncio.run(crawl_urls(["https://example.com"], crawler_config))

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].markdown == "# Hello"
        assert results[0].url == "https://example.com"

    @patch("rag_crawler.crawler.crawler.AsyncWebCrawler")
    def test_failed_crawl_returns_error(self, MockCrawler, crawler_config):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "404 Not Found"

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockCrawler.return_value = mock_instance

        results = asyncio.run(crawl_urls(["https://bad.com"], crawler_config))

        assert len(results) == 1
        assert results[0].success is False
        assert "404" in results[0].error

    @patch("rag_crawler.crawler.crawler.AsyncWebCrawler")
    def test_exception_during_crawl(self, MockCrawler, crawler_config):
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(side_effect=ConnectionError("timeout"))
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockCrawler.return_value = mock_instance

        results = asyncio.run(crawl_urls(["https://timeout.com"], crawler_config))

        assert len(results) == 1
        assert results[0].success is False
        assert "timeout" in results[0].error

    @patch("rag_crawler.crawler.crawler.AsyncWebCrawler")
    def test_multiple_urls_concurrent(self, MockCrawler, crawler_config):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = "<p>ok</p>"
        mock_result.markdown = "ok"

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockCrawler.return_value = mock_instance

        urls = ["https://a.com", "https://b.com", "https://c.com"]
        results = asyncio.run(crawl_urls(urls, crawler_config))

        assert len(results) == 3
        assert all(r.success for r in results)

    @patch("rag_crawler.crawler.crawler.AsyncWebCrawler")
    def test_markdown_object_with_raw_markdown(self, MockCrawler, crawler_config):
        """crawl4ai 0.8+ returns markdown as object with raw_markdown attr."""
        md_obj = MagicMock()
        md_obj.raw_markdown = "# From Object"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = "<h1>From Object</h1>"
        mock_result.markdown = md_obj

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockCrawler.return_value = mock_instance

        results = asyncio.run(crawl_urls(["https://example.com"], crawler_config))

        assert results[0].markdown == "# From Object"

    @patch("rag_crawler.crawler.crawler._http_fallback_crawl")
    @patch("rag_crawler.crawler.crawler.AsyncWebCrawler")
    def test_http_fallback_recovers_antibot_result(self, MockCrawler, mock_fallback, crawler_config):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Blocked by anti-bot protection: Structural: minimal_text, no_content_elements"

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockCrawler.return_value = mock_instance

        mock_fallback.return_value = CrawlResult(
            url="https://example.com",
            html="<html><body><h1>Recovered</h1></body></html>",
            markdown="# Recovered",
            success=True,
        )

        results = asyncio.run(crawl_urls(["https://example.com"], crawler_config))

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].markdown == "# Recovered"
        mock_fallback.assert_called_once()

    @patch("rag_crawler.crawler.crawler._http_fallback_crawl")
    @patch("rag_crawler.crawler.crawler.AsyncWebCrawler")
    def test_http_fallback_recovers_navigation_exception(self, MockCrawler, mock_fallback, crawler_config):
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(side_effect=RuntimeError("Page.goto: net::ERR_NETWORK_CHANGED"))
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockCrawler.return_value = mock_instance

        mock_fallback.return_value = CrawlResult(
            url="https://example.com",
            html="<html><body><p>Recovered</p></body></html>",
            markdown="# Recovered\n\nRecovered",
            success=True,
        )

        results = asyncio.run(crawl_urls(["https://example.com"], crawler_config))

        assert len(results) == 1
        assert results[0].success is True
        mock_fallback.assert_called_once()


# ---------------------------------------------------------------------------
# robots.txt handling
# ---------------------------------------------------------------------------

class TestRobotsTxt:
    @patch("rag_crawler.crawler.crawler._is_allowed_by_robots", return_value=False)
    @patch("rag_crawler.crawler.crawler.AsyncWebCrawler")
    def test_blocked_by_robots(self, MockCrawler, mock_robots, robots_config):
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockCrawler.return_value = mock_instance

        results = asyncio.run(crawl_urls(["https://blocked.com/secret"], robots_config))

        assert len(results) == 1
        assert results[0].success is False
        assert "robots.txt" in results[0].error

    @patch("rag_crawler.crawler.crawler._robots_parser_for")
    def test_is_allowed_caches_per_origin(self, mock_parser):
        _robots_cache.clear()
        rp = MagicMock()
        rp.can_fetch.return_value = True
        mock_parser.return_value = rp

        _is_allowed_by_robots("https://site.com/page1", "Bot")
        _is_allowed_by_robots("https://site.com/page2", "Bot")

        # Parser created only once per origin
        mock_parser.assert_called_once()
        assert rp.can_fetch.call_count == 2
        _robots_cache.clear()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    @patch("rag_crawler.crawler.crawler.AsyncWebCrawler")
    def test_rate_limit_adds_delay(self, MockCrawler):
        config = CrawlerConfig(
            rate_limit=10.0,  # 10 req/s = 0.1s delay
            concurrency=1,
            timeout=10,
            user_agent="Bot",
            respect_robots_txt=False,
            headers={},
        )

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.html = "<p>ok</p>"
        mock_result.markdown = "ok"

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockCrawler.return_value = mock_instance

        # Just verify it completes without error with rate limiting active
        results = asyncio.run(crawl_urls(["https://a.com"], config))
        assert results[0].success is True


class TestHttpFallbackFormatting:
    def test_html_fallback_extracts_title_and_links(self):
        html = """
        <html>
          <head><title>招生简章</title></head>
          <body>
            <nav>首页</nav>
            <div class="content">
              <h1>南方科技大学2026年综合评价招生简章</h1>
              <p>欢迎报考南方科技大学。</p>
              <p><a href="/apply">报名入口</a></p>
            </div>
          </body>
        </html>
        """

        markdown = _html_to_markdown_fallback(html, "https://example.com/post")

        assert "# 招生简章" in markdown or "# 南方科技大学2026年综合评价招生简章" in markdown
        assert "欢迎报考南方科技大学。" in markdown
        assert "[报名入口](https://example.com/apply)" in markdown
