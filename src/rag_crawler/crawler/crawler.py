"""Main crawler module using crawl4ai for async web crawling."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

from rag_crawler.config import CrawlerConfig

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Result of crawling a single URL."""

    url: str
    html: str = ""
    markdown: str = ""
    success: bool = False
    error: Optional[str] = None


# Cache robots.txt per origin to avoid repeated fetches.
_robots_cache: dict[str, RobotFileParser] = {}


def _robots_parser_for(url: str) -> RobotFileParser:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        rp.allow_all = True
    return rp


def _is_allowed_by_robots(url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in _robots_cache:
        _robots_cache[origin] = _robots_parser_for(url)
    return _robots_cache[origin].can_fetch(user_agent, url)


async def _crawl_single(
    url: str,
    crawler: AsyncWebCrawler,
    run_config: CrawlerRunConfig,
    config: CrawlerConfig,
    semaphore: asyncio.Semaphore,
) -> CrawlResult:
    """Crawl a single URL with rate-limiting and error handling."""

    if config.respect_robots_txt:
        if not _is_allowed_by_robots(url, config.user_agent):
            msg = f"Blocked by robots.txt: {url}"
            logger.info(msg)
            return CrawlResult(url=url, error=msg)

    async with semaphore:
        try:
            result = await crawler.arun(url=url, config=run_config)

            if not result or not result.success:
                error_msg = getattr(result, "error_message", "Unknown crawl error")
                logger.warning("Failed to crawl %s: %s", url, error_msg)
                return CrawlResult(url=url, error=str(error_msg))

            # crawl4ai 0.8+ returns markdown as object with raw_markdown attr
            md = result.markdown
            if hasattr(md, "raw_markdown"):
                md = md.raw_markdown
            md = str(md) if md else ""

            logger.info("Successfully crawled %s (%d chars)", url, len(md))
            return CrawlResult(
                url=url,
                html=result.html or "",
                markdown=md,
                success=True,
            )
        except Exception as exc:
            logger.error("Exception crawling %s: %s", url, exc)
            return CrawlResult(url=url, error=str(exc))
        finally:
            if config.rate_limit > 0:
                await asyncio.sleep(1.0 / config.rate_limit)


async def crawl_urls(
    urls: list[str], config: CrawlerConfig
) -> list[CrawlResult]:
    """Crawl a list of URLs concurrently with rate limiting."""
    if not urls:
        return []

    _robots_cache.clear()

    semaphore = asyncio.Semaphore(config.concurrency)

    browser_cfg = BrowserConfig(
        headless=True,
        java_script_enabled=True,
        user_agent=config.user_agent,
        headers=config.headers,
    )
    run_cfg = CrawlerRunConfig(
        wait_until="networkidle",
        page_timeout=config.timeout * 1000,
        cache_mode=CacheMode.BYPASS,
        scan_full_page=True,
        magic=True,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        tasks = [
            _crawl_single(url, crawler, run_cfg, config, semaphore)
            for url in urls
        ]
        results = await asyncio.gather(*tasks)

    return list(results)
