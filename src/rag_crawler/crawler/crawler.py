"""Main crawler module using crawl4ai for async web crawling."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse
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
_BLOCK_LEVEL_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr")
_FALLBACK_CONTAINER_SELECTORS = (
    "article",
    "main",
    "#content",
    ".content",
    ".article",
    ".article-content",
    ".detail",
    ".detail-content",
    ".news_content",
    ".news-detail",
    ".post-content",
    ".wp_articlecontent",
)


def _looks_like_browser_false_negative(error_msg: str) -> bool:
    lowered = error_msg.lower()
    markers = (
        "blocked by anti-bot protection",
        "minimal_text",
        "no_content_elements",
        "err_network_changed",
        "failed on navigating",
    )
    return any(marker in lowered for marker in markers)


def _select_fallback_root(soup):
    for selector in _FALLBACK_CONTAINER_SELECTORS:
        node = soup.select_one(selector)
        if node:
            return node
    return soup.body or soup


def _extract_title_from_html(soup) -> str:
    for selector in ("h1", "title", "h2"):
        node = soup.select_one(selector)
        if not node:
            continue
        text = " ".join(node.stripped_strings).strip()
        if text:
            return text
    return ""


def _html_to_markdown_fallback(html: str, source_url: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    for selector in ("header", "footer", "nav", ".header", ".footer", ".nav", ".menu", ".breadcrumb", ".crumb", ".sidebar"):
        for tag in soup.select(selector):
            tag.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")

    for link in soup.find_all("a"):
        text = " ".join(link.stripped_strings).strip()
        href = (link.get("href") or "").strip()
        if href:
            href = urljoin(source_url, href)
        replacement = f"[{text}]({href})" if text and href else text or href
        link.replace_with(replacement)

    root = _select_fallback_root(soup)
    title = _extract_title_from_html(soup)
    lines: list[str] = []

    for tag in root.find_all(_BLOCK_LEVEL_TAGS):
        if tag.find_parent(_BLOCK_LEVEL_TAGS):
            continue
        if tag.name == "tr":
            cells = [" ".join(cell.stripped_strings).strip() for cell in tag.find_all(["th", "td"])]
            text = " | ".join(cell for cell in cells if cell)
        else:
            text = " ".join(tag.stripped_strings).strip()
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if tag.name.startswith("h") and tag.name[1:].isdigit():
            level = min(int(tag.name[1]), 6)
            text = f"{'#' * level} {text}"
        elif tag.name == "li":
            text = f"- {text}"
        lines.append(text)

    if not lines:
        raw_text = root.get_text("\n", strip=True)
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw_text.splitlines() if line.strip()]

    cleaned_lines: list[str] = []
    previous = ""
    for line in lines:
        if line == previous:
            continue
        cleaned_lines.append(line)
        previous = line

    if title and cleaned_lines:
        first_plain = cleaned_lines[0].lstrip("# ").strip()
        if first_plain != title:
            cleaned_lines.insert(0, "")
            cleaned_lines.insert(0, f"# {title}")
    elif title:
        cleaned_lines = [f"# {title}"]

    return "\n\n".join(line for line in cleaned_lines if line.strip())


def _http_fallback_crawl(url: str, config: CrawlerConfig) -> CrawlResult:
    import requests

    headers = {
        "User-Agent": config.user_agent,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        **config.headers,
    }
    response = requests.get(url, headers=headers, timeout=config.timeout)
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        apparent = getattr(response, "apparent_encoding", None)
        if apparent:
            response.encoding = apparent
    html = response.text or ""
    markdown = _html_to_markdown_fallback(html, source_url=url)
    if not markdown.strip():
        raise ValueError("HTTP fallback returned empty markdown")

    logger.info("Recovered %s via direct HTTP fallback (%d chars)", url, len(markdown))
    return CrawlResult(
        url=url,
        html=html,
        markdown=markdown,
        success=True,
    )


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
                if _looks_like_browser_false_negative(str(error_msg)):
                    try:
                        return await asyncio.to_thread(_http_fallback_crawl, url, config)
                    except Exception as fallback_exc:
                        logger.warning("HTTP fallback failed for %s: %s", url, fallback_exc)
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
            if _looks_like_browser_false_negative(str(exc)):
                try:
                    return await asyncio.to_thread(_http_fallback_crawl, url, config)
                except Exception as fallback_exc:
                    logger.warning("HTTP fallback failed for %s after exception: %s", url, fallback_exc)
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
