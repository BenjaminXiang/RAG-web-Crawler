"""Web crawler module for fetching and extracting web content."""

from rag_crawler.crawler.crawler import CrawlResult, crawl_urls
from rag_crawler.crawler.url_reader import parse_urls, read_urls_from_file

__all__ = [
    "CrawlResult",
    "crawl_urls",
    "parse_urls",
    "read_urls_from_file",
]
