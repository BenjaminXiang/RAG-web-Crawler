"""Main processor orchestrating markdown output and chunking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rag_crawler.config import LlmConfig, ProcessorConfig
from rag_crawler.crawler.crawler import CrawlResult
from rag_crawler.processor.chunker import Chunk, chunk_text
from rag_crawler.processor.cleaner import clean_markdown
from rag_crawler.processor.llm_converter import convert_html_to_markdown_with_llm
from rag_crawler.processor.markdown_writer import (
    extract_attachments,
    extract_links,
    write_markdown_output,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Fully processed document ready for downstream consumption."""

    url: str
    title: str
    folder_path: str
    markdown: str
    chunks: list[Chunk] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    attachments: list[dict] = field(default_factory=list)
    crawled_at: str = ""


def _extract_title(markdown: str) -> str:
    """Extract title from the first # heading, or return empty string."""
    import re

    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else ""


def process_results(
    results: list[CrawlResult],
    config: ProcessorConfig,
    llm_config: LlmConfig | None = None,
) -> list[ProcessedDocument]:
    """Process crawl results into structured documents.

    Writes markdown files, chunks text, and extracts link/attachment metadata.
    Skips failed or empty results with a warning.

    When *llm_config* is provided and its provider is not ``"none"``, the raw
    HTML is converted to Markdown via an LLM.  The crawl4ai markdown is kept as
    a fallback in case the LLM call fails.

    Args:
        results: Crawl results from the crawler module.
        config: Processor configuration (output_dir, chunk_size, chunk_overlap).
        llm_config: Optional LLM configuration for HTML-to-Markdown conversion.

    Returns:
        List of successfully processed documents.
    """
    use_llm = llm_config is not None and llm_config.provider != "none"
    if use_llm:
        _llm_cfg = {
            "base_url": llm_config.base_url,
            "api_key": llm_config.api_key,
            "model": llm_config.model,
        }
        logger.info("LLM conversion enabled (provider=%s, model=%s)", llm_config.provider, llm_config.model)

    documents: list[ProcessedDocument] = []

    for result in results:
        if not result.success:
            logger.warning(
                "Skipping failed crawl result for %s: %s",
                result.url,
                result.error or "unknown error",
            )
            continue

        if not result.markdown or not result.markdown.strip():
            logger.warning(
                "Skipping empty content for %s",
                result.url,
            )
            continue

        if use_llm and result.html:
            # Use LLM to convert raw HTML; fall back to cleaned crawl4ai markdown.
            fallback_md = clean_markdown(result.markdown, source_url=result.url)
            result.markdown = convert_html_to_markdown_with_llm(
                html=result.html,
                source_url=result.url,
                fallback_markdown=fallback_md,
                llm_config=_llm_cfg,
            )
        else:
            # Clean raw markdown before writing.
            result.markdown = clean_markdown(result.markdown, source_url=result.url)

        # Write markdown + metadata to disk.
        folder_path = write_markdown_output(result, config.output_dir)

        # Chunk the markdown content.
        chunks = chunk_text(
            result.markdown,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        title = _extract_title(result.markdown)
        links = extract_links(result.markdown)
        attachments = extract_attachments(result.markdown)

        doc = ProcessedDocument(
            url=result.url,
            title=title,
            folder_path=folder_path,
            markdown=result.markdown,
            chunks=chunks,
            links=links,
            attachments=attachments,
            crawled_at=datetime.now(timezone.utc).isoformat(),
        )
        documents.append(doc)
        logger.info(
            "Processed %s: %d chunks, %d links, %d attachments",
            result.url,
            len(chunks),
            len(links),
            len(attachments),
        )

    logger.info("Processed %d/%d crawl results", len(documents), len(results))
    return documents
