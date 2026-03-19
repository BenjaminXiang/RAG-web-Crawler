"""CLI entry point for the RAG web crawler."""

from __future__ import annotations

import asyncio
import logging
import sys

import click

from rag_crawler.config import load_config
from rag_crawler.crawler import crawl_urls, parse_urls, read_urls_from_file
from rag_crawler.processor import process_results
from rag_crawler.store import search, store_documents
from rag_crawler.store.searcher import SearchResult


def _configure_logging() -> None:
    """Set up root logger with timestamps at INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
def main():
    """RAG Web Crawler - crawl, process, store, and retrieve documents."""


@main.command()
@click.option("--url", multiple=True, help="URL(s) to crawl")
@click.option(
    "--urls",
    type=click.Path(exists=True),
    help="File containing URLs (one per line)",
)
@click.option("--output", "-o", default="./output", help="Output directory")
@click.option("--store/--no-store", default=True, help="Store documents in vector DB")
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    help="Config file path",
)
def crawl(url: tuple[str, ...], urls: str | None, output: str, store: bool, config_path: str):
    """Crawl URLs and save as Markdown files."""
    _configure_logging()
    logger = logging.getLogger(__name__)

    # Collect URLs from --url args and --urls file.
    collected: list[str] = list(url)
    if urls:
        collected.extend(read_urls_from_file(urls))

    if not collected:
        click.echo("Error: provide at least one URL via --url or --urls.", err=True)
        sys.exit(1)

    # Normalise and validate URLs.
    valid_urls = parse_urls(collected)
    if not valid_urls:
        click.echo("Error: none of the provided URLs are valid.", err=True)
        sys.exit(1)

    # Load configuration and override output directory.
    config = load_config(config_path)
    config.processor.output_dir = output

    click.echo(f"Crawling {len(valid_urls)} URL(s)...")
    logger.info("Starting crawl for %d URL(s)", len(valid_urls))

    # Run the async crawler.
    results = asyncio.run(crawl_urls(valid_urls, config.crawler))

    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    logger.info("Crawl complete: %d succeeded, %d failed", succeeded, failed)

    # Process successful results into markdown files.
    documents = process_results(results, config.processor, llm_config=config.llm)

    # Print summary.
    click.echo("")
    click.echo("Crawl Summary")
    click.echo("-" * 40)
    click.echo(f"  Total URLs:   {len(valid_urls)}")
    click.echo(f"  Succeeded:    {succeeded}")
    click.echo(f"  Failed:       {failed}")
    click.echo(f"  Documents:    {len(documents)}")
    click.echo(f"  Output dir:   {output}")

    if documents:
        click.echo("")
        click.echo("Saved documents:")
        for doc in documents:
            click.echo(f"  {doc.folder_path}")

    # Store to vector DB if requested
    if store and documents:
        click.echo("")
        click.echo("Storing documents in vector DB...")
        try:
            stored = store_documents(documents, config.store)
            click.echo(f"  Stored {stored} chunks in Milvus")
        except Exception as exc:
            logger.error("Failed to store documents: %s", exc)
            click.echo(f"  Warning: vector storage failed - {exc}", err=True)

    if failed:
        click.echo("")
        click.echo("Failed URLs:")
        for r in results:
            if not r.success:
                click.echo(f"  {r.url} - {r.error}")


@main.command()
@click.argument("query_text")
@click.option("--top-k", "-k", default=10, help="Number of results to return")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["hybrid", "vector", "keyword"]),
    default="hybrid",
    help="Search mode",
)
@click.option("--url-filter", multiple=True, help="Filter by source URL")
@click.option("--after", default=None, help="Filter: crawled after (ISO timestamp)")
@click.option("--before", default=None, help="Filter: crawled before (ISO timestamp)")
@click.option("--llm/--no-llm", default=False, help="Enable LLM augmented answer")
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    help="Config file path",
)
def query(
    query_text: str,
    top_k: int,
    mode: str,
    url_filter: tuple[str, ...],
    after: str | None,
    before: str | None,
    llm: bool,
    config_path: str,
):
    """Query the vector store for relevant text chunks."""
    _configure_logging()

    config = load_config(config_path)

    results = search(
        query=query_text,
        config=config.store,
        top_k=top_k,
        mode=mode,
        filter_urls=list(url_filter) if url_filter else None,
        crawled_after=after,
        crawled_before=before,
    )

    if not results:
        click.echo("No results found.")
        return

    if llm and config.llm.provider != "none":
        from rag_crawler.api.llm_answer import generate_answer

        answer = generate_answer(query_text, results, config.llm)
        click.echo("\n" + answer)
        return

    click.echo(f"\nTop {len(results)} results for: {query_text}\n")
    click.echo("=" * 60)
    for i, r in enumerate(results, 1):
        click.echo(f"\n[{i}] Score: {r.score:.4f}")
        click.echo(f"    Source: {r.source_url}")
        if r.title:
            click.echo(f"    Title:  {r.title}")
        click.echo(f"    Text:   {r.text[:200]}{'...' if len(r.text) > 200 else ''}")
    click.echo("")


if __name__ == "__main__":
    main()
