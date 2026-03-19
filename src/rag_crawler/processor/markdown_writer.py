"""Write crawl results as Markdown files with metadata."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from rag_crawler.crawler.crawler import CrawlResult

logger = logging.getLogger(__name__)

# File extensions considered attachments.
_ATTACHMENT_EXTS = frozenset((
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".gz", ".tar", ".csv", ".rtf", ".odt",
))


def _slugify(text: str, max_length: int = 80) -> str:
    """Turn arbitrary text into a filesystem-safe slug.

    Keeps Chinese characters and other Unicode letters intact.
    Replaces non-alphanumeric (excluding CJK) with hyphens, collapses runs,
    strips leading/trailing hyphens, and truncates to *max_length*.
    """
    # Replace any character that is NOT a unicode letter, digit, or hyphen
    slug = re.sub(r"[^\w-]", "-", text, flags=re.UNICODE)
    # Collapse multiple hyphens
    slug = re.sub(r"-{2,}", "-", slug)
    # Strip leading/trailing hyphens and underscores
    slug = slug.strip("-_")
    # Truncate without breaking mid-word when possible
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit("-", 1)[0] or slug[:max_length]
    return slug.strip("-_") or "untitled"


def _title_from_markdown(markdown: str) -> str | None:
    """Extract a meaningful title from markdown content.

    Tries in order:
    1. First markdown heading (# or ##)
    2. Line with 【】brackets (common Chinese article title pattern)
    3. First substantial text line (>15 chars, not nav/breadcrumb)
    """
    # Try headings first
    match = re.search(r"^#{1,2}\s+(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Try lines with 【】brackets (article titles)
    match = re.search(r"^(【[^】]+】.+)$", markdown, re.MULTILINE)
    if match:
        title = match.group(1).strip()
        return title[:80]

    # Fallback: first meaningful text line
    for line in markdown.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip image-only lines
        if re.match(r"^!\[", line):
            continue
        # Skip bullet/list items
        if re.match(r"^[\*\-]\s", line):
            continue
        # Skip lines that are just links
        if re.match(r"^\[.*\]\(.*\)$", line):
            continue
        # Skip breadcrumb-like lines (A>B>C)
        if ">" in line and line.count(">") >= 2:
            continue
        # Skip short/navigation-like fragments
        if len(line) < 15:
            continue
        title = re.split(r"[。！？\.\!\?]", line)[0][:80]
        return title.strip()
    return None


def _slug_from_url(url: str) -> str:
    """Derive a slug from the URL path when no title is available."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path or path == "/":
        # Include query string info for disambiguation
        if parsed.query:
            return _slugify(f"{parsed.netloc}-{parsed.query[:40]}")
        return _slugify(parsed.netloc)
    segments = [s for s in path.split("/") if s]
    name = "-".join(segments[-2:]) if len(segments) >= 2 else segments[-1]
    name = re.sub(r"\.(html?|php|aspx?|jsp)$", "", name, flags=re.IGNORECASE)
    return _slugify(name)


def generate_folder_name(url: str, title: str | None) -> str:
    """Generate a meaningful, filesystem-safe folder name for a URL."""
    if title:
        return _slugify(title)
    return _slug_from_url(url)


def extract_links(markdown: str) -> list[dict]:
    """Extract all markdown links as ``[{url, anchor_text}]``."""
    results: list[dict] = []
    for match in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", markdown):
        anchor_text, href = match.group(1), match.group(2)
        results.append({"url": href, "anchor_text": anchor_text})
    return results


def extract_attachments(markdown: str) -> list[dict]:
    """Extract links that point to downloadable file types."""
    attachments: list[dict] = []
    for link in extract_links(markdown):
        href_lower = link["url"].lower().split("?")[0]
        if any(href_lower.endswith(ext) for ext in _ATTACHMENT_EXTS):
            attachments.append(link)
    return attachments


def write_markdown_output(result: CrawlResult, output_dir: str) -> str:
    """Write a crawl result to disk and return the folder path.

    Creates ``{output_dir}/{folder_name}/content.md`` and
    ``{output_dir}/{folder_name}/metadata.json``.
    """
    title = _title_from_markdown(result.markdown) if result.markdown else None
    folder_name = generate_folder_name(result.url, title)

    folder_path = os.path.join(output_dir, folder_name)
    # Handle duplicate folder names by appending a counter
    base_path = folder_path
    counter = 1
    while os.path.exists(folder_path):
        counter += 1
        folder_path = f"{base_path}-{counter}"

    os.makedirs(folder_path, exist_ok=True)

    crawled_at = datetime.now(timezone.utc).isoformat()

    # --- content.md with YAML front matter ---
    front_matter = (
        "---\n"
        f"source_url: {result.url}\n"
        f"title: {title or ''}\n"
        f"crawled_at: {crawled_at}\n"
        "---\n\n"
    )
    content_path = os.path.join(folder_path, "content.md")
    with open(content_path, "w", encoding="utf-8") as fh:
        fh.write(front_matter)
        fh.write(result.markdown or "")

    # --- metadata.json ---
    links = extract_links(result.markdown) if result.markdown else []
    attachments = extract_attachments(result.markdown) if result.markdown else []

    metadata = {
        "url": result.url,
        "title": title or "",
        "crawled_at": crawled_at,
        "links": links,
        "attachments": attachments,
    }
    meta_path = os.path.join(folder_path, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    logger.info("Wrote output for %s -> %s", result.url, folder_path)
    return folder_path
