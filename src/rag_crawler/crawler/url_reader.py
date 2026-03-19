"""Utilities for reading and normalising URL lists."""

from __future__ import annotations

import logging
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


def read_urls_from_file(filepath: str) -> list[str]:
    """Read URLs from a text file (one per line).

    Blank lines and lines starting with ``#`` are skipped.
    Leading/trailing whitespace is stripped.
    """
    urls: list[str] = []
    with open(filepath) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def _normalise_url(url: str) -> str | None:
    """Return a normalised URL string, or None if invalid."""
    url = url.strip()
    if not url:
        return None

    # Add scheme when missing so urlparse works correctly.
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    # Must have both scheme and netloc to be valid.
    if not parsed.scheme or not parsed.netloc:
        return None

    # Rebuild to normalise (lowercases scheme/host, removes default port, etc.)
    normalised = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path or "/",
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))
    return normalised


def parse_urls(urls: list[str] | str) -> list[str]:
    """Normalise and validate one or more URLs.

    Accepts either a single URL string or a list of strings.
    Invalid entries are logged and dropped.
    """
    if isinstance(urls, str):
        urls = [urls]

    result: list[str] = []
    for raw in urls:
        normalised = _normalise_url(raw)
        if normalised is None:
            logger.warning("Skipping invalid URL: %r", raw)
            continue
        result.append(normalised)
    return result
