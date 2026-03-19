"""LLM-powered HTML-to-Markdown converter for RAG-optimized output.

Uses an OpenAI-compatible API to convert raw HTML into well-structured
Markdown. Falls back to the provided fallback_markdown on any failure.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_LLM_CONFIG = {
    "base_url": "http://star.sustech.edu.cn/service/model/qwen35/v1",
    "api_key": "k8#pL2@mN9!qjfkew87@#$0204",
    "model": "qwen3",
}

# Tags whose entire content should be stripped before sending to the LLM.
_STRIP_TAGS = re.compile(
    r"<\s*(script|style|svg|noscript)\b[^>]*>.*?</\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Tags to remove (open+close) but keep inner content is NOT needed for these;
# we strip the full block for nav/header/footer to reduce noise.
_REMOVE_BLOCK_TAGS = re.compile(
    r"<\s*(nav|header|footer)\b[^>]*>.*?</\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Maximum HTML character count to send to the LLM.
_MAX_HTML_CHARS = 30_000

_SYSTEM_PROMPT = """\
You are an expert HTML-to-Markdown converter for a university admissions RAG system.

The HTML content comes from official university websites and is the AUTHORITATIVE source of truth. Your job is to convert it into well-structured Markdown while preserving EVERY piece of information with 100% accuracy.

CRITICAL RULES:
1. PRESERVE ALL TEXT CONTENT EXACTLY. Do not omit, summarize, paraphrase, or rephrase ANY text.
2. All numbers, dates, deadlines, requirements, conditions, and policy details MUST be reproduced verbatim.
3. Use proper heading hierarchy: # for the page/document title, ## for major sections (一、二、三), ### for subsections (（一）（二）).
4. Convert HTML tables to Markdown tables preserving ALL rows and columns.
5. Keep hyperlinks in [text](url) format.
6. Remove ONLY: navigation menus, site headers/footers, breadcrumbs, sidebars, logos, QR codes.
7. Do NOT add YAML front matter.
8. Do NOT add any commentary, explanations, or notes.
9. Output ONLY the clean Markdown content.

The downstream RAG system relies on this output to answer student queries about admissions. ANY information loss or inaccuracy will cause incorrect answers to students."""

_USER_PROMPT_TEMPLATE = """\
Source URL: {source_url}

Convert the following HTML to well-structured Markdown:

{html}"""


def _strip_html_noise(html: str) -> str:
    """Remove script, style, svg, noscript, nav, header, footer blocks."""
    cleaned = _STRIP_TAGS.sub("", html)
    cleaned = _REMOVE_BLOCK_TAGS.sub("", cleaned)
    # Collapse runs of whitespace to single spaces within tags.
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


def convert_html_to_markdown_with_llm(
    html: str,
    source_url: str,
    fallback_markdown: str,
    llm_config: dict | None = None,
) -> str:
    """Convert HTML to Markdown using an LLM via OpenAI-compatible API.

    Args:
        html: Raw HTML content from the crawler.
        source_url: The URL the HTML was fetched from (for context).
        fallback_markdown: Pre-converted markdown to return on failure.
        llm_config: Dict with keys ``base_url``, ``api_key``, ``model``.
            Falls back to ``DEFAULT_LLM_CONFIG`` when *None*.

    Returns:
        LLM-generated Markdown string, or *fallback_markdown* on any error.
    """
    if not html or not html.strip():
        return fallback_markdown

    cfg = llm_config or DEFAULT_LLM_CONFIG

    # --- Pre-process HTML to reduce token usage ---
    stripped_html = _strip_html_noise(html)

    if not stripped_html:
        logger.warning("HTML empty after stripping noise for %s, using fallback", source_url)
        return fallback_markdown

    truncated = False
    if len(stripped_html) > _MAX_HTML_CHARS:
        logger.warning(
            "HTML for %s is %d chars after stripping (limit %d), truncating",
            source_url,
            len(stripped_html),
            _MAX_HTML_CHARS,
        )
        stripped_html = stripped_html[:_MAX_HTML_CHARS]
        truncated = True

    user_content = _USER_PROMPT_TEMPLATE.format(
        source_url=source_url,
        html=stripped_html,
    )

    try:
        import openai

        client = openai.OpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
        )

        # Use streaming to collect the response incrementally.
        stream = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=16384,
            stream=True,
        )

        parts: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)

        result = "".join(parts).strip()

        if not result:
            logger.warning("LLM returned empty result for %s, using fallback", source_url)
            return fallback_markdown

        # Strip wrapping code fences if the LLM added them despite instructions.
        if result.startswith("```"):
            lines = result.split("\n")
            # Remove first line (```markdown or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            result = "\n".join(lines).strip()

        if truncated:
            result += "\n\n> *Note: Content was truncated due to length limits.*\n"

        logger.info("LLM conversion successful for %s (%d chars)", source_url, len(result))
        return result

    except Exception:
        logger.exception("LLM conversion failed for %s, using fallback markdown", source_url)
        return fallback_markdown
