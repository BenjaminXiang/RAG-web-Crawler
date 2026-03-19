"""Post-process crawl4ai raw markdown into well-structured RAG content.

Removes site navigation, headers/footers, breadcrumbs, and converts
bold-only lines into proper markdown headings.
"""

from __future__ import annotations

import re

# Navigation fragments commonly found in Chinese academic/institutional sites.
_NAV_FRAGMENTS = frozenset((
    "首页", "联系我们", "关注我们", "网站地图", "设为首页",
    "加入收藏", "English", "中文", "登录", "注册",
    "返回顶部", "回到顶部", "在线咨询",
))

# Footer markers -- any line containing one of these triggers footer state.
_FOOTER_MARKERS = (
    "版权所有", "Copyright", "\u00a9", "All Rights Reserved",
    "友情链接", "备案号", "ICP备", "技术支持",
)

# Breadcrumb pattern: two or more ">" separating short fragments.
_BREADCRUMB_RE = re.compile(r"^[^\n]{0,80}(?:>|>){2,}")

# Bold-only line: optional whitespace, then **text**, nothing else.
_BOLD_ONLY_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")

# Chinese numbered section patterns for heading level detection.
_MAJOR_SECTION_RE = re.compile(
    r"^[一二三四五六七八九十]+[、.．]"  # 一、 二、
    r"|^\d+[、.．]\s"                    # 1、 1.
    r"|^（[一二三四五六七八九十]+）"      # （一）
)
_MINOR_SECTION_RE = re.compile(
    r"^（\d+）"                          # （1）
    r"|^\(\d+\)"                         # (1)
    r"|^\d+\.\d+"                        # 1.1
)

# Logo/banner image patterns.
_SKIP_IMAGE_RE = re.compile(
    r"^!\[([^\]]*)\]\([^)]+\)\s*$"
)
_LOGO_BANNER_RE = re.compile(
    r"logo|banner|icon|favicon|qrcode|二维码",
    re.IGNORECASE,
)


def _is_nav_line(line: str) -> bool:
    """Check if a stripped line looks like navigation text."""
    stripped = line.strip().rstrip("| ").strip()
    if not stripped:
        return True
    # Exact match against known nav fragments.
    if stripped in _NAV_FRAGMENTS:
        return True
    # Bullet/list item containing only a short nav-like phrase.
    clean = re.sub(r"^[\*\-\+]\s*", "", stripped)
    clean = re.sub(r"^\[([^\]]*)\]\([^)]*\)$", r"\1", clean).strip()
    if clean in _NAV_FRAGMENTS:
        return True
    # Short text without punctuation (likely nav item, not content)
    if len(clean) <= 6 and not re.search(r"[。！？，；：]", clean):
        return True
    # Compressed nav: bullet with multiple short items joined (e.g. "首页招生政策综合评价...")
    if re.match(r"^[\*\-\+]\s*", stripped):
        inner = re.sub(r"^[\*\-\+]\s*", "", stripped)
        # If it contains multiple nav-like keywords concatenated
        nav_keywords = ["首页", "招生", "联系", "关于", "登录", "报名", "走进", "校园", "参观", "预约"]
        matches = sum(1 for kw in nav_keywords if kw in inner)
        if matches >= 3:
            return True
    return False


def _is_breadcrumb(line: str) -> bool:
    stripped = line.strip()
    return bool(_BREADCRUMB_RE.match(stripped))


def _is_footer_marker(line: str) -> bool:
    for marker in _FOOTER_MARKERS:
        if marker in line:
            return True
    return False


def _is_logo_image(line: str) -> bool:
    m = _SKIP_IMAGE_RE.match(line.strip())
    if not m:
        return False
    alt = m.group(1)
    # No alt text or alt text contains logo/banner keywords.
    if not alt.strip() or _LOGO_BANNER_RE.search(alt) or _LOGO_BANNER_RE.search(line):
        return True
    return False


def _convert_bold_heading(line: str) -> str:
    """Convert a bold-only line to a markdown heading."""
    m = _BOLD_ONLY_RE.match(line)
    if not m:
        return line
    text = m.group(1).strip()
    if _MINOR_SECTION_RE.match(text):
        return f"### {text}"
    # Major sections and everything else get ##.
    return f"## {text}"


def _is_substantial(line: str) -> bool:
    """A line is 'substantial' if it has >15 chars of actual text content."""
    stripped = line.strip()
    if not stripped:
        return False
    # Remove markdown image/link syntax for length check.
    plain = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", stripped)
    plain = re.sub(r"\[[^\]]*\]\([^)]*\)", "", plain)
    plain = re.sub(r"[#*_\-\[\]()>]", "", plain).strip()
    return len(plain) > 15


def _extract_title_candidate(lines: list[str]) -> str | None:
    """Find a document title from content lines.

    Priority: bracket title 【】 > first substantial text line > H1/H2 heading.
    """
    # First pass: look for 【】bracket title (strongest signal).
    for line in lines:
        stripped = line.strip()
        m = re.match(r"^(?:#{1,2}\s+)?(【[^】]+】.+)$", stripped)
        if m:
            return m.group(1).strip()[:80]

    # Second pass: find the first substantial non-heading text line.
    # This is often the actual document title in Chinese content.
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Skip date-like lines
        if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", stripped):
            continue
        # Skip short lines, tables, images
        if len(stripped) < 8 or stripped.startswith("|") or stripped.startswith("!"):
            continue
        # This looks like a title candidate
        return stripped[:80]

    # Fallback: first heading
    for line in lines:
        stripped = line.strip()
        m = re.match(r"^#{1,2}\s+(.+)$", stripped)
        if m:
            return m.group(1).strip()
    return None


def clean_markdown(markdown: str, source_url: str = "") -> str:
    """Post-process crawl4ai markdown output into well-structured content.

    Uses a three-state machine (header -> content -> footer) to strip
    navigation, site furniture, and footer boilerplate. Converts bold-only
    lines to proper headings and normalizes whitespace.

    Args:
        markdown: Raw markdown string from crawl4ai.
        source_url: Optional source URL for context (currently unused,
            reserved for future domain-specific rules).

    Returns:
        Cleaned, well-structured markdown string.
    """
    if not markdown or not markdown.strip():
        return ""

    lines = markdown.split("\n")
    state = "header"  # header -> content -> footer
    content_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # --- Footer detection (from any state) ---
        if _is_footer_marker(stripped):
            state = "footer"
            continue

        if state == "footer":
            continue

        # --- Header state: skip nav/logos until first substantial content ---
        if state == "header":
            if _is_logo_image(line):
                continue
            if _is_breadcrumb(line):
                continue
            if _is_nav_line(line):
                continue
            # Transition to content on first substantial line or heading.
            if _is_substantial(line) or re.match(r"^#{1,6}\s", stripped):
                state = "content"
                # Fall through to content processing.
            elif _BOLD_ONLY_RE.match(line):
                # A bold-only line can also start content (it's a section title).
                state = "content"
            else:
                # Non-substantial, non-nav line in header -- keep skipping.
                continue

        # --- Content state ---
        if state == "content":
            # Skip breadcrumbs and logo images even in content.
            if _is_breadcrumb(line):
                continue
            if _is_logo_image(line):
                continue
            # Convert bold-only lines to headings.
            line = _convert_bold_heading(line)
            # Convert plain-text Chinese numbered section titles to headings.
            # Only when the line IS the title (short, standalone, not a paragraph start).
            stripped_for_heading = line.strip()
            if (not stripped_for_heading.startswith("#")
                    and _MAJOR_SECTION_RE.match(stripped_for_heading)):
                # Only treat as heading if the entire line is short (a pure title)
                # NOT when it's "（二）报名参加...须符合下列条件：" (long paragraph)
                plain_text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", stripped_for_heading)
                if len(plain_text) <= 30:
                    line = f"## {stripped_for_heading}"
            content_lines.append(line)

    # Collapse 3+ consecutive blank lines to 2.
    result_lines: list[str] = []
    blank_count = 0
    for line in content_lines:
        stripped_line = line.rstrip()
        if not stripped_line:
            blank_count += 1
            if blank_count <= 2:
                result_lines.append("")
        else:
            blank_count = 0
            result_lines.append(stripped_line)

    # Strip leading/trailing blank lines.
    while result_lines and not result_lines[0].strip():
        result_lines.pop(0)
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()

    # Add H1 title if content doesn't start with a heading.
    if result_lines and not re.match(r"^#", result_lines[0]):
        title = _extract_title_candidate(result_lines)
        if title:
            result_lines.insert(0, f"# {title}")
            result_lines.insert(1, "")
            # Remove duplicate: if line right after blank is the same as title
            if len(result_lines) > 2 and result_lines[2].strip() == title:
                result_lines.pop(2)

    return "\n".join(result_lines) + "\n" if result_lines else ""
