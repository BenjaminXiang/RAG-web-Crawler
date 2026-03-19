"""Unit tests for rag_crawler.processor module."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from rag_crawler.config import LlmConfig, ProcessorConfig
from rag_crawler.crawler.crawler import CrawlResult
from rag_crawler.processor.cleaner import (
    _convert_bold_heading,
    _is_breadcrumb,
    _is_footer_marker,
    _is_logo_image,
    _is_nav_line,
    clean_markdown,
)
from rag_crawler.processor.markdown_writer import (
    _slugify,
    _title_from_markdown,
    extract_attachments,
    extract_links,
    generate_folder_name,
    write_markdown_output,
)
from rag_crawler.processor.processor import ProcessedDocument, _extract_title, process_results


# ---------------------------------------------------------------------------
# Cleaner tests
# ---------------------------------------------------------------------------

class TestNavLineDetection:
    def test_known_nav_fragments(self):
        assert _is_nav_line("首页") is True
        assert _is_nav_line("联系我们") is True
        assert _is_nav_line("English") is True

    def test_empty_line_is_nav(self):
        assert _is_nav_line("") is True
        assert _is_nav_line("   ") is True

    def test_short_text_is_nav(self):
        assert _is_nav_line("关于") is True

    def test_long_content_is_not_nav(self):
        assert _is_nav_line("这是一段很长的正文内容，包含了详细的说明信息。") is False

    def test_bullet_with_nav_keywords(self):
        assert _is_nav_line("* 首页招生政策综合评价报名参观预约") is True


class TestBreadcrumbDetection:
    def test_breadcrumb_with_consecutive_gt(self):
        assert _is_breadcrumb("Home >> Admission >> Policy") is True

    def test_spaced_gt_not_breadcrumb(self):
        # Single > with spaces don't match the {2,} requirement
        assert _is_breadcrumb("首页 > 招生 > 政策") is False

    def test_not_breadcrumb(self):
        assert _is_breadcrumb("This is normal text.") is False


class TestFooterDetection:
    def test_copyright_marker(self):
        assert _is_footer_marker("版权所有 南方科技大学") is True
        assert _is_footer_marker("Copyright 2026") is True

    def test_icp_marker(self):
        assert _is_footer_marker("粤ICP备12345678号") is True

    def test_normal_content(self):
        assert _is_footer_marker("这是正文内容。") is False


class TestLogoImageDetection:
    def test_logo_image(self):
        assert _is_logo_image("![logo](https://example.com/logo.png)") is True

    def test_empty_alt_image(self):
        assert _is_logo_image("![](https://example.com/img.png)") is True

    def test_content_image(self):
        assert _is_logo_image("![School campus view](https://example.com/campus.jpg)") is False


class TestBoldHeadingConversion:
    def test_major_section(self):
        assert _convert_bold_heading("**一、招生对象**") == "## 一、招生对象"

    def test_minor_section(self):
        assert _convert_bold_heading("**（1）基本条件**") == "### （1）基本条件"

    def test_non_bold_line(self):
        assert _convert_bold_heading("Normal text") == "Normal text"


class TestCleanMarkdown:
    def test_empty_input(self):
        assert clean_markdown("") == ""
        assert clean_markdown("   \n  \n  ") == ""

    def test_strips_nav_and_footer(self):
        md = "首页\n联系我们\n\n# Title\n\nContent here.\n\n版权所有 2026"
        result = clean_markdown(md)
        assert "首页" not in result
        assert "联系我们" not in result
        assert "版权所有" not in result
        assert "Content here." in result

    def test_preserves_content(self):
        md = "# Main Title\n\nThis is important content with details.\n\nMore paragraphs here."
        result = clean_markdown(md)
        assert "Main Title" in result
        assert "important content" in result
        assert "More paragraphs" in result

    def test_collapses_blank_lines(self):
        md = "# Title\n\n\n\n\n\nContent"
        result = clean_markdown(md)
        # Collapses 5+ blanks to max 2 consecutive empty lines
        assert "\n\n\n\n" not in result


# ---------------------------------------------------------------------------
# Markdown writer tests
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic_slug(self):
        assert _slugify("Hello World") == "Hello-World"

    def test_chinese_characters_preserved(self):
        slug = _slugify("南方科技大学招生简章")
        assert "南方科技大学" in slug

    def test_max_length(self):
        long = "a" * 200
        assert len(_slugify(long, max_length=80)) <= 80

    def test_empty_returns_untitled(self):
        assert _slugify("") == "untitled"
        assert _slugify("---") == "untitled"


class TestTitleFromMarkdown:
    def test_heading_title(self):
        assert _title_from_markdown("# My Title\n\nContent") == "My Title"

    def test_bracket_title(self):
        title = _title_from_markdown("【重要通知】关于2026年招生的说明")
        assert "重要通知" in title

    def test_no_title(self):
        assert _title_from_markdown("") is None

    def test_skips_images_and_short_lines(self):
        md = "![img](url)\n短\n\nThis is a real paragraph with enough text."
        title = _title_from_markdown(md)
        assert title is not None
        assert "real paragraph" in title


class TestGenerateFolderName:
    def test_with_title(self):
        name = generate_folder_name("https://example.com", "Test Page")
        assert name == "Test-Page"

    def test_without_title_uses_url(self):
        name = generate_folder_name("https://example.com/docs/guide", None)
        assert "guide" in name.lower() or "docs" in name.lower()


class TestExtractLinks:
    def test_extracts_markdown_links(self):
        md = "Visit [Google](https://google.com) and [GitHub](https://github.com)."
        links = extract_links(md)
        assert len(links) == 2
        assert links[0]["url"] == "https://google.com"
        assert links[0]["anchor_text"] == "Google"

    def test_no_links(self):
        assert extract_links("No links here.") == []


class TestExtractAttachments:
    def test_extracts_pdf_links(self):
        md = "Download [report](https://example.com/report.pdf) here."
        attachments = extract_attachments(md)
        assert len(attachments) == 1
        assert attachments[0]["url"] == "https://example.com/report.pdf"

    def test_ignores_non_attachment_links(self):
        md = "Visit [page](https://example.com/page.html)."
        assert extract_attachments(md) == []

    def test_handles_query_params(self):
        md = "Get [doc](https://example.com/file.docx?v=1)."
        attachments = extract_attachments(md)
        assert len(attachments) == 1


class TestWriteMarkdownOutput:
    def test_creates_folder_and_files(self, tmp_path):
        result = CrawlResult(
            url="https://example.com/page",
            html="<h1>Hello</h1>",
            markdown="# Hello\n\nWorld content here for testing.",
            success=True,
        )
        folder = write_markdown_output(result, str(tmp_path))

        assert os.path.isdir(folder)
        assert os.path.exists(os.path.join(folder, "content.md"))
        assert os.path.exists(os.path.join(folder, "metadata.json"))

        content = open(os.path.join(folder, "content.md")).read()
        assert "source_url: https://example.com/page" in content
        assert "# Hello" in content

        meta = json.loads(open(os.path.join(folder, "metadata.json")).read())
        assert meta["url"] == "https://example.com/page"

    def test_duplicate_folder_gets_suffix(self, tmp_path):
        result = CrawlResult(url="https://a.com", markdown="# Same Title\n\nContent.", success=True)

        folder1 = write_markdown_output(result, str(tmp_path))
        folder2 = write_markdown_output(result, str(tmp_path))

        assert folder1 != folder2
        assert os.path.isdir(folder1)
        assert os.path.isdir(folder2)


# ---------------------------------------------------------------------------
# Processor orchestration tests
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_h1_title(self):
        assert _extract_title("# My Title\n\nContent") == "My Title"

    def test_h2_not_matched(self):
        # _extract_title only matches h1 headings (^#\s+)
        assert _extract_title("## Section Title\n\nText") == ""

    def test_no_heading(self):
        assert _extract_title("Just plain text.") == ""


class TestProcessResults:
    def test_skips_failed_results(self, tmp_path):
        config = ProcessorConfig(chunk_size=512, chunk_overlap=50, output_dir=str(tmp_path))
        results = [
            CrawlResult(url="https://fail.com", success=False, error="timeout"),
        ]
        docs = process_results(results, config)
        assert docs == []

    def test_skips_empty_content(self, tmp_path):
        config = ProcessorConfig(chunk_size=512, chunk_overlap=50, output_dir=str(tmp_path))
        results = [
            CrawlResult(url="https://empty.com", markdown="", success=True),
            CrawlResult(url="https://blank.com", markdown="   \n  ", success=True),
        ]
        docs = process_results(results, config)
        assert docs == []

    def test_processes_successful_result(self, tmp_path):
        config = ProcessorConfig(chunk_size=512, chunk_overlap=50, output_dir=str(tmp_path))
        results = [
            CrawlResult(
                url="https://example.com",
                html="<h1>Test</h1><p>Content paragraph with enough text for chunking.</p>",
                markdown="# Test\n\nContent paragraph with enough text for chunking.",
                success=True,
            ),
        ]
        docs = process_results(results, config)

        assert len(docs) == 1
        assert docs[0].url == "https://example.com"
        # Title extracted from cleaned markdown (cleaner may transform content)
        assert docs[0].title != ""
        assert len(docs[0].chunks) >= 1
        assert docs[0].crawled_at != ""

    def test_processes_multiple_results(self, tmp_path):
        config = ProcessorConfig(chunk_size=512, chunk_overlap=50, output_dir=str(tmp_path))
        results = [
            CrawlResult(url="https://a.com", markdown="# A\n\nContent A is here.", success=True),
            CrawlResult(url="https://b.com", markdown="# B\n\nContent B is here.", success=True),
            CrawlResult(url="https://c.com", success=False, error="err"),
        ]
        docs = process_results(results, config)
        assert len(docs) == 2

    def test_extracts_links_and_attachments(self, tmp_path):
        config = ProcessorConfig(chunk_size=512, chunk_overlap=50, output_dir=str(tmp_path))
        md = "# Doc with Links\n\nSee [guide](https://example.com/guide) and download [pdf](https://example.com/file.pdf)."
        results = [CrawlResult(url="https://example.com", markdown=md, success=True)]
        docs = process_results(results, config)

        # At least the original links should be present
        assert len(docs[0].links) >= 2
        # PDF attachment must be detected
        pdf_urls = [a["url"] for a in docs[0].attachments]
        assert "https://example.com/file.pdf" in pdf_urls

    @patch("rag_crawler.processor.processor.convert_html_to_markdown_with_llm")
    def test_llm_conversion_when_enabled(self, mock_llm, tmp_path):
        mock_llm.return_value = "# LLM Title\n\nLLM processed content here."

        config = ProcessorConfig(chunk_size=512, chunk_overlap=50, output_dir=str(tmp_path))
        llm_config = LlmConfig(provider="local", model="test")
        results = [
            CrawlResult(
                url="https://example.com",
                html="<h1>Raw</h1><p>HTML content</p>",
                markdown="# Raw\n\nHTML content",
                success=True,
            ),
        ]
        docs = process_results(results, config, llm_config=llm_config)

        mock_llm.assert_called_once()
        assert len(docs) == 1
        assert "LLM" in docs[0].markdown

    def test_no_llm_when_provider_none(self, tmp_path):
        config = ProcessorConfig(chunk_size=512, chunk_overlap=50, output_dir=str(tmp_path))
        llm_config = LlmConfig(provider="none")
        results = [
            CrawlResult(url="https://a.com", markdown="# Title\n\nContent here.", success=True),
        ]
        with patch("rag_crawler.processor.processor.convert_html_to_markdown_with_llm") as mock_llm:
            docs = process_results(results, config, llm_config=llm_config)
            mock_llm.assert_not_called()
        assert len(docs) == 1
