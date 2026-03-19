"""Tests for rag_crawler.processor.markdown_writer module."""

import json
import os
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from rag_crawler.processor.markdown_writer import (
    extract_attachments,
    extract_links,
    generate_folder_name,
    write_markdown_output,
    _slugify,
    _slug_from_url,
    _title_from_markdown,
)


# Lightweight stand-in so we don't import crawl4ai at test time.
@dataclass
class FakeCrawlResult:
    url: str
    html: str = ""
    markdown: str = ""
    success: bool = True
    error: str | None = None


class TestGenerateFolderName:
    """generate_folder_name produces filesystem-safe names."""

    def test_with_english_title(self):
        name = generate_folder_name("https://example.com", "My Page Title")
        assert name == "My-Page-Title"

    def test_with_chinese_title(self):
        name = generate_folder_name("https://example.com", "招生简章2024年发布")
        assert "招生简章" in name
        # Should be filesystem safe (no slashes, etc.)
        assert "/" not in name
        assert "\\" not in name

    def test_fallback_to_url(self):
        name = generate_folder_name("https://example.com/admissions/guide.html", None)
        assert "admissions" in name
        assert "guide" in name
        # .html extension should be stripped
        assert ".html" not in name

    def test_fallback_bare_domain(self):
        name = generate_folder_name("https://example.com/", None)
        assert "example" in name

    def test_long_title_truncated(self):
        long_title = "A" * 200
        name = generate_folder_name("https://example.com", long_title)
        assert len(name) <= 80

    def test_special_chars_replaced(self):
        name = generate_folder_name("https://example.com", "Hello: World! (2024)")
        assert ":" not in name
        assert "!" not in name
        assert "(" not in name


class TestExtractLinks:
    """extract_links finds markdown links."""

    def test_finds_links(self):
        md = "Visit [Google](https://google.com) and [GitHub](https://github.com)."
        links = extract_links(md)
        assert len(links) == 2
        assert links[0]["url"] == "https://google.com"
        assert links[0]["anchor_text"] == "Google"
        assert links[1]["url"] == "https://github.com"

    def test_no_links(self):
        links = extract_links("No links here.")
        assert links == []

    def test_link_with_empty_anchor(self):
        links = extract_links("[](https://example.com)")
        assert len(links) == 1
        assert links[0]["anchor_text"] == ""
        assert links[0]["url"] == "https://example.com"

    def test_image_links_found(self):
        md = "![alt](https://img.com/pic.png)"
        links = extract_links(md)
        # Image syntax ![alt](url) - the regex matches [alt] inside ![alt]
        assert len(links) == 1
        assert links[0]["anchor_text"] == "alt"


class TestExtractAttachments:
    """extract_attachments detects PDF/doc/etc links."""

    def test_detects_pdf(self):
        md = "[Download](https://example.com/file.pdf)"
        attachments = extract_attachments(md)
        assert len(attachments) == 1
        assert attachments[0]["url"] == "https://example.com/file.pdf"

    def test_detects_docx(self):
        md = "[Report](https://example.com/report.docx)"
        attachments = extract_attachments(md)
        assert len(attachments) == 1

    def test_detects_xlsx(self):
        md = "[Data](https://example.com/data.xlsx)"
        attachments = extract_attachments(md)
        assert len(attachments) == 1

    def test_ignores_html_links(self):
        md = "[Page](https://example.com/page.html)"
        attachments = extract_attachments(md)
        assert attachments == []

    def test_ignores_plain_links(self):
        md = "[Home](https://example.com/)"
        attachments = extract_attachments(md)
        assert attachments == []

    def test_pdf_with_query_string(self):
        md = "[File](https://example.com/file.pdf?token=abc)"
        attachments = extract_attachments(md)
        assert len(attachments) == 1

    def test_multiple_attachment_types(self):
        md = (
            "[PDF](https://a.com/f.pdf) "
            "[ZIP](https://a.com/f.zip) "
            "[PPT](https://a.com/f.pptx) "
            "[Page](https://a.com/page)"
        )
        attachments = extract_attachments(md)
        assert len(attachments) == 3


class TestWriteMarkdownOutput:
    """write_markdown_output creates files with front matter."""

    def test_creates_content_and_metadata(self, tmp_path):
        result = FakeCrawlResult(
            url="https://example.com/page",
            markdown="# Test Title\n\nSome content here.",
        )
        with patch("rag_crawler.processor.markdown_writer.CrawlResult", FakeCrawlResult):
            folder = write_markdown_output(result, str(tmp_path))

        content_path = os.path.join(folder, "content.md")
        meta_path = os.path.join(folder, "metadata.json")

        assert os.path.exists(content_path)
        assert os.path.exists(meta_path)

        content = open(content_path, encoding="utf-8").read()
        assert "---" in content
        assert "source_url: https://example.com/page" in content
        assert "title: Test Title" in content
        assert "# Test Title" in content

        meta = json.loads(open(meta_path, encoding="utf-8").read())
        assert meta["url"] == "https://example.com/page"
        assert meta["title"] == "Test Title"

    def test_front_matter_has_crawled_at(self, tmp_path):
        result = FakeCrawlResult(
            url="https://example.com",
            markdown="# Page\n\nContent.",
        )
        with patch("rag_crawler.processor.markdown_writer.CrawlResult", FakeCrawlResult):
            folder = write_markdown_output(result, str(tmp_path))

        content = open(os.path.join(folder, "content.md"), encoding="utf-8").read()
        assert "crawled_at:" in content

    def test_empty_markdown(self, tmp_path):
        result = FakeCrawlResult(
            url="https://example.com/empty",
            markdown="",
        )
        with patch("rag_crawler.processor.markdown_writer.CrawlResult", FakeCrawlResult):
            folder = write_markdown_output(result, str(tmp_path))

        assert os.path.exists(os.path.join(folder, "content.md"))
        meta = json.loads(open(os.path.join(folder, "metadata.json"), encoding="utf-8").read())
        assert meta["links"] == []
        assert meta["attachments"] == []


class TestDuplicateFolderNameHandling:
    """Duplicate folder names get a counter suffix."""

    def test_appends_counter(self, tmp_path):
        result = FakeCrawlResult(
            url="https://example.com/page",
            markdown="# Same Title\n\nContent.",
        )
        with patch("rag_crawler.processor.markdown_writer.CrawlResult", FakeCrawlResult):
            folder1 = write_markdown_output(result, str(tmp_path))
            folder2 = write_markdown_output(result, str(tmp_path))

        assert folder1 != folder2
        assert os.path.exists(folder1)
        assert os.path.exists(folder2)
        # Second folder should have "-2" suffix
        assert folder2.endswith("-2")
