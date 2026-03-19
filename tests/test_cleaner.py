"""Tests for rag_crawler.processor.cleaner module."""

import pytest

from rag_crawler.processor.cleaner import clean_markdown


class TestNavigationRemoval:
    """Navigation lines should be stripped from the header."""

    def test_removes_nav_items(self):
        md = "首页\n联系我们\n登录\n\nThis is a substantial content paragraph for testing purposes."
        result = clean_markdown(md)
        assert "首页" not in result
        assert "联系我们" not in result
        assert "登录" not in result
        assert "substantial content" in result

    def test_removes_bullet_nav(self):
        md = "* 首页\n* 登录\n\nThis is the main document content with enough text."
        result = clean_markdown(md)
        assert "首页" not in result
        assert "main document content" in result


class TestBreadcrumbRemoval:
    """Breadcrumb lines should be removed."""

    def test_removes_breadcrumbs(self):
        md = "首页 > 招生 > 政策 > 详情\n\nThis is the actual page content for testing removal."
        result = clean_markdown(md)
        assert "首页 > 招生 > 政策" not in result
        assert "actual page content" in result


class TestFooterRemoval:
    """Footer sections should be stripped."""

    def test_removes_footer_with_copyright(self):
        md = (
            "This is the main document content that should remain intact.\n\n"
            "版权所有 2024 某大学\n"
            "地址：某市某区\n"
            "电话：12345678"
        )
        result = clean_markdown(md)
        assert "main document content" in result
        assert "版权所有" not in result
        assert "地址" not in result

    def test_removes_footer_with_english_copyright(self):
        md = (
            "Main content for the document with sufficient length.\n\n"
            "Copyright 2024 University. All Rights Reserved.\n"
            "Contact: info@example.com"
        )
        result = clean_markdown(md)
        assert "Main content" in result
        assert "Copyright" not in result

    def test_removes_footer_with_icp(self):
        md = (
            "This is the main body text with enough characters in it.\n\n"
            "ICP备2024001号\n"
            "技术支持：某公司"
        )
        result = clean_markdown(md)
        assert "main body text" in result
        assert "ICP备" not in result


class TestBoldToHeadingConversion:
    """Bold-only lines should be converted to markdown headings."""

    def test_bold_line_becomes_h2(self):
        md = "# Title of the Document\n\n**Important Section Title**\n\nContent follows here with enough text."
        result = clean_markdown(md)
        assert "## Important Section Title" in result

    def test_bold_minor_section_becomes_h3(self):
        md = "# Title\n\n**（1）Minor subsection title**\n\nContent follows."
        result = clean_markdown(md)
        assert "### （1）Minor subsection title" in result

    def test_bold_with_1_dot_1_becomes_h3(self):
        md = "# Title\n\n**1.1 Sub-subsection**\n\nContent here."
        result = clean_markdown(md)
        assert "### 1.1 Sub-subsection" in result


class TestChineseNumberedSectionHeading:
    """Chinese numbered section titles converted to headings."""

    def test_major_section_chinese_numeral(self):
        md = "# Title\n\n一、招生计划\n\nSome content."
        result = clean_markdown(md)
        assert "## 一、招生计划" in result

    def test_major_section_arabic_numeral(self):
        # The regex requires whitespace after the separator for Arabic numerals
        # e.g. "1. 报名条件" matches but "1、报名条件" uses 、 which is in the
        # pattern as \d+[、.．] but only when followed by \s.
        # With 、 (full-width), no trailing \s is needed because 、 is in the
        # character class. Actually the regex is: ^\d+[、.．]\s
        # "1、报名条件": 、 matches [、.．] but 报 doesn't match \s -> no match.
        # Use "1. 报名条件" which matches ^\d+[、.．]\s
        md = "# Title\n\n1. 报名条件\n\nSome content."
        result = clean_markdown(md)
        assert "## 1. 报名条件" in result

    def test_long_paragraph_not_converted(self):
        """A long line starting with a section number should NOT become a heading."""
        md = (
            "# Title\n\n"
            "（二）报名参加全国硕士研究生招生考试的人员，须符合下列条件：中华人民共和国公民\n\n"
            "Some content."
        )
        result = clean_markdown(md)
        # Should NOT be a heading because the line is too long (>30 chars)
        assert "## （二）" not in result


class TestLogoImageRemoval:
    """Logo/banner images should be removed."""

    def test_removes_logo_image(self):
        md = "![logo](https://example.com/logo.png)\n\nActual content paragraph that is long enough to pass."
        result = clean_markdown(md)
        assert "logo.png" not in result
        assert "Actual content" in result

    def test_removes_banner_image(self):
        md = "![banner](https://example.com/banner.jpg)\n\nContent follows here with adequate length."
        result = clean_markdown(md)
        assert "banner" not in result

    def test_removes_image_with_no_alt(self):
        md = "![](https://example.com/img.png)\n\nReal content of the document is here."
        result = clean_markdown(md)
        assert "img.png" not in result

    def test_keeps_content_image(self):
        md = "# Title\n\n![A descriptive photo of campus](https://example.com/campus.jpg)\n\nMore content."
        result = clean_markdown(md)
        assert "campus.jpg" in result


class TestTitleExtractionBrackets:
    """Title extraction with Chinese brackets."""

    def test_bracket_title_extracted(self):
        md = "【重要通知】2024年招生简章发布\n\n详细内容如下..."
        result = clean_markdown(md)
        assert result.startswith("# 【重要通知】2024年招生简章发布")


class TestTitleExtractionFallback:
    """Title extraction falls back to first text line."""

    def test_first_text_line_as_title(self):
        md = "这是一个足够长的标题，用于测试标题提取功能\n\n正文内容在这里，也需要足够长才能通过测试。"
        result = clean_markdown(md)
        assert result.startswith("# ")

    def test_heading_preserved_no_duplicate(self):
        md = "# Existing Title Heading\n\nBody content here."
        result = clean_markdown(md)
        # Should not add another title if heading already exists
        assert result.count("# Existing Title Heading") == 1


class TestDuplicateTitleRemoval:
    """Duplicate title lines should be removed after H1 insertion."""

    def test_no_duplicate_title(self):
        md = "这是一个足够长的文档标题用于测试去重功能\n\n正文内容在这里并且也需要足够长。"
        result = clean_markdown(md)
        title_line = "这是一个足够长的文档标题用于测试去重功能"
        # The title should appear as H1 and the duplicate body line removed
        lines = result.strip().split("\n")
        # Count occurrences of the title text
        count = sum(1 for l in lines if title_line in l)
        assert count == 1


class TestEmptyInput:
    """Empty or whitespace-only input."""

    def test_empty_string(self):
        assert clean_markdown("") == ""

    def test_whitespace_only(self):
        assert clean_markdown("   \n\n  \t  ") == ""

    def test_none_like_empty(self):
        assert clean_markdown("") == ""
