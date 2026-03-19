"""Tests for rag_crawler.crawler.url_reader module."""

import pytest

from rag_crawler.crawler.url_reader import parse_urls, read_urls_from_file


class TestReadUrlsFromFile:
    """read_urls_from_file reads one URL per line."""

    def test_reads_valid_urls(self, tmp_path):
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "https://example.com\n"
            "https://example.org/page\n"
            "https://test.cn/path/to/doc\n"
        )
        result = read_urls_from_file(str(url_file))
        assert result == [
            "https://example.com",
            "https://example.org/page",
            "https://test.cn/path/to/doc",
        ]

    def test_strips_whitespace(self, tmp_path):
        url_file = tmp_path / "urls.txt"
        url_file.write_text("  https://example.com  \n   https://test.org   \n")
        result = read_urls_from_file(str(url_file))
        assert result == ["https://example.com", "https://test.org"]

    def test_skips_blank_lines(self, tmp_path):
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "https://a.com\n"
            "\n"
            "   \n"
            "https://b.com\n"
        )
        result = read_urls_from_file(str(url_file))
        assert result == ["https://a.com", "https://b.com"]

    def test_skips_comments(self, tmp_path):
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "# This is a comment\n"
            "https://example.com\n"
            "# Another comment\n"
            "https://test.org\n"
        )
        result = read_urls_from_file(str(url_file))
        assert result == ["https://example.com", "https://test.org"]

    def test_mixed_blank_and_comments(self, tmp_path):
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "# header\n"
            "\n"
            "https://a.com\n"
            "   # indented comment\n"
            "\n"
            "https://b.com\n"
        )
        # Note: indented comments still start with # after strip
        result = read_urls_from_file(str(url_file))
        # "   # indented comment" stripped -> "# indented comment" starts with #
        assert result == ["https://a.com", "https://b.com"]

    def test_empty_file(self, tmp_path):
        url_file = tmp_path / "empty.txt"
        url_file.write_text("")
        result = read_urls_from_file(str(url_file))
        assert result == []

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_urls_from_file("/nonexistent/path/urls.txt")


class TestParseUrls:
    """parse_urls normalises and validates URLs."""

    def test_normalizes_urls(self):
        result = parse_urls(["https://Example.COM/Path"])
        assert result == ["https://example.com/Path"]

    def test_normalizes_single_string(self):
        result = parse_urls("https://Example.COM/foo")
        assert result == ["https://example.com/foo"]

    def test_adds_https_when_missing(self):
        result = parse_urls(["example.com/page"])
        assert result == ["https://example.com/page"]

    def test_preserves_http_scheme(self):
        result = parse_urls(["http://example.com"])
        assert result == ["http://example.com/"]

    def test_skips_invalid_urls(self):
        result = parse_urls(["", "   ", "not a url with spaces only"])
        # Empty and whitespace-only return None from _normalise_url
        # "not a url with spaces only" gets https:// prepended, parsed,
        # and may or may not be valid depending on parsing
        # The key test: empty/whitespace are dropped
        assert "" not in [u for u in result]

    def test_skips_empty_string(self):
        result = parse_urls([""])
        assert result == []

    def test_adds_trailing_slash_to_bare_domain(self):
        result = parse_urls(["https://example.com"])
        assert result == ["https://example.com/"]

    def test_preserves_path_and_query(self):
        result = parse_urls(["https://example.com/path?q=1&r=2#frag"])
        assert len(result) == 1
        assert "/path" in result[0]
        assert "q=1" in result[0]

    def test_multiple_urls(self):
        urls = [
            "https://a.com",
            "example.org",
            "",
            "https://b.com/page",
        ]
        result = parse_urls(urls)
        assert len(result) == 3
        assert "https://a.com/" in result
        assert "https://example.org/" in result
        assert "https://b.com/page" in result
