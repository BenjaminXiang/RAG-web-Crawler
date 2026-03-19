"""Tests for rag_crawler.config module."""

import textwrap

import pytest
import yaml

from rag_crawler.config import (
    AppConfig,
    CrawlerConfig,
    LlmConfig,
    MilvusConfig,
    ProcessorConfig,
    StoreConfig,
    _build_dataclass,
    load_config,
)


class TestLoadConfigValidYaml:
    """load_config with a valid YAML file."""

    def test_loads_top_level_fields(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            crawler:
              rate_limit: 2.5
              concurrency: 4
              timeout: 60
            processor:
              chunk_size: 256
              output_dir: /tmp/out
            """)
        )
        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, AppConfig)
        assert cfg.crawler.rate_limit == 2.5
        assert cfg.crawler.concurrency == 4
        assert cfg.crawler.timeout == 60
        assert cfg.processor.chunk_size == 256
        assert cfg.processor.output_dir == "/tmp/out"

    def test_loads_nested_store_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            store:
              milvus:
                uri: http://milvus:19530
                collection_name: my_chunks
              embedding:
                provider: openai
                model: text-embedding-3-small
              export_dir: /data/export
            """)
        )
        cfg = load_config(str(cfg_file))
        assert cfg.store.milvus.uri == "http://milvus:19530"
        assert cfg.store.milvus.collection_name == "my_chunks"
        assert cfg.store.embedding.provider == "openai"
        assert cfg.store.export_dir == "/data/export"

    def test_unknown_keys_ignored(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            crawler:
              rate_limit: 3.0
              unknown_key: should_be_ignored
            totally_unknown:
              foo: bar
            """)
        )
        cfg = load_config(str(cfg_file))
        assert cfg.crawler.rate_limit == 3.0
        # Defaults preserved for unspecified fields
        assert cfg.crawler.concurrency == 1


class TestLoadConfigMissingFile:
    """load_config with a missing file returns defaults."""

    def test_returns_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(cfg, AppConfig)
        assert cfg.crawler.rate_limit == 1.0
        assert cfg.crawler.concurrency == 1
        assert cfg.crawler.timeout == 30
        assert cfg.processor.chunk_size == 512
        assert cfg.processor.chunk_overlap == 50
        assert cfg.store.milvus.uri == "http://localhost:19530"

    def test_defaults_have_correct_types(self, tmp_path):
        cfg = load_config(str(tmp_path / "nope.yaml"))
        assert isinstance(cfg.crawler, CrawlerConfig)
        assert isinstance(cfg.processor, ProcessorConfig)
        assert isinstance(cfg.store, StoreConfig)
        assert isinstance(cfg.store.milvus, MilvusConfig)
        assert isinstance(cfg.llm, LlmConfig)


class TestBuildDataclass:
    """_build_dataclass nested construction."""

    def test_none_data_returns_defaults(self):
        result = _build_dataclass(CrawlerConfig, None)
        assert result.rate_limit == 1.0

    def test_empty_dict_returns_defaults(self):
        result = _build_dataclass(CrawlerConfig, {})
        assert result.concurrency == 1

    def test_partial_override(self):
        result = _build_dataclass(CrawlerConfig, {"timeout": 99})
        assert result.timeout == 99
        assert result.rate_limit == 1.0  # default preserved

    def test_nested_dataclass_build(self):
        data = {
            "milvus": {"uri": "http://custom:1234"},
            "export_dir": "/custom",
        }
        result = _build_dataclass(StoreConfig, data)
        assert result.milvus.uri == "http://custom:1234"
        assert result.milvus.collection_name == "rag_chunks"  # default
        assert result.export_dir == "/custom"

    def test_empty_yaml_file(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, AppConfig)
        assert cfg.crawler.rate_limit == 1.0
