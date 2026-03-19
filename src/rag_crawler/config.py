"""Configuration loader for RAG Web Crawler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class CrawlerConfig:
    rate_limit: float = 1.0
    concurrency: int = 1
    timeout: int = 30
    user_agent: str = "RAG-web-Crawler/0.1.0"
    respect_robots_txt: bool = True
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ProcessorConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    output_dir: str = "./output/markdown"


@dataclass
class MilvusConfig:
    uri: str = "http://localhost:19530"
    collection_name: str = "rag_chunks"


@dataclass
class EmbeddingConfig:
    provider: str = "local"
    model: str = "all-MiniLM-L6-v2"
    openai_model: Optional[str] = None


@dataclass
class StoreConfig:
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    export_dir: str = "./output/jsonl"


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class LlmConfig:
    provider: str = "local"  # "none", "local", "openai"
    base_url: str = "http://star.sustech.edu.cn/service/model/qwen35/v1"
    api_key: str = "k8#pL2@mN9!qjfkew87@#$0204"
    model: str = "qwen3"
    openai_model: Optional[str] = None


@dataclass
class AppConfig:
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)


def _build_dataclass(cls: type, data: dict[str, Any] | None):
    """Recursively build a dataclass from a dict, ignoring unknown keys."""
    if not data:
        return cls()
    filtered = {}
    for f in cls.__dataclass_fields__:
        if f not in data:
            continue
        field_type = cls.__dataclass_fields__[f].type
        # Resolve nested dataclass fields by checking default_factory
        default = cls.__dataclass_fields__[f].default_factory  # type: ignore[arg-type]
        if default is not field.__class__ and callable(default):
            nested_cls = type(default())
            if hasattr(nested_cls, "__dataclass_fields__") and isinstance(data[f], dict):
                filtered[f] = _build_dataclass(nested_cls, data[f])
                continue
        filtered[f] = data[f]
    return cls(**filtered)


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults."""
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    with open(config_path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return _build_dataclass(AppConfig, raw)
