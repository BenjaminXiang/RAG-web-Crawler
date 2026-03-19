"""Tests for rag_crawler.store module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, call

import pytest

from rag_crawler.config import EmbeddingConfig, MilvusConfig, StoreConfig
from rag_crawler.processor.chunker import Chunk
from rag_crawler.processor.processor import ProcessedDocument


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def milvus_config():
    return MilvusConfig(uri="http://localhost:19530", collection_name="test_chunks")


@pytest.fixture
def embedding_config():
    return EmbeddingConfig(provider="local", model="all-MiniLM-L6-v2")


@pytest.fixture
def store_config(milvus_config, embedding_config):
    return StoreConfig(milvus=milvus_config, embedding=embedding_config, export_dir="/tmp/export")


@pytest.fixture
def sample_doc():
    return ProcessedDocument(
        url="https://example.com/page1",
        title="Test Page",
        folder_path="/tmp/output/test-page",
        markdown="# Test Page\n\nSome content here.",
        chunks=[
            Chunk(text="Test Page", chunk_index=0, start_char=0, end_char=9),
            Chunk(text="Some content here.", chunk_index=1, start_char=12, end_char=30),
        ],
        links=[],
        attachments=[],
        crawled_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture
def sample_docs(sample_doc):
    doc2 = ProcessedDocument(
        url="https://example.com/page2",
        title="Second Page",
        folder_path="/tmp/output/second-page",
        markdown="# Second\n\nMore text.",
        chunks=[
            Chunk(text="More text.", chunk_index=0, start_char=0, end_char=10),
        ],
        crawled_at="2026-01-02T00:00:00+00:00",
    )
    return [sample_doc, doc2]


def _fake_embed(texts, config):
    """Return deterministic fake embeddings (dim=4)."""
    return [[float(i)] * 4 for i in range(len(texts))]


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    """embed_texts with local and openai backends."""

    def test_empty_input_returns_empty(self, embedding_config):
        from rag_crawler.store.embedding import embed_texts
        assert embed_texts([], embedding_config) == []

    @patch("rag_crawler.store.embedding._get_local_model")
    def test_local_backend_calls_encode(self, mock_get_model, embedding_config):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        mock_get_model.return_value = mock_model

        from rag_crawler.store.embedding import embed_texts
        result = embed_texts(["hello", "world"], embedding_config)

        mock_model.encode.assert_called_once_with(["hello", "world"], show_progress_bar=False)
        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])

    @patch("openai.OpenAI")
    def test_openai_backend_calls_api(self, MockOpenAI):
        config = EmbeddingConfig(provider="openai", openai_model="text-embedding-3-small")

        mock_item1 = MagicMock()
        mock_item1.embedding = [0.5, 0.6]
        mock_item2 = MagicMock()
        mock_item2.embedding = [0.7, 0.8]
        mock_response = MagicMock()
        mock_response.data = [mock_item1, mock_item2]
        MockOpenAI.return_value.embeddings.create.return_value = mock_response

        from rag_crawler.store.embedding import _embed_openai
        result = _embed_openai(["a", "b"], config)

        assert result == [[0.5, 0.6], [0.7, 0.8]]


class TestGetEmbeddingDim:
    """get_embedding_dim returns correct dimensions."""

    def test_openai_known_model(self):
        from rag_crawler.store.embedding import get_embedding_dim
        config = EmbeddingConfig(provider="openai", openai_model="text-embedding-3-small")
        assert get_embedding_dim(config) == 1536

    def test_openai_large_model(self):
        from rag_crawler.store.embedding import get_embedding_dim
        config = EmbeddingConfig(provider="openai", openai_model="text-embedding-3-large")
        assert get_embedding_dim(config) == 3072

    @patch("rag_crawler.store.embedding._get_local_model")
    def test_local_model_queries_dim(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_get_model.return_value = mock_model

        from rag_crawler.store.embedding import get_embedding_dim
        config = EmbeddingConfig(provider="local", model="all-MiniLM-L6-v2")
        assert get_embedding_dim(config) == 384


# ---------------------------------------------------------------------------
# Milvus client tests
# ---------------------------------------------------------------------------

class TestMilvusClient:
    """milvus_client connection and collection management."""

    @patch("rag_crawler.store.milvus_client.MilvusClient")
    def test_get_client_connects(self, MockClient, milvus_config):
        from rag_crawler.store.milvus_client import get_client
        client = get_client(milvus_config)
        MockClient.assert_called_once_with(uri="http://localhost:19530")

    @patch("rag_crawler.store.milvus_client.MilvusClient")
    def test_ensure_collection_skips_if_exists(self, MockClient, milvus_config):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True

        from rag_crawler.store.milvus_client import ensure_collection
        name = ensure_collection(mock_client, milvus_config)

        assert name == "test_chunks"
        mock_client.create_collection.assert_not_called()

    @patch("rag_crawler.store.milvus_client.MilvusClient")
    def test_ensure_collection_creates_if_missing(self, MockClient, milvus_config):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client.prepare_index_params.return_value = MagicMock()

        from rag_crawler.store.milvus_client import ensure_collection
        name = ensure_collection(mock_client, milvus_config, embedding_dim=384)

        assert name == "test_chunks"
        mock_client.create_collection.assert_called_once()
        mock_client.create_index.assert_called_once()

    @patch("rag_crawler.store.milvus_client.MilvusClient")
    def test_drop_collection_when_exists(self, MockClient, milvus_config):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True

        from rag_crawler.store.milvus_client import drop_collection
        drop_collection(mock_client, milvus_config)

        mock_client.drop_collection.assert_called_once_with("test_chunks")

    @patch("rag_crawler.store.milvus_client.MilvusClient")
    def test_drop_collection_noop_when_missing(self, MockClient, milvus_config):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False

        from rag_crawler.store.milvus_client import drop_collection
        drop_collection(mock_client, milvus_config)

        mock_client.drop_collection.assert_not_called()


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------

class TestStoreDocuments:
    """store_documents batch writing and incremental update."""

    @patch("rag_crawler.store.writer.embed_texts", side_effect=_fake_embed)
    @patch("rag_crawler.store.writer.get_embedding_dim", return_value=4)
    @patch("rag_crawler.store.writer.ensure_collection", return_value="test_chunks")
    @patch("rag_crawler.store.writer.get_client")
    def test_stores_all_chunks(self, mock_get_client, mock_ensure, mock_dim, mock_embed, store_config, sample_doc):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        from rag_crawler.store.writer import store_documents
        count = store_documents([sample_doc], store_config)

        assert count == 2
        # Delete called before insert (incremental update)
        mock_client.delete.assert_called_once()
        mock_client.insert.assert_called_once()
        insert_data = mock_client.insert.call_args[1]["data"]
        assert len(insert_data) == 2
        assert insert_data[0]["source_url"] == "https://example.com/page1"
        assert insert_data[0]["chunk_index"] == 0
        assert insert_data[1]["chunk_index"] == 1

    @patch("rag_crawler.store.writer.embed_texts", side_effect=_fake_embed)
    @patch("rag_crawler.store.writer.get_embedding_dim", return_value=4)
    @patch("rag_crawler.store.writer.ensure_collection", return_value="test_chunks")
    @patch("rag_crawler.store.writer.get_client")
    def test_stores_multiple_documents(self, mock_get_client, mock_ensure, mock_dim, mock_embed, store_config, sample_docs):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        from rag_crawler.store.writer import store_documents
        count = store_documents(sample_docs, store_config)

        assert count == 3  # 2 chunks from doc1 + 1 chunk from doc2
        assert mock_client.delete.call_count == 2
        assert mock_client.insert.call_count == 2

    @patch("rag_crawler.store.writer.get_embedding_dim", return_value=4)
    @patch("rag_crawler.store.writer.ensure_collection", return_value="test_chunks")
    @patch("rag_crawler.store.writer.get_client")
    def test_empty_documents_returns_zero(self, mock_get_client, mock_ensure, mock_dim, store_config):
        from rag_crawler.store.writer import store_documents
        assert store_documents([], store_config) == 0

    @patch("rag_crawler.store.writer.embed_texts", side_effect=_fake_embed)
    @patch("rag_crawler.store.writer.get_embedding_dim", return_value=4)
    @patch("rag_crawler.store.writer.ensure_collection", return_value="test_chunks")
    @patch("rag_crawler.store.writer.get_client")
    def test_skips_doc_with_no_chunks(self, mock_get_client, mock_ensure, mock_dim, mock_embed, store_config):
        doc = ProcessedDocument(
            url="https://example.com/empty",
            title="Empty",
            folder_path="/tmp/empty",
            markdown="",
            chunks=[],
            crawled_at="2026-01-01T00:00:00+00:00",
        )
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        from rag_crawler.store.writer import store_documents
        count = store_documents([doc], store_config)

        assert count == 0
        mock_client.insert.assert_not_called()

    @patch("rag_crawler.store.writer.embed_texts", side_effect=_fake_embed)
    @patch("rag_crawler.store.writer.get_embedding_dim", return_value=4)
    @patch("rag_crawler.store.writer.ensure_collection", return_value="test_chunks")
    @patch("rag_crawler.store.writer.get_client")
    def test_incremental_delete_uses_correct_filter(self, mock_get_client, mock_ensure, mock_dim, mock_embed, store_config, sample_doc):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        from rag_crawler.store.writer import store_documents
        store_documents([sample_doc], store_config)

        mock_client.delete.assert_called_once_with(
            collection_name="test_chunks",
            filter='source_url == "https://example.com/page1"',
        )


# ---------------------------------------------------------------------------
# Searcher tests
# ---------------------------------------------------------------------------

class TestSearch:
    """search with different modes and filters."""

    @patch("rag_crawler.store.searcher.embed_texts")
    @patch("rag_crawler.store.searcher.get_client")
    def test_vector_search(self, mock_get_client, mock_embed, store_config):
        mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_client = MagicMock()
        mock_client.search.return_value = [[
            {"entity": {"text": "result text", "source_url": "https://ex.com", "title": "T", "chunk_index": 0, "crawled_at": "2026-01-01"}, "distance": 0.95},
        ]]
        mock_get_client.return_value = mock_client

        from rag_crawler.store.searcher import search
        results = search("test query", store_config, top_k=5, mode="vector")

        assert len(results) == 1
        assert results[0].text == "result text"
        assert results[0].score == 0.95
        mock_client.search.assert_called_once()

    @patch("rag_crawler.store.searcher.get_client")
    def test_keyword_search(self, mock_get_client, store_config):
        mock_client = MagicMock()
        mock_client.search.return_value = [[
            {"entity": {"text": "keyword hit", "source_url": "https://ex.com", "title": "T", "chunk_index": 1, "crawled_at": ""}, "distance": 5.2},
        ]]
        mock_get_client.return_value = mock_client

        from rag_crawler.store.searcher import search
        results = search("test query", store_config, top_k=3, mode="keyword")

        assert len(results) == 1
        assert results[0].text == "keyword hit"
        # Keyword search passes query string directly to sparse field
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["anns_field"] == "sparse_embedding"
        assert call_kwargs["data"] == ["test query"]

    @patch("rag_crawler.store.searcher.embed_texts")
    @patch("rag_crawler.store.searcher.get_client")
    def test_hybrid_search(self, mock_get_client, mock_embed, store_config):
        mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_client = MagicMock()
        mock_client.hybrid_search.return_value = [[
            {"entity": {"text": "hybrid hit", "source_url": "https://ex.com", "title": "T", "chunk_index": 0, "crawled_at": ""}, "distance": 0.88},
        ]]
        mock_get_client.return_value = mock_client

        from rag_crawler.store.searcher import search
        results = search("test query", store_config, top_k=5, mode="hybrid")

        assert len(results) == 1
        assert results[0].text == "hybrid hit"
        mock_client.hybrid_search.assert_called_once()

    @patch("rag_crawler.store.searcher.embed_texts")
    @patch("rag_crawler.store.searcher.get_client")
    def test_empty_results(self, mock_get_client, mock_embed, store_config):
        mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_client = MagicMock()
        mock_client.search.return_value = [[]]
        mock_get_client.return_value = mock_client

        from rag_crawler.store.searcher import search
        results = search("nothing", store_config, top_k=5, mode="vector")
        assert results == []


class TestBuildFilter:
    """_build_filter constructs correct expressions."""

    def test_no_filters(self):
        from rag_crawler.store.searcher import _build_filter
        assert _build_filter(None, None, None) == ""

    def test_url_filter(self):
        from rag_crawler.store.searcher import _build_filter
        result = _build_filter(["https://a.com", "https://b.com"], None, None)
        assert 'source_url in ["https://a.com", "https://b.com"]' == result

    def test_time_range_filter(self):
        from rag_crawler.store.searcher import _build_filter
        result = _build_filter(None, "2026-01-01", "2026-12-31")
        assert 'crawled_at >= "2026-01-01"' in result
        assert 'crawled_at <= "2026-12-31"' in result

    def test_combined_filters(self):
        from rag_crawler.store.searcher import _build_filter
        result = _build_filter(["https://a.com"], "2026-01-01", None)
        assert "source_url in" in result
        assert "crawled_at >=" in result
        assert " and " in result


# ---------------------------------------------------------------------------
# Exporter tests
# ---------------------------------------------------------------------------

class TestExportJsonl:
    """export_jsonl writes correct JSONL format."""

    @patch("rag_crawler.store.exporter.get_client")
    def test_export_with_embedding(self, mock_get_client, store_config, tmp_path):
        mock_client = MagicMock()
        mock_client.query.return_value = [
            {"text": "chunk1", "source_url": "https://a.com", "title": "T", "chunk_index": 0, "crawled_at": "2026-01-01", "embedding": [0.1, 0.2]},
            {"text": "chunk2", "source_url": "https://a.com", "title": "T", "chunk_index": 1, "crawled_at": "2026-01-01", "embedding": [0.3, 0.4]},
        ]
        mock_get_client.return_value = mock_client

        out_file = str(tmp_path / "export.jsonl")

        from rag_crawler.store.exporter import export_jsonl
        result = export_jsonl(store_config, output_path=out_file, include_embedding=True)

        assert result == out_file
        lines = open(out_file).readlines()
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert row["text"] == "chunk1"
        assert row["metadata"]["source_url"] == "https://a.com"
        assert row["embedding"] == [0.1, 0.2]

    @patch("rag_crawler.store.exporter.get_client")
    def test_export_without_embedding(self, mock_get_client, store_config, tmp_path):
        mock_client = MagicMock()
        mock_client.query.return_value = [
            {"text": "chunk1", "source_url": "https://a.com", "title": "T", "chunk_index": 0, "crawled_at": "2026-01-01"},
        ]
        mock_get_client.return_value = mock_client

        out_file = str(tmp_path / "export_no_emb.jsonl")

        from rag_crawler.store.exporter import export_jsonl
        result = export_jsonl(store_config, output_path=out_file, include_embedding=False)

        lines = open(out_file).readlines()
        row = json.loads(lines[0])
        assert "embedding" not in row
        assert row["text"] == "chunk1"

    @patch("rag_crawler.store.exporter.get_client")
    def test_export_empty_collection(self, mock_get_client, store_config, tmp_path):
        mock_client = MagicMock()
        mock_client.query.return_value = []
        mock_get_client.return_value = mock_client

        out_file = str(tmp_path / "empty.jsonl")

        from rag_crawler.store.exporter import export_jsonl
        result = export_jsonl(store_config, output_path=out_file)

        lines = open(out_file).readlines()
        assert len(lines) == 0

    @patch("rag_crawler.store.exporter.get_client")
    def test_export_default_path(self, mock_get_client, store_config, tmp_path):
        mock_client = MagicMock()
        mock_client.query.return_value = [
            {"text": "x", "source_url": "", "title": "", "chunk_index": 0, "crawled_at": ""},
        ]
        mock_get_client.return_value = mock_client

        store_config.export_dir = str(tmp_path / "jsonl_out")

        from rag_crawler.store.exporter import export_jsonl
        result = export_jsonl(store_config)

        assert "export_full.jsonl" in result
        assert (tmp_path / "jsonl_out" / "export_full.jsonl").exists()
