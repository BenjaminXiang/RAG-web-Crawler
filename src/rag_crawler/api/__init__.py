"""REST API module for serving retrieval endpoints."""

from rag_crawler.api.app import create_app
from rag_crawler.api.llm_answer import generate_answer

__all__ = ["create_app", "generate_answer"]
