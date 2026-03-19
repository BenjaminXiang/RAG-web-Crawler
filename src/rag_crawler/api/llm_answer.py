"""LLM-augmented answer generation with source citations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_crawler.config import LlmConfig

if TYPE_CHECKING:
    from rag_crawler.store.searcher import SearchResult

logger = logging.getLogger(__name__)


def generate_answer(
    query: str,
    results: list[SearchResult],
    llm_config: LlmConfig,
) -> str:
    """Generate an LLM answer based on retrieved chunks with citations.

    Args:
        query: The user's query.
        results: Search results with text and metadata.
        llm_config: LLM configuration (provider, base_url, api_key, model).

    Returns:
        A formatted answer string with citations.
    """
    if not results:
        return "No relevant context found to answer the query."

    # Build context block with numbered sources
    context_parts: list[str] = []
    for i, r in enumerate(results, 1):
        source = r.source_url or "unknown"
        context_parts.append(f"[{i}] (Source: {source})\n{r.text}")

    context = "\n\n".join(context_parts)

    prompt = (
        "Based on the following retrieved context, answer the user's question. "
        "Cite sources using [N] notation. If the context doesn't contain enough "
        "information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    import openai

    client = openai.OpenAI(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
    )

    model = llm_config.openai_model or llm_config.model
    logger.info("Generating LLM answer with model=%s", model)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    answer = response.choices[0].message.content or ""

    # Append source list
    sources = "\n\nSources:\n"
    seen: set[str] = set()
    for i, r in enumerate(results, 1):
        url = r.source_url or "unknown"
        if url not in seen:
            sources += f"  [{i}] {url}\n"
            seen.add(url)

    return answer + sources
