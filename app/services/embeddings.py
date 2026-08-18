"""
Orbit Backend — Embedding generation via Gemini text-embedding-004.

Generates 768-dimensional vectors for semantic search in pgvector.
Free tier via Google AI Studio API key.

Uses the google-genai SDK (v2+).
"""

import logging

from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create a reusable client instance
_client = genai.Client(api_key=settings.gemini_api_key)


async def generate_embedding(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    """
    Generate a single embedding vector for the given text.

    Args:
        text: Input text to embed.
        task_type: One of 'RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY',
                   'SEMANTIC_SIMILARITY', 'CLASSIFICATION', 'CLUSTERING'.
                   Use 'RETRIEVAL_DOCUMENT' when storing, 'RETRIEVAL_QUERY'
                   when searching.

    Returns:
        768-dimensional float vector.
    """
    try:
        result = await _client.aio.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
            ),
        )
        embedding = result.embeddings[0].values
        logger.debug("Generated embedding: %d dimensions", len(embedding))
        return list(embedding)
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        raise


async def generate_embeddings(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts in a batch.

    Args:
        texts: List of texts to embed.
        task_type: Embedding task type.

    Returns:
        List of 768-dimensional float vectors.
    """
    if not texts:
        return []

    try:
        result = await _client.aio.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
            ),
        )
        embeddings = [list(e.values) for e in result.embeddings]
        logger.info("Generated %d embeddings in batch", len(embeddings))
        return embeddings
    except Exception as e:
        logger.error("Batch embedding generation failed: %s", e)
        raise
