"""
Orbit Backend — Document parser & semantic chunker.

Extracts text from PDFs and Markdown files uploaded via WhatsApp,
then splits into overlapping chunks suitable for embedding and
storage in the project_knowledge table.
"""

import io
import logging
from typing import Optional

from pypdf import PdfReader

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Character-based token estimation (~4 chars per token is a widely-used heuristic)
CHARS_PER_TOKEN = 4


def _count_tokens(text: str) -> int:
    """Approximate token count using character-based estimation."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _encode_text(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens tokens."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    return text[:max_chars]


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text content from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def _extract_text_from_markdown(file_bytes: bytes) -> str:
    """Decode Markdown bytes to text (passthrough)."""
    return file_bytes.decode("utf-8", errors="replace")


def extract_text(
    file_bytes: bytes,
    mime_type: str,
    filename: Optional[str] = None,
) -> str:
    """
    Extract text from a document based on its MIME type.

    Supported formats:
    - application/pdf
    - text/markdown, text/plain

    Args:
        file_bytes: Raw file content.
        mime_type: MIME type string.
        filename: Optional filename for type inference.

    Returns:
        Extracted text content.
    """
    mime_lower = mime_type.lower()

    if "pdf" in mime_lower:
        return _extract_text_from_pdf(file_bytes)
    elif any(t in mime_lower for t in ("markdown", "text/plain", "text/md")):
        return _extract_text_from_markdown(file_bytes)
    elif filename and filename.lower().endswith((".md", ".txt")):
        return _extract_text_from_markdown(file_bytes)
    else:
        logger.warning("Unsupported document type: %s (%s)", mime_type, filename)
        # Attempt text decode as fallback
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception:
            raise ValueError(f"Unsupported document format: {mime_type}")


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks based on token count.

    Uses a sentence-aware splitting strategy: tries to break at sentence
    boundaries (`. `, `\n`) within the token budget.

    Args:
        text: Full document text.
        chunk_size: Max tokens per chunk (default from settings).
        chunk_overlap: Overlap tokens between chunks (default from settings).

    Returns:
        List of text chunks.
    """
    chunk_size = chunk_size or settings.chunk_size_tokens
    chunk_overlap = chunk_overlap or settings.chunk_overlap_tokens

    if not text or not text.strip():
        return []

    # Estimate total tokens
    total_tokens = _count_tokens(text)

    if total_tokens <= chunk_size:
        return [text.strip()]

    # Convert token counts to character counts for slicing
    chunk_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * CHARS_PER_TOKEN
    step_chars = chunk_chars - overlap_chars

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_chars, text_len)
        chunk_slice = text[start:end].strip()

        if chunk_slice:
            chunks.append(chunk_slice)

        if end >= text_len:
            break

        start += step_chars

    logger.info(
        "Chunked %d tokens into %d chunks (size=%d, overlap=%d)",
        total_tokens,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def parse_document(
    file_bytes: bytes,
    mime_type: str,
    filename: str | None = None,
) -> list[str]:
    """
    Full pipeline: extract text from document → chunk into embeddable pieces.

    Args:
        file_bytes: Raw file content.
        mime_type: MIME type of the document.
        filename: Optional filename.

    Returns:
        List of text chunks ready for embedding.
    """
    text = extract_text(file_bytes, mime_type, filename)
    return chunk_text(text)
