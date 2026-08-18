"""
Orbit Backend — Audio transcription via Groq Whisper API.

Downloads voice note bytes from WhatsApp, sends to Groq's
whisper-large-v3 endpoint, returns the transcription text.
Free tier friendly.
"""

import io
import logging

from groq import AsyncGroq

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_groq_client: AsyncGroq | None = None


def _get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


async def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/ogg",
    filename: str = "voice_note.ogg",
) -> str:
    """
    Transcribe audio bytes using Groq's Whisper API.

    WhatsApp voice notes are typically OGG/Opus format. Groq's Whisper
    API handles this natively.

    Args:
        audio_bytes: Raw audio file content.
        mime_type: MIME type of the audio (e.g., 'audio/ogg').
        filename: Filename to pass to the API.

    Returns:
        Transcription text.
    """
    client = _get_groq()

    # Wrap bytes in a file-like object with a name
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    try:
        transcription = await client.audio.transcriptions.create(
            file=(filename, audio_file),
            model="whisper-large-v3",
            response_format="text",
            language="en",
        )

        result = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
        logger.info("Transcribed %d bytes → %d chars", len(audio_bytes), len(result))
        return result

    except Exception as e:
        logger.error("Whisper transcription failed: %s", e)
        raise
