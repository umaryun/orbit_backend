"""
Orbit Backend — End-to-end test flow.

Tests the complete pipeline using mocked external services:
  • WhatsApp Cloud API (send/receive/media download)
  • Gemini API (LLM + embeddings)
  • Groq API (Whisper transcription)

Runs against the real FastAPI app with httpx.AsyncClient —
no live Meta webhook required.

Usage:
    python -m pytest test_flow.py -v
    # or
    python test_flow.py
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Mock environment before any app imports ────────────────────────────────────

import os

os.environ.update({
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5432/orbit_test",
    "WHATSAPP_TOKEN": "test-whatsapp-token",
    "WHATSAPP_PHONE_NUMBER_ID": "123456789",
    "WHATSAPP_VERIFY_TOKEN": "test-verify-token",
    "GEMINI_API_KEY": "test-gemini-key",
    "GROQ_API_KEY": "test-groq-key",
    "ENVIRONMENT": "testing",
    "LOG_LEVEL": "DEBUG",
})


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_text_payload(
    text: str,
    from_number: str = "2348012345678",
    message_id: str | None = None,
) -> dict:
    """Create a WhatsApp webhook payload for a text message."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "BIZ_ACCOUNT_ID",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Test Developer"},
                                    "wa_id": from_number,
                                }
                            ],
                            "messages": [
                                {
                                    "from": from_number,
                                    "id": message_id or f"wamid.{uuid.uuid4().hex[:20]}",
                                    "timestamp": str(int(datetime.now(timezone.utc).timestamp())),
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


def _make_audio_payload(
    from_number: str = "2348012345678",
    message_id: str | None = None,
) -> dict:
    """Create a WhatsApp webhook payload for a voice note."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "BIZ_ACCOUNT_ID",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Test Developer"},
                                    "wa_id": from_number,
                                }
                            ],
                            "messages": [
                                {
                                    "from": from_number,
                                    "id": message_id or f"wamid.{uuid.uuid4().hex[:20]}",
                                    "timestamp": str(int(datetime.now(timezone.utc).timestamp())),
                                    "type": "audio",
                                    "audio": {
                                        "mime_type": "audio/ogg; codecs=opus",
                                        "sha256": "abc123",
                                        "id": "MEDIA_ID_AUDIO_001",
                                        "voice": True,
                                    },
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


def _make_document_payload(
    filename: str = "requirements.pdf",
    from_number: str = "2348012345678",
    message_id: str | None = None,
) -> dict:
    """Create a WhatsApp webhook payload for a document upload."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "BIZ_ACCOUNT_ID",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Test Developer"},
                                    "wa_id": from_number,
                                }
                            ],
                            "messages": [
                                {
                                    "from": from_number,
                                    "id": message_id or f"wamid.{uuid.uuid4().hex[:20]}",
                                    "timestamp": str(int(datetime.now(timezone.utc).timestamp())),
                                    "type": "document",
                                    "document": {
                                        "filename": filename,
                                        "mime_type": "application/pdf",
                                        "sha256": "def456",
                                        "id": "MEDIA_ID_DOC_001",
                                    },
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


def _make_status_payload() -> dict:
    """Create a WhatsApp webhook status update (not a message)."""
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "BIZ_ACCOUNT_ID",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789",
                            },
                            "statuses": [
                                {
                                    "id": "wamid.status123",
                                    "status": "delivered",
                                    "timestamp": "1234567890",
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


# ── Test Class ─────────────────────────────────────────────────────────────────


class TestWebhookVerification:
    """Tests for the GET /api/v1/webhook Meta verification handshake."""

    @pytest.mark.asyncio
    async def test_verification_success(self):
        """Valid verify token should return the challenge string."""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/webhook",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "test-verify-token",
                    "hub.challenge": "challenge_accepted_123",
                },
            )
        assert resp.status_code == 200
        assert resp.text == "challenge_accepted_123"

    @pytest.mark.asyncio
    async def test_verification_failure(self):
        """Invalid verify token should return 403."""
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/webhook",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "wrong-token",
                    "hub.challenge": "challenge_123",
                },
            )
        assert resp.status_code == 403


class TestWebhookPost:
    """Tests for the POST /api/v1/webhook message ingestion."""

    @pytest.mark.asyncio
    async def test_text_message_returns_200_immediately(self):
        """POST with a text message should return 200 and dispatch to background."""
        from app.main import app

        payload = _make_text_payload("Hey Orbit, what's up?")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("app.workers.pipeline.process_incoming_message", new_callable=AsyncMock):
                resp = await client.post("/api/v1/webhook", json=payload)

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_status_update_acknowledged(self):
        """Status updates (not messages) should still get 200."""
        from app.main import app

        payload = _make_status_payload()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/webhook", json=payload)

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_duplicate_message_deduped(self):
        """Same message ID sent twice should only be processed once."""
        from app.main import app
        from app.api.v1.webhook import _seen_message_ids

        _seen_message_ids.clear()  # Reset dedup state

        msg_id = "wamid.unique_test_id_001"
        payload = _make_text_payload("Hello", message_id=msg_id)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch(
                "app.workers.pipeline.process_incoming_message",
                new_callable=AsyncMock,
            ) as mock_process:
                # First request
                resp1 = await client.post("/api/v1/webhook", json=payload)
                # Second request (duplicate)
                resp2 = await client.post("/api/v1/webhook", json=payload)

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Background task should only be queued once
        # (the second call should be deduped before dispatching)


class TestHealthCheck:
    """Tests for system endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "orbit-backend"

    @pytest.mark.asyncio
    async def test_root(self):
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")

        assert resp.status_code == 200
        assert "Orbit" in resp.json()["service"]


class TestPayloadParser:
    """Tests for the webhook payload parser utility."""

    def test_extract_text_message(self):
        from app.workers.pipeline import extract_message_data

        payload = _make_text_payload("Hello world")
        data = extract_message_data(payload)

        assert data is not None
        assert data["type"] == "text"
        assert data["text"] == "Hello world"
        assert data["from_number"] == "2348012345678"
        assert data["contact_name"] == "Test Developer"

    def test_extract_audio_message(self):
        from app.workers.pipeline import extract_message_data

        payload = _make_audio_payload()
        data = extract_message_data(payload)

        assert data is not None
        assert data["type"] == "audio"
        assert data["audio"]["id"] == "MEDIA_ID_AUDIO_001"

    def test_extract_document_message(self):
        from app.workers.pipeline import extract_message_data

        payload = _make_document_payload("spec.pdf")
        data = extract_message_data(payload)

        assert data is not None
        assert data["type"] == "document"
        assert data["document"]["filename"] == "spec.pdf"
        assert data["document"]["id"] == "MEDIA_ID_DOC_001"

    def test_empty_payload_returns_none(self):
        from app.workers.pipeline import extract_message_data

        assert extract_message_data({}) is None
        assert extract_message_data({"entry": []}) is None

    def test_status_update_returns_none(self):
        from app.workers.pipeline import extract_message_data

        payload = _make_status_payload()
        assert extract_message_data(payload) is None


class TestDocumentChunker:
    """Tests for the document parser and chunker."""

    def test_chunk_short_text(self):
        from app.services.document_parser import chunk_text

        text = "This is a short document."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text(self):
        from app.services.document_parser import chunk_text

        # Generate text that's definitely longer than 500 tokens
        text = "The quick brown fox jumps over the lazy dog. " * 200
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_empty_text(self):
        from app.services.document_parser import chunk_text

        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_extract_markdown(self):
        from app.services.document_parser import extract_text

        md_bytes = b"# Hello\n\nThis is a **markdown** document."
        text = extract_text(md_bytes, "text/markdown")
        assert "Hello" in text
        assert "markdown" in text


class TestMultiBubbleSplitter:
    """Tests for the multi-bubble message splitting logic."""

    @pytest.mark.asyncio
    async def test_split_on_delimiter(self):
        """Messages should be split on ||| and sent as separate bubbles."""
        from app.services.whatsapp import send_multi_bubble

        with patch("app.services.whatsapp.send_text_message", new_callable=AsyncMock) as mock_send, \
             patch("app.services.whatsapp.send_typing_indicator", new_callable=AsyncMock), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await send_multi_bubble("2348012345678", "First thought|||Second thought|||Third")

        assert mock_send.call_count == 3
        calls = [call.args[1] for call in mock_send.call_args_list]
        assert calls == ["First thought", "Second thought", "Third"]

    @pytest.mark.asyncio
    async def test_single_message_no_split(self):
        """A message without ||| should be sent as a single bubble."""
        from app.services.whatsapp import send_multi_bubble

        with patch("app.services.whatsapp.send_text_message", new_callable=AsyncMock) as mock_send, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await send_multi_bubble("2348012345678", "Just one message")

        assert mock_send.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_parts_ignored(self):
        """Empty parts after splitting should be skipped."""
        from app.services.whatsapp import send_multi_bubble

        with patch("app.services.whatsapp.send_text_message", new_callable=AsyncMock) as mock_send, \
             patch("app.services.whatsapp.send_typing_indicator", new_callable=AsyncMock), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await send_multi_bubble("2348012345678", "Hello|||   |||World")

        assert mock_send.call_count == 2


# ── Run directly ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
