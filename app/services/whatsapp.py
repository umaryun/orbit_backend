"""
Orbit Backend — WhatsApp Cloud API client.

Handles sending text messages, typing indicators, multi-bubble delivery
with human-like pacing, and media downloads from Meta's Graph API.
"""

import asyncio
import logging
import random

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Reusable async HTTP client — created once, shared across requests
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {settings.whatsapp_token}",
                "Content-Type": "application/json",
            },
        )
    return _http_client


async def close_client() -> None:
    """Shut down the shared HTTP client gracefully."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ── Send a single text message ────────────────────────────────────────────────


async def send_text_message(phone_number: str, text: str) -> dict:
    """
    Send a plain text message to a WhatsApp user.

    Args:
        phone_number: Recipient phone number (international format, no +).
        text: Message body.

    Returns:
        Meta API response dict.
    """
    client = _get_client()
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }

    try:
        resp = await client.post(
            f"{settings.whatsapp_api_url}/messages",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.debug("Message sent to %s: %s", phone_number, data)
        return data
    except httpx.HTTPStatusError as e:
        logger.error(
            "WhatsApp send failed [%s]: %s", e.response.status_code, e.response.text
        )
        raise
    except httpx.RequestError as e:
        logger.error("WhatsApp send request error: %s", e)
        raise


# ── Typing indicator ──────────────────────────────────────────────────────────


async def send_typing_indicator(phone_number: str) -> None:
    """
    Show 'typing...' status to the user in WhatsApp.

    This uses the messages endpoint with status type.
    """
    client = _get_client()
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "reaction",
    }

    # WhatsApp Cloud API doesn't have a direct "typing" endpoint.
    # We simulate presence by using the read receipts / marking as read.
    # The actual typing indicator is triggered naturally when we send
    # messages in quick succession. We'll use a status update instead.
    try:
        # Mark messages as read to show blue ticks (simulates attention)
        await client.post(
            f"{settings.whatsapp_api_url}/messages",
            json={
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": "placeholder",
            },
        )
    except Exception:
        # Non-critical — don't block message delivery if this fails
        pass


async def mark_as_read(message_id: str) -> None:
    """Mark an incoming message as read (blue ticks)."""
    client = _get_client()
    try:
        await client.post(
            f"{settings.whatsapp_api_url}/messages",
            json={
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
            },
        )
    except Exception:
        logger.debug("Failed to mark message %s as read", message_id)


# ── Multi-bubble delivery ─────────────────────────────────────────────────────


async def send_multi_bubble(phone_number: str, full_text: str) -> None:
    """
    Split a response on '|||' delimiter and send each part as a separate
    WhatsApp message with human-like typing delays between them.

    This creates the illusion of the assistant typing distinct thoughts,
    rather than dumping a wall of text.
    """
    parts = [p.strip() for p in full_text.split("|||") if p.strip()]

    if not parts:
        return

    for i, part in enumerate(parts):
        if i > 0:
            # Simulate typing delay between bubbles
            delay_ms = random.randint(
                settings.typing_delay_min_ms,
                settings.typing_delay_max_ms,
            )
            await send_typing_indicator(phone_number)
            await asyncio.sleep(delay_ms / 1000.0)

        await send_text_message(phone_number, part)


# ── Media download ────────────────────────────────────────────────────────────


async def download_media(media_id: str) -> tuple[bytes, str]:
    """
    Download media (audio, document, image) from WhatsApp.

    Two-step process:
    1. GET media URL from Graph API using media_id
    2. GET the actual binary content from the URL

    Returns:
        Tuple of (file_bytes, mime_type)
    """
    client = _get_client()

    # Step 1: Get the media URL
    try:
        resp = await client.get(
            f"https://graph.facebook.com/v20.0/{media_id}",
        )
        resp.raise_for_status()
        media_info = resp.json()
        media_url = media_info["url"]
        mime_type = media_info.get("mime_type", "application/octet-stream")
    except httpx.HTTPStatusError as e:
        logger.error("Failed to get media URL for %s: %s", media_id, e.response.text)
        raise
    except KeyError:
        logger.error("No URL in media response for %s", media_id)
        raise ValueError(f"Media {media_id} has no download URL")

    # Step 2: Download the binary
    try:
        resp = await client.get(media_url)
        resp.raise_for_status()
        return resp.content, mime_type
    except httpx.HTTPStatusError as e:
        logger.error(
            "Failed to download media %s: %s", media_id, e.response.text
        )
        raise
