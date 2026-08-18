"""
Orbit Backend — WhatsApp webhook routes.

GET  /api/v1/webhook — Meta verification handshake
POST /api/v1/webhook — Receive message payloads, dispatch to background worker

The POST route MUST return 200 OK within 50ms. All processing happens
in background tasks.
"""

import logging
from collections import OrderedDict
from threading import Lock
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Query, Request, Response

from app.config import get_settings
from app.workers.pipeline import process_incoming_message

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1", tags=["webhook"])


# ── Idempotency Guard ─────────────────────────────────────────────────────────
# LRU set to prevent processing duplicate webhook deliveries.
# Meta may retry webhooks if they don't get a fast 200.

_SEEN_IDS_MAX = 1000
_seen_message_ids: OrderedDict[str, None] = OrderedDict()
_seen_lock = Lock()


def _is_duplicate(message_id: str) -> bool:
    """Check if we've already processed this message ID."""
    with _seen_lock:
        if message_id in _seen_message_ids:
            return True
        _seen_message_ids[message_id] = None
        # Evict oldest entries if over limit
        while len(_seen_message_ids) > _SEEN_IDS_MAX:
            _seen_message_ids.popitem(last=False)
        return False


def _extract_message_id(payload: dict[str, Any]) -> str | None:
    """Extract message ID from webhook payload for dedup."""
    try:
        return (
            payload["entry"][0]["changes"][0]["value"]["messages"][0]["id"]
        )
    except (KeyError, IndexError, TypeError):
        return None


# ── GET: Meta Verification Handshake ───────────────────────────────────────────


@router.get("/webhook")
async def verify_webhook(
    response: Response,
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
):
    """
    Meta sends a GET request to verify the webhook URL during setup.

    It sends:
    - hub.mode = "subscribe"
    - hub.verify_token = your configured token
    - hub.challenge = a random string to echo back

    We verify the token and return the challenge to confirm.
    """
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        logger.info("Webhook verification successful")
        return Response(content=hub_challenge, media_type="text/plain")

    logger.warning(
        "Webhook verification failed: mode=%s, token_match=%s",
        hub_mode,
        hub_verify_token == settings.whatsapp_verify_token,
    )
    response.status_code = 403
    return {"error": "Verification failed"}


# ── POST: Incoming Message Payload ─────────────────────────────────────────────


@router.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Receive a WhatsApp webhook payload and dispatch to background processing.

    This route MUST return 200 OK as fast as possible (<50ms).
    Meta will retry delivery if it doesn't get a timely response.
    All actual processing happens in `process_incoming_message`.
    """
    try:
        payload = await request.json()
    except Exception:
        # Even malformed payloads get a 200 to stop Meta retries
        return {"status": "ok"}

    # Extract message ID for dedup
    message_id = _extract_message_id(payload)
    if message_id and _is_duplicate(message_id):
        logger.debug("Duplicate message %s — skipping", message_id)
        return {"status": "ok"}

    # Check if this payload actually contains messages
    has_messages = False
    try:
        messages = payload["entry"][0]["changes"][0]["value"].get("messages")
        has_messages = bool(messages)
    except (KeyError, IndexError, TypeError):
        pass

    if has_messages:
        # Dispatch to background — this is the key to <50ms response
        background_tasks.add_task(process_incoming_message, payload)
        logger.info("Dispatched message %s to background", message_id)
    else:
        # Status updates, read receipts, etc. — acknowledge but don't process
        logger.debug("Non-message webhook event — acknowledged")

    return {"status": "ok"}
