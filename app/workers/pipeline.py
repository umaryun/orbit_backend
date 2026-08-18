"""
Orbit Backend — Main background orchestrator pipeline.

This is the non-blocking worker that handles the complete message lifecycle:
  1. Parse webhook payload → extract sender, message type, content
  2. Upsert user in DB
  3. Process by type: text / audio (transcribe) / document (parse+embed)
  4. Log user message
  5. Run agent loop → get response
  6. Log assistant response
  7. Send response via multi-bubble delivery
  8. Fire passive fact extractor in background

This runs entirely in a BackgroundTask — the webhook returns 200 immediately.
"""

import asyncio
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ConversationLog, ProjectKnowledge, User
from app.db.session import async_session_factory
from app.services.agent import run_agent_loop
from app.services.audio import transcribe_audio
from app.services.document_parser import parse_document
from app.services.embeddings import generate_embedding
from app.services.passive_extractor import extract_facts
from app.services.whatsapp import (
    download_media,
    mark_as_read,
    send_multi_bubble,
    send_text_message,
)

logger = logging.getLogger(__name__)


# ── Webhook Payload Parser ────────────────────────────────────────────────────


def extract_message_data(payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extract the relevant message data from a WhatsApp webhook payload.

    Returns None if the payload doesn't contain a processable message.
    """
    try:
        entry = payload.get("entry", [])
        if not entry:
            return None

        changes = entry[0].get("changes", [])
        if not changes:
            return None

        value = changes[0].get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return None

        message = messages[0]
        contacts = value.get("contacts", [])
        contact = contacts[0] if contacts else {}

        return {
            "message_id": message.get("id"),
            "from_number": message.get("from"),
            "timestamp": message.get("timestamp"),
            "type": message.get("type"),
            "text": message.get("text", {}).get("body"),
            "audio": message.get("audio"),
            "document": message.get("document"),
            "image": message.get("image"),
            "contact_name": contact.get("profile", {}).get("name"),
        }
    except (IndexError, KeyError, TypeError) as e:
        logger.error("Failed to parse webhook payload: %s", e)
        return None


# ── User Upsert ────────────────────────────────────────────────────────────────


async def upsert_user(
    phone_number: str,
    first_name: str | None,
    db: AsyncSession,
) -> User:
    """Get or create a user by phone number."""
    result = await db.execute(
        select(User).where(User.phone_number == phone_number)
    )
    user = result.scalar_one_or_none()

    if user:
        if first_name and not user.first_name:
            user.first_name = first_name
            await db.commit()
        return user

    user = User(
        phone_number=phone_number,
        first_name=first_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info("New user created: %s (%s)", phone_number, first_name)
    return user


# ── Conversation Logger ───────────────────────────────────────────────────────


async def log_message(
    user_id: uuid.UUID,
    role: str,
    content: str,
    media_type: str | None,
    db: AsyncSession,
) -> None:
    """Log a conversation turn to the database."""
    log = ConversationLog(
        user_id=user_id,
        role=role,
        content=content,
        media_type=media_type,
    )
    db.add(log)
    await db.commit()


# ── Document Ingestion ─────────────────────────────────────────────────────────


async def ingest_document(
    user_id: uuid.UUID,
    media_id: str,
    filename: str | None,
    mime_type: str | None,
    db: AsyncSession,
) -> str:
    """
    Download, parse, chunk, embed, and store a document in project_knowledge.

    Returns a summary message for the user.
    """
    # Download from WhatsApp
    file_bytes, actual_mime = await download_media(media_id)
    mime = mime_type or actual_mime

    # Parse and chunk
    chunks = parse_document(file_bytes, mime, filename)

    if not chunks:
        return "I received the document but couldn't extract any text from it. Could you try sending it in a different format?"

    # Find user's most recent active project
    from app.db.models import Project
    from sqlalchemy import and_

    result = await db.execute(
        select(Project)
        .where(and_(Project.user_id == user_id, Project.status == "active"))
        .order_by(Project.created_at.desc())
        .limit(1)
    )
    project = result.scalar_one_or_none()

    if not project:
        return f"I parsed {len(chunks)} sections from your document, but you don't have an active project to attach it to. |||Create a project first and then send the doc again?"

    # Embed and store each chunk
    embeddings = await generate_embedding_batch(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        knowledge = ProjectKnowledge(
            project_id=project.id,
            category="prd",
            title=f"{filename or 'Document'} — Section {i + 1}",
            content=chunk,
            embedding=embedding,
        )
        db.add(knowledge)

    await db.commit()

    return f"Got it — I've ingested '{filename or 'your document'}' ({len(chunks)} sections) into {project.name}'s knowledge base. |||Ask me anything about it."


async def generate_embedding_batch(chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of text chunks."""
    from app.services.embeddings import generate_embeddings
    return await generate_embeddings(chunks, task_type="retrieval_document")


# ── Main Pipeline ──────────────────────────────────────────────────────────────


async def process_incoming_message(payload: dict[str, Any]) -> None:
    """
    Main background worker — processes one incoming WhatsApp message end-to-end.

    This function creates its own database session (since it runs in a
    background task, not in a request context).
    """
    # Extract message data
    msg_data = extract_message_data(payload)
    if not msg_data:
        logger.debug("No processable message in payload")
        return

    phone_number = msg_data["from_number"]
    message_type = msg_data["type"]
    message_id = msg_data["message_id"]

    logger.info(
        "Processing message %s from %s (type: %s)",
        message_id,
        phone_number,
        message_type,
    )

    async with async_session_factory() as db:
        try:
            # ── Step 1: Mark as read & upsert user ──
            asyncio.create_task(_safe_mark_read(message_id))

            user = await upsert_user(
                phone_number, msg_data.get("contact_name"), db
            )

            # ── Step 2: Resolve message content by type ──
            user_text: str | None = None
            media_type: str | None = None

            if message_type == "text" and msg_data["text"]:
                user_text = msg_data["text"]
                media_type = "text"

            elif message_type == "audio" and msg_data["audio"]:
                audio_info = msg_data["audio"]
                audio_bytes, mime = await download_media(audio_info["id"])
                user_text = await transcribe_audio(
                    audio_bytes,
                    mime_type=mime,
                    filename=f"voice_{message_id}.ogg",
                )
                media_type = "audio"
                logger.info("Transcribed voice note: %s...", user_text[:100])

            elif message_type == "document" and msg_data["document"]:
                doc_info = msg_data["document"]
                media_type = "document"

                # Ingest document into knowledge base
                doc_response = await ingest_document(
                    user.id,
                    doc_info["id"],
                    doc_info.get("filename"),
                    doc_info.get("mime_type"),
                    db,
                )

                # Log the document event
                await log_message(
                    user.id,
                    "user",
                    f"[Uploaded document: {doc_info.get('filename', 'unknown')}]",
                    media_type,
                    db,
                )
                await log_message(user.id, "assistant", doc_response, None, db)

                # Send the document ingestion response
                await send_multi_bubble(phone_number, doc_response)
                return

            else:
                logger.info("Unsupported message type: %s — ignoring", message_type)
                return

            if not user_text:
                logger.warning("No text extracted from message %s", message_id)
                return

            # ── Step 3: Log user message ──
            await log_message(user.id, "user", user_text, media_type, db)

            # ── Step 4: Run agent loop ──
            response = await run_agent_loop(user.id, user_text, db)

            # ── Step 5: Log assistant response ──
            await log_message(user.id, "assistant", response, None, db)

            # ── Step 6: Send response via multi-bubble ──
            await send_multi_bubble(phone_number, response)

            # ── Step 7: Fire passive extractor (non-blocking) ──
            conversation_snippet = f"User: {user_text}\nAssistant: {response}"
            asyncio.create_task(
                _safe_extract_facts(user.id, conversation_snippet)
            )

            logger.info("Pipeline complete for message %s", message_id)

        except Exception as e:
            logger.error(
                "Pipeline failed for message %s: %s",
                message_id,
                e,
                exc_info=True,
            )
            # Try to send an error message to the user
            try:
                await send_text_message(
                    phone_number,
                    "Hmm, something went wrong on my end. Give me a moment and try again? 🔧",
                )
            except Exception:
                logger.error("Failed to send error message to %s", phone_number)


# ── Safe Background Tasks ─────────────────────────────────────────────────────


async def _safe_mark_read(message_id: str) -> None:
    """Fire-and-forget read receipt."""
    try:
        await mark_as_read(message_id)
    except Exception:
        pass


async def _safe_extract_facts(user_id: uuid.UUID, conversation_text: str) -> None:
    """Fire-and-forget fact extraction with its own DB session."""
    try:
        async with async_session_factory() as db:
            count = await extract_facts(user_id, conversation_text, db)
            if count > 0:
                logger.info("Background extraction: %d facts stored", count)
    except Exception as e:
        logger.error("Background extraction failed: %s", e)
