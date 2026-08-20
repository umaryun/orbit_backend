"""
Orbit Backend — Event-Driven Reminder Service.

Handles scheduled proactive notifications without recurring database polling:
  1. Stores reminder in PostgreSQL (persisted state).
  2. Spawns an in-memory asyncio timer targeting the exact execution time.
  3. Restores all pending reminders from PostgreSQL once on server startup.
  4. Delivers the proactive reminder via WhatsApp and updates status to 'delivered'.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ConversationLog, Reminder, User
from app.db.session import async_session_factory
from app.services.whatsapp import send_text_message

logger = logging.getLogger(__name__)

# In-memory registry of active asyncio timer tasks: {reminder_id: Task}
_active_timers: dict[uuid.UUID, asyncio.Task] = {}


async def _deliver_reminder(
    reminder_id: uuid.UUID,
    user_id: uuid.UUID,
    phone_number: str,
    message: str,
) -> None:
    """Deliver a reminder via WhatsApp and record delivery in the DB."""
    try:
        logger.info("Delivering reminder %s to %s: %s", reminder_id, phone_number, message)
        
        # Send proactive message
        reminder_text = f"⏰ Reminder: {message}"
        await send_text_message(phone_number, reminder_text)

        # Update DB status and log to conversation history
        async with async_session_factory() as session:
            result = await session.execute(
                select(Reminder).where(Reminder.id == reminder_id)
            )
            reminder = result.scalar_one_or_none()
            if reminder:
                reminder.status = "delivered"

            # Log to conversation logs so the assistant knows it sent this reminder
            log = ConversationLog(
                user_id=user_id,
                role="assistant",
                content=reminder_text,
                media_type="reminder",
            )
            session.add(log)
            await session.commit()

        logger.info("Reminder %s successfully delivered and logged.", reminder_id)
    except Exception as e:
        logger.error("Failed to deliver reminder %s: %s", reminder_id, e, exc_info=True)
        try:
            async with async_session_factory() as session:
                result = await session.execute(
                    select(Reminder).where(Reminder.id == reminder_id)
                )
                reminder = result.scalar_one_or_none()
                if reminder:
                    reminder.status = "failed"
                    await session.commit()
        except Exception:
            pass
    finally:
        _active_timers.pop(reminder_id, None)


async def _timer_worker(
    reminder_id: uuid.UUID,
    user_id: uuid.UUID,
    phone_number: str,
    message: str,
    delay_seconds: float,
) -> None:
    """Sleep until the target time and execute delivery."""
    try:
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        await _deliver_reminder(reminder_id, user_id, phone_number, message)
    except asyncio.CancelledError:
        logger.debug("Timer for reminder %s was cancelled.", reminder_id)
    except Exception as e:
        logger.error("Error in timer worker for %s: %s", reminder_id, e)


def schedule_reminder_timer(
    reminder_id: uuid.UUID,
    user_id: uuid.UUID,
    phone_number: str,
    message: str,
    remind_at: datetime,
) -> None:
    """
    Schedule an in-memory asyncio timer for a reminder.

    If remind_at is in the past or now, executes immediately.
    """
    now = datetime.now(timezone.utc)
    # Ensure remind_at is timezone-aware
    if remind_at.tzinfo is None:
        remind_at = remind_at.replace(tzinfo=timezone.utc)

    delay_seconds = max(0.0, (remind_at - now).total_seconds())

    # Cancel existing timer if one was running for this reminder
    if reminder_id in _active_timers:
        _active_timers[reminder_id].cancel()

    task = asyncio.create_task(
        _timer_worker(reminder_id, user_id, phone_number, message, delay_seconds)
    )
    _active_timers[reminder_id] = task
    logger.info(
        "Scheduled reminder %s for %s in %.1f seconds",
        reminder_id,
        remind_at.isoformat(),
        delay_seconds,
    )


async def load_and_start_reminders() -> None:
    """
    Load all pending reminders from PostgreSQL and schedule in-memory timers.
    Called once during FastAPI startup.
    """
    async with async_session_factory() as session:
        result = await session.execute(
            select(Reminder, User.phone_number)
            .join(User, Reminder.user_id == User.id)
            .where(Reminder.status == "pending")
        )
        rows = result.all()

        loaded_count = 0
        for reminder, phone_number in rows:
            schedule_reminder_timer(
                reminder_id=reminder.id,
                user_id=reminder.user_id,
                phone_number=phone_number,
                message=reminder.message,
                remind_at=reminder.remind_at,
            )
            loaded_count += 1

        logger.info("Restored and scheduled %d pending reminder(s) from database.", loaded_count)


def cancel_active_timers() -> None:
    """Cancel all active reminder timers during graceful shutdown."""
    count = len(_active_timers)
    for task in _active_timers.values():
        task.cancel()
    _active_timers.clear()
    logger.info("Cancelled %d active reminder timer(s).", count)
