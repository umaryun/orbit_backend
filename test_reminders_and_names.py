"""
Test script for scheduled reminders and preferred_name support.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from sqlalchemy import select

from app.db.models import Reminder, User
from app.db.session import async_session_factory, init_db
from app.services.memory import assemble_full_context
from app.services.reminders import (
    cancel_active_timers,
    load_and_start_reminders,
    schedule_reminder_timer,
)
from app.tools.definitions import execute_tool


async def run_tests():
    print("=== Step 1: Initializing DB ===")
    await init_db()
    print("DB initialized successfully.")

    test_phone = f"234{uuid.uuid4().hex[:10]}"
    async with async_session_factory() as db:
        user = User(
            phone_number=test_phone,
            first_name="Umar",
            timezone="Africa/Lagos",
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        user_id = user.id

    print(f"Created test user: {user_id} with name 'Umar'")

    # Test 1: Check assemble_full_context injects default name and local time
    print("\n=== Step 2: Testing Context Assembly ===")
    async with async_session_factory() as db:
        history, context = await assemble_full_context(user_id, "hello", db)
        print("Injected Context:\n", context)
        assert "Default Name (from WhatsApp): Umar" in context
        assert "Preferred Name: Not set" in context
        assert "Active Name to Address User: Umar" in context
        assert "Africa/Lagos" in context or "WAT" in context or "UTC" in context
        print("Context assembly test passed.")

    # Test 2: Execute set_preferred_name tool
    print("\n=== Step 3: Testing set_preferred_name tool ===")
    async with async_session_factory() as db:
        result_json = await execute_tool(
            "set_preferred_name",
            {"preferred_name": "Chief Dev"},
            user_id,
            db,
        )
        result = json.loads(result_json)
        print("Tool Result:", result)
        assert result["status"] == "updated"
        assert result["preferred_name"] == "Chief Dev"

        # Verify in DB
        res = await db.execute(select(User).where(User.id == user_id))
        u = res.scalar_one()
        assert u.preferred_name == "Chief Dev"

        # Check context assembly now uses preferred name
        _, updated_context = await assemble_full_context(user_id, "hello", db)
        print("Updated Context:\n", updated_context)
        assert "Preferred Name: Chief Dev" in updated_context
        assert "Active Name to Address User: Chief Dev" in updated_context
        print("set_preferred_name test passed.")

    # Test 3: Execute set_reminder tool
    print("\n=== Step 4: Testing set_reminder tool & delivery ===")
    mock_send = AsyncMock(return_value={"messages": [{"id": "wamid.test"}]})

    with patch("app.services.reminders.send_text_message", mock_send):
        remind_time = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()
        async with async_session_factory() as db:
            result_json = await execute_tool(
                "set_reminder",
                {
                    "message": "Push the staging migration",
                    "remind_at": remind_time,
                },
                user_id,
                db,
            )
            result = json.loads(result_json)
            print("Set Reminder Tool Result:", result)
            assert result["status"] == "scheduled"
            reminder_id = uuid.UUID(result["reminder_id"])

        # Wait for delivery to complete
        print("Waiting for delivery to complete...")
        delivered = False
        for _ in range(10):
            await asyncio.sleep(0.5)
            async with async_session_factory() as db:
                res = await db.execute(select(Reminder).where(Reminder.id == reminder_id))
                rem = res.scalar_one()
                if rem.status == "delivered":
                    delivered = True
                    break

        assert delivered, f"Expected reminder status 'delivered', got '{rem.status}'"
        print(f"Reminder in DB status: {rem.status}")

        # Verify send_text_message was called
        mock_send.assert_called_once()
        args, kwargs = mock_send.call_args
        print(f"send_text_message called with: {args}")
        assert args[0] == test_phone
        assert "Push the staging migration" in args[1]

    print("set_reminder test passed.")

    # Clean up
    cancel_active_timers()
    print("\n All tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(run_tests())
