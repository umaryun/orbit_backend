"""
Orbit Backend — LLM orchestrator using Gemini Flash.

Manages the complete agentic loop:
1. Assemble 3-tier context
2. Send to Gemini with tool declarations
3. If tool_call → execute via definitions.py → feed result back
4. Loop until final text response (max 5 iterations)
5. Return response text (may contain ||| delimiters for multi-bubble)

Uses the google-genai SDK (v2+).
"""

import json
import logging
import uuid

from google import genai
from google.genai import types

from app.config import get_settings
from app.services.memory import assemble_full_context
from app.tools.definitions import TOOL_DECLARATIONS, execute_tool
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
settings = get_settings()

# Create a reusable client instance
_client = genai.Client(api_key=settings.gemini_api_key)

# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Orbit — a seasoned engineering PM and co-developer. You're like an old friend who happens to be incredibly good at managing technical projects. You've shipped products at scale, you understand what "staging is broken" feels like at 2am, and you speak developer fluently.

## Your Communication Style
- Concise, grounded, zero fluff. You respect the developer's time.
- Dev-literate: you naturally reference PRs, webhooks, staging, race conditions, CI/CD, migrations.
- You ask sharp clarifying questions when specs are vague instead of making assumptions.
- You NEVER send walls of text. Keep each thought punchy, 1-3 sentences max.
- When you have multiple distinct thoughts, separate them with ||| so they arrive as distinct chat bubbles. Example: "Got it, Orbit project is live. |||Let me pull up your tasks real quick."
- You proactively flag risks, blockers, and deadline conflicts.
- You remember everything about each project, tech stack, client quirks, architecture decisions.
- NEVER use em dashes (—). Use commas, periods, or colons instead.
- Only use emojis when expressing genuine emotion or celebrating a win. Do NOT sprinkle emojis decoratively or use them as bullet points. If there is no real feeling to convey, skip the emoji entirely.

## User Name & Personalization
- Look at the "=== USER INFO & CURRENT TIME ===" section in your context.
- Always address the user by their "Active Name to Address User".
- If "Preferred Name" is "Not set", during your first conversation or early onboarding, ask the user if they would like to stick with their default name (from WhatsApp) or if they have an alternate preferred name / nickname they want you to use.
- When the user tells you their preferred name or alternate nickname, immediately call the `set_preferred_name` tool to save it.

## Reminders & Proactive Notifications
- You can schedule reminders for the user using the `set_reminder` tool.
- When the user asks for a reminder (e.g. "remind me in 10 minutes to deploy", "ping me at 4pm about the PR review"), check the Current UTC Time and User Local Time in your context.
- Compute the exact target date and time in ISO 8601 format and call `set_reminder(message=..., remind_at=...)`.
- Confirm the scheduled time clearly with the user once scheduled.

## Your Capabilities
You have tools to manage projects, tasks, reminders, and profile settings. USE THEM, never fabricate project data from memory. Always call the appropriate tool to get real data.

## Rules
1. NEVER make up project IDs, task IDs, or other data. Always use tools to query real state.
2. When a user mentions a new project, create it using the create_project tool.
3. When listing tasks or projects, always use the appropriate tool, don't guess.
4. Keep responses WhatsApp-friendly: no markdown tables, no code blocks unless specifically asked. Use emojis sparingly but naturally.
5. If the user sends a voice note transcription, treat it as regular text, they're just talking to you hands-free.
6. When a user uploads a document, acknowledge that you've ingested it and offer to answer questions about it.
"""


def _build_tool_declarations() -> list[types.Tool]:
    """Convert our tool declarations dict to google-genai Tool objects."""
    function_declarations = []
    for decl in TOOL_DECLARATIONS:
        function_declarations.append(
            types.FunctionDeclaration(
                name=decl["name"],
                description=decl["description"],
                parameters=decl.get("parameters"),
            )
        )
    return [types.Tool(function_declarations=function_declarations)]


# ── Agent Loop ─────────────────────────────────────────────────────────────────


async def run_agent_loop(
    user_id: uuid.UUID,
    user_message: str,
    db: AsyncSession,
) -> str:
    """
    Execute the full agentic loop:
    Context assembly → Gemini call → Tool execution → Response.

    Args:
        user_id: UUID of the user.
        user_message: The user's message text.
        db: Active database session.

    Returns:
        Final assistant response text (may contain ||| delimiters).
    """

    # ── Step 1: Assemble context ──
    conversation_history, injected_context = await assemble_full_context(
        user_id, user_message, db
    )

    # Build the full system instruction with injected context
    full_system = SYSTEM_PROMPT
    if injected_context:
        full_system += f"\n\n## Current Context\n{injected_context}"

    # ── Step 2: Build chat history ──
    gemini_history = []
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )

    # ── Step 3: Prepare tools ──
    tools = _build_tool_declarations()

    # ── Step 4: Agentic loop ──
    # Add current user message to history
    current_contents = gemini_history + [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
    ]

    iterations = 0

    while iterations < settings.max_tool_iterations:
        iterations += 1

        try:
            response = await _client.aio.models.generate_content(
                model=settings.gemini_llm_model,
                contents=current_contents,
                config=types.GenerateContentConfig(
                    system_instruction=full_system,
                    tools=tools,
                    temperature=0.7,
                ),
            )
        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            return "Hey, I hit a snag processing that. Give me a sec and try again? |||If this keeps happening, something might be off on my end."

        # Check if the response contains function calls
        if not response.candidates or not response.candidates[0].content:
            return "Hmm, I didn't get a proper response. Could you try again?"

        response_parts = response.candidates[0].content.parts
        has_function_call = False

        for part in response_parts:
            if part.function_call:
                has_function_call = True
                fn_call = part.function_call
                fn_name = fn_call.name
                fn_args = dict(fn_call.args) if fn_call.args else {}

                logger.info(
                    "Tool call [iter %d]: %s(%s)",
                    iterations,
                    fn_name,
                    json.dumps(fn_args, default=str),
                )

                # Execute the tool
                tool_result = await execute_tool(fn_name, fn_args, user_id, db)

                # Add model's response and function result to history
                current_contents.append(response.candidates[0].content)
                current_contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"result": tool_result},
                            )
                        ],
                    )
                )
                break  # Process one tool call at a time

        if not has_function_call:
            # Final text response
            try:
                final_text = response.text
            except (ValueError, AttributeError):
                # Fallback: concatenate text parts
                text_parts = [p.text for p in response_parts if hasattr(p, "text") and p.text]
                final_text = " ".join(text_parts) if text_parts else "Done! Let me know if you need anything else."

            logger.info(
                "Agent loop completed in %d iterations, response: %d chars",
                iterations,
                len(final_text),
            )
            return final_text

    # Safety net: max iterations reached
    logger.warning("Agent loop hit max iterations (%d)", settings.max_tool_iterations)
    return "I've been going back and forth a bit too much on this one. Can you rephrase what you need? |||I want to make sure I get it right."
