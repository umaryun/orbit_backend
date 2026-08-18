"""
Orbit Backend — Passive fact extractor.

Runs asynchronously after each conversation turn to silently inspect
the dialogue for extractable knowledge: technical decisions, client
rules, architecture choices, gotchas, etc.

Extracted facts are embedded and stored in project_knowledge for
future semantic retrieval (Tier 3 memory).

Uses the google-genai SDK (v2+).
"""

import json
import logging
import uuid

from google import genai
from google.genai import types
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Project, ProjectKnowledge
from app.services.embeddings import generate_embedding

logger = logging.getLogger(__name__)
settings = get_settings()

_client = genai.Client(api_key=settings.gemini_api_key)

EXTRACTION_PROMPT = """You are a knowledge extraction engine. Analyze the following conversation between a developer and their PM assistant (Orbit).

Extract any CONCRETE, SPECIFIC facts that would be valuable to remember for future conversations. Focus on:
1. **Technical decisions**: Tech stack choices, architecture patterns, API contracts, database schemas
2. **Client rules/quirks**: Communication preferences, approval processes, timezone constraints, pet peeves
3. **Gotchas**: Known bugs, workarounds, deployment quirks, integration issues
4. **Architecture**: System design decisions, service boundaries, data flow patterns

Rules:
- Only extract facts that are SPECIFIC and ACTIONABLE — not vague observations
- Each fact should be self-contained (understandable without reading the full conversation)
- If no extractable facts exist, return an empty array
- Do NOT extract task statuses or scheduling info (those are tracked separately)

Return a JSON array of objects with this exact structure:
[
  {
    "category": "technical_decision" | "client_note" | "architecture" | "gotcha" | "api_contract",
    "title": "Short descriptive title",
    "content": "Detailed fact content (2-3 sentences max)"
  }
]

If there are no facts to extract, return: []
"""


async def extract_facts(
    user_id: uuid.UUID,
    conversation_text: str,
    db: AsyncSession,
) -> int:
    """
    Analyze a conversation snippet and extract knowledge facts.

    This runs in the background after each message exchange. It's
    intentionally fire-and-forget — failures are logged but never
    surface to the user.

    Args:
        user_id: UUID of the user.
        conversation_text: The recent conversation text to analyze.
        db: Active database session.

    Returns:
        Number of facts extracted and stored.
    """
    if len(conversation_text.strip()) < 50:
        # Too short to contain meaningful extractable facts
        return 0

    # Find the user's active projects (we'll associate facts with the first active one)
    result = await db.execute(
        select(Project)
        .where(and_(Project.user_id == user_id, Project.status == "active"))
        .order_by(Project.created_at.desc())
        .limit(1)
    )
    active_project = result.scalar_one_or_none()

    if not active_project:
        logger.debug("No active project for user %s — skipping extraction", user_id)
        return 0

    # Ask Gemini to extract facts
    try:
        response = await _client.aio.models.generate_content(
            model=settings.gemini_llm_model,
            contents=f"Conversation to analyze:\n\n{conversation_text}",
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_PROMPT,
                temperature=0.1,  # Low creativity for extraction
                response_mime_type="application/json",
            ),
        )

        raw_text = response.text.strip()

        # Parse the JSON response
        try:
            facts = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response if wrapped in markdown
            if "```" in raw_text:
                json_block = raw_text.split("```")[1]
                if json_block.startswith("json"):
                    json_block = json_block[4:]
                facts = json.loads(json_block.strip())
            else:
                logger.warning("Failed to parse extraction response: %s", raw_text[:200])
                return 0

        if not isinstance(facts, list) or not facts:
            return 0

        # Embed and store each fact
        stored_count = 0
        for fact in facts:
            if not isinstance(fact, dict):
                continue

            category = fact.get("category", "general")
            title = fact.get("title", "")
            content = fact.get("content", "")

            if not content:
                continue

            # Map extracted categories to valid enum values
            category_map = {
                "technical_decision": "technical_decision",
                "client_note": "client_note",
                "architecture": "architecture",
                "gotcha": "gotcha",
                "api_contract": "api_contract",
            }
            db_category = category_map.get(category, "general")

            try:
                embedding = await generate_embedding(
                    f"{title}: {content}" if title else content,
                    task_type="RETRIEVAL_DOCUMENT",
                )

                knowledge = ProjectKnowledge(
                    project_id=active_project.id,
                    category=db_category,
                    title=title,
                    content=content,
                    embedding=embedding,
                )
                db.add(knowledge)
                stored_count += 1
            except Exception as e:
                logger.error("Failed to embed/store fact '%s': %s", title, e)
                continue

        if stored_count > 0:
            await db.commit()
            logger.info(
                "Extracted %d facts for project '%s' (user %s)",
                stored_count,
                active_project.name,
                user_id,
            )

        return stored_count

    except Exception as e:
        logger.error("Passive extraction failed: %s", e, exc_info=True)
        return 0
