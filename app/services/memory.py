"""
Orbit Backend — 3-Tier context assembly for LLM prompts.

Combines:
  Tier 1 (Short-term): Last N conversation turns from conversation_logs
  Tier 2 (Relational):  Active projects, profiles, upcoming tasks
  Tier 3 (Semantic):    pgvector cosine similarity search on project_knowledge

The assembled context becomes part of the system/user prompt sent to Gemini.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (
    ConversationLog,
    Project,
    ProjectKnowledge,
    Task,
)
from app.services.embeddings import generate_embedding

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Tier 1: Short-Term Buffer ─────────────────────────────────────────────────


async def get_short_term_context(
    user_id: uuid.UUID,
    db: AsyncSession,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """
    Retrieve the last N conversation turns for the user.

    Returns list of dicts: [{"role": "user", "content": "..."}, ...]
    """
    limit = limit or settings.short_term_memory_limit

    result = await db.execute(
        select(ConversationLog)
        .where(ConversationLog.user_id == user_id)
        .order_by(ConversationLog.created_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()

    # Reverse to chronological order
    return [
        {"role": log.role, "content": log.content}
        for log in reversed(logs)
    ]


# ── Tier 2: Relational Context ────────────────────────────────────────────────


async def get_relational_context(
    user_id: uuid.UUID,
    db: AsyncSession,
) -> str:
    """
    Build a structured summary of the user's active projects and upcoming tasks.

    This gives the LLM awareness of current state without requiring tool calls
    for basic context.
    """
    # Get active projects with profiles
    result = await db.execute(
        select(Project)
        .where(and_(Project.user_id == user_id, Project.status == "active"))
        .order_by(Project.created_at.desc())
    )
    projects = result.scalars().all()

    if not projects:
        return "No active projects."

    sections = []
    now = datetime.now(timezone.utc)
    upcoming_cutoff = now + timedelta(days=7)

    for project in projects:
        lines = [f"📁 **{project.name}**"]
        if project.client_name:
            lines.append(f"   Client: {project.client_name}")

        if project.profile:
            p = project.profile
            if p.tech_stack:
                stack = ", ".join(p.tech_stack) if isinstance(p.tech_stack, list) else str(p.tech_stack)
                lines.append(f"   Stack: {stack}")
            if p.primary_goal:
                lines.append(f"   Goal: {p.primary_goal}")
            if p.client_quirks:
                lines.append(f"   Client notes: {p.client_quirks}")

        # Upcoming/overdue tasks for this project
        task_result = await db.execute(
            select(Task)
            .where(
                and_(
                    Task.project_id == project.id,
                    Task.status.in_(["todo", "in_progress", "blocked"]),
                )
            )
            .order_by(Task.due_at.asc().nullslast())
            .limit(5)
        )
        tasks = task_result.scalars().all()

        if tasks:
            lines.append("   Tasks:")
            for t in tasks:
                status_icon = {"todo": "⬜", "in_progress": "🔄", "blocked": "🚫"}.get(t.status, "❓")
                due_str = ""
                if t.due_at:
                    if t.due_at < now:
                        due_str = f" ⚠️ OVERDUE (was {t.due_at.strftime('%b %d')})"
                    else:
                        due_str = f" (due {t.due_at.strftime('%b %d')})"
                priority_str = f" [{t.priority}]" if t.priority in ("high", "urgent") else ""
                lines.append(f"     {status_icon} {t.title}{priority_str}{due_str}")
                if t.blocker_reason:
                    lines.append(f"       Blocked: {t.blocker_reason}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# ── Tier 3: Semantic Context (pgvector RAG) ────────────────────────────────────


async def get_semantic_context(
    user_id: uuid.UUID,
    query: str,
    db: AsyncSession,
    top_k: int | None = None,
) -> str:
    """
    Retrieve top-K relevant knowledge chunks via cosine similarity search.

    Searches across ALL active projects for the user.
    """
    top_k = top_k or settings.semantic_search_top_k

    # Generate query embedding
    query_embedding = await generate_embedding(query, task_type="RETRIEVAL_QUERY")

    # Find active project IDs for this user
    project_result = await db.execute(
        select(Project.id).where(
            and_(Project.user_id == user_id, Project.status == "active")
        )
    )
    project_ids = [row[0] for row in project_result.all()]

    if not project_ids:
        return ""

    # Semantic search across all active projects
    result = await db.execute(
        select(ProjectKnowledge)
        .where(ProjectKnowledge.project_id.in_(project_ids))
        .order_by(ProjectKnowledge.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    chunks = result.scalars().all()

    if not chunks:
        return ""

    sections = []
    for chunk in chunks:
        label = f"[{chunk.category}]"
        if chunk.title:
            label += f" {chunk.title}"
        sections.append(f"{label}\n{chunk.content}")

    return "\n---\n".join(sections)


# ── Full Context Assembly ──────────────────────────────────────────────────────


async def assemble_full_context(
    user_id: uuid.UUID,
    user_message: str,
    db: AsyncSession,
) -> tuple[list[dict[str, str]], str]:
    """
    Assemble all 3 tiers into a prompt-ready format.

    Returns:
        Tuple of (conversation_history, injected_context_block)
        - conversation_history: list of role/content dicts for chat history
        - injected_context_block: string block to inject into the system prompt
    """
    # Tier 1: Conversation history
    conversation = await get_short_term_context(user_id, db)

    # Tier 2: Relational state
    relational = await get_relational_context(user_id, db)

    # Tier 3: Semantic knowledge (search with current message)
    semantic = await get_semantic_context(user_id, user_message, db)

    # Build the injected context block
    context_parts = []

    if relational and relational != "No active projects.":
        context_parts.append(f"=== ACTIVE PROJECTS & TASKS ===\n{relational}")

    if semantic:
        context_parts.append(f"=== RELEVANT PROJECT KNOWLEDGE ===\n{semantic}")

    injected_context = "\n\n".join(context_parts) if context_parts else ""

    return conversation, injected_context
