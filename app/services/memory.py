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
import zoneinfo
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (
    ConversationLog,
    Project,
    ProjectKnowledge,
    Task,
    User,
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
        lines = [f"📁 **{project.name}** (Project ID: `{project.id}`)"]
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
                lines.append(f"     {status_icon} {t.title} [Task ID: `{t.id}`]{priority_str}{due_str}")
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
    user: User | None = None,
) -> tuple[list[dict[str, str]], str]:
    """
    Assemble all 3 tiers into a prompt-ready format.

    Runs Tier 1, Tier 2, and Tier 3 concurrently via asyncio.gather
    to minimize total latency.

    Args:
        user_id: UUID of the user.
        user_message: The current message text (used for semantic search).
        db: Active database session.
        user: Optional pre-loaded User object to avoid re-querying.

    Returns:
        Tuple of (conversation_history, injected_context_block)
        - conversation_history: list of role/content dicts for chat history
        - injected_context_block: string block to inject into the system prompt
    """
    # NOTE: These run sequentially on the same AsyncSession since
    # SQLAlchemy's AsyncSession is not safe for concurrent coroutines.
    # The main perf win comes from eliminating the double context build
    # (User model eager-loading + explicit queries).
    conversation = await get_short_term_context(user_id, db)
    relational = await get_relational_context(user_id, db)
    semantic = await get_semantic_context(user_id, user_message, db)

    # Build the injected context block
    context_parts = []

    # Use passed-in user or fetch if not provided
    if not user:
        user_result = await db.execute(select(User).where(User.id == user_id))
        user = user_result.scalar_one_or_none()

    now_utc = datetime.now(timezone.utc)
    user_tz_str = user.timezone if user and user.timezone else "UTC"

    try:
        user_tz = zoneinfo.ZoneInfo(user_tz_str)
        now_local = now_utc.astimezone(user_tz)
        local_time_str = now_local.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")
    except Exception:
        local_time_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    default_name = user.first_name if user and user.first_name else "Unknown"
    preferred_name = user.preferred_name if user and user.preferred_name else "Not set"
    active_name = user.preferred_name if (user and user.preferred_name) else (user.first_name if (user and user.first_name) else "there")

    user_info_lines = [
        f"Default Name (from WhatsApp): {default_name}",
        f"Preferred Name: {preferred_name}",
        f"Active Name to Address User: {active_name}",
        f"Current UTC Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"User Local Time: {local_time_str}",
    ]
    context_parts.append("=== USER INFO & CURRENT TIME ===\n" + "\n".join(user_info_lines))

    if relational and relational != "No active projects.":
        context_parts.append(f"=== ACTIVE PROJECTS & TASKS ===\n{relational}")

    if semantic:
        context_parts.append(f"=== RELEVANT PROJECT KNOWLEDGE ===\n{semantic}")

    injected_context = "\n\n".join(context_parts) if context_parts else ""

    return conversation, injected_context

