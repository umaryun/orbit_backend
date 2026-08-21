"""
Orbit Backend — Pydantic tool definitions and database handler functions.

This module defines the ONLY interface between the LLM and the database.
The LLM calls tools by name with validated arguments; each tool executes
safe, deterministic ORM operations. Zero raw SQL ever touches the LLM.

Tool Registry:
  • create_project       — Create a new project with optional profile
  • update_project_profile — Update project tech stack, goals, quirks
  • create_task           — Add a task to a project
  • update_task_status    — Move task status, set blockers
  • list_tasks            — Query tasks by project/status/timeframe
  • list_projects         — List user's projects by status
  • query_project_knowledge — Semantic search over project knowledge
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Project,
    ProjectProfile,
    Task,
    ProjectKnowledge,
    Reminder,
    User,
)
from app.services.embeddings import generate_embedding
from app.services.reminders import schedule_reminder_timer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pydantic Argument Schemas (strict validation before DB touch)
# ═══════════════════════════════════════════════════════════════════════════════


class CreateProjectArgs(BaseModel):
    name: str = Field(description="Project name")
    client_name: Optional[str] = Field(None, description="Client or company name")
    tech_stack: Optional[list[str]] = Field(None, description="List of technologies (e.g. ['React', 'FastAPI', 'PostgreSQL'])")
    primary_goal: Optional[str] = Field(None, description="The main deliverable or objective")
    client_quirks: Optional[str] = Field(None, description="Client preferences, communication style, special rules")


class UpdateProjectProfileArgs(BaseModel):
    project_id: str = Field(description="UUID of the project to update")
    tech_stack: Optional[list[str]] = Field(None, description="Updated tech stack list")
    repo_url: Optional[str] = Field(None, description="Repository URL")
    deployment_target: Optional[str] = Field(None, description="Deployment platform (e.g. 'Vercel', 'AWS ECS')")
    client_quirks: Optional[str] = Field(None, description="Updated client quirks/rules")
    primary_goal: Optional[str] = Field(None, description="Updated primary goal")


class CreateTaskArgs(BaseModel):
    project_id: str = Field(description="UUID of the project")
    title: str = Field(description="Task title/description")
    priority: str = Field("medium", description="Priority: low, medium, high, urgent")
    due_at: Optional[str] = Field(None, description="Due date in ISO format (e.g. '2024-12-31T23:59:00Z')")


class UpdateTaskStatusArgs(BaseModel):
    task_id: str = Field(description="UUID of the task")
    status: str = Field(description="New status: todo, in_progress, blocked, done")
    blocker_reason: Optional[str] = Field(None, description="Reason for blocker (required when status is 'blocked')")


class ListTasksArgs(BaseModel):
    project_id: Optional[str] = Field(None, description="Filter by project UUID")
    status: Optional[str] = Field(None, description="Filter by status: todo, in_progress, blocked, done")
    timeframe: Optional[str] = Field(None, description="Filter timeframe: 'today', 'this_week', 'overdue'")


class ListProjectsArgs(BaseModel):
    status: Optional[str] = Field(None, description="Filter by status: active, paused, completed, archived")


class QueryProjectKnowledgeArgs(BaseModel):
    project_id: str = Field(description="UUID of the project to search")
    search_query: str = Field(description="Natural language search query")


class SetReminderArgs(BaseModel):
    message: str = Field(description="What the user wants to be reminded of")
    remind_at: str = Field(
        description="Exact date and time to send the reminder in ISO 8601 format (e.g. '2026-08-20T22:30:00Z' or '2026-08-20T22:30:00+01:00')"
    )


class SetPreferredNameArgs(BaseModel):
    preferred_name: str = Field(
        description="The preferred name or nickname the user wants the assistant to call them"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Gemini Function Declarations (exported for agent.py)
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_DECLARATIONS = [
    {
        "name": "create_project",
        "description": "Create a new project for the user. Use this when they mention starting a new project, client engagement, or freelance gig.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Project name"},
                "client_name": {"type": "string", "description": "Client or company name"},
                "tech_stack": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of technologies",
                },
                "primary_goal": {"type": "string", "description": "Main deliverable or objective"},
                "client_quirks": {"type": "string", "description": "Client preferences or special rules"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "update_project_profile",
        "description": "Update a project's profile with new tech stack, repo URL, deployment target, client quirks, or primary goal.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "UUID of the project"},
                "tech_stack": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Updated tech stack",
                },
                "repo_url": {"type": "string", "description": "Repository URL"},
                "deployment_target": {"type": "string", "description": "Deployment platform"},
                "client_quirks": {"type": "string", "description": "Client quirks/rules"},
                "primary_goal": {"type": "string", "description": "Primary goal"},
            },
            "required": ["project_id"],
        },
    },
    {
        "name": "create_task",
        "description": "Create a new task under a project. Use when the user mentions something they need to do, a deliverable, or an action item.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "UUID of the project"},
                "title": {"type": "string", "description": "Task title/description"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "Priority level",
                },
                "due_at": {"type": "string", "description": "Due date in ISO format"},
            },
            "required": ["project_id", "title"],
        },
    },
    {
        "name": "update_task_status",
        "description": "Update a task's status. Use when the user says they finished, started, or are blocked on something.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "UUID of the task"},
                "status": {
                    "type": "string",
                    "enum": ["todo", "in_progress", "blocked", "done"],
                    "description": "New status",
                },
                "blocker_reason": {"type": "string", "description": "Reason for blocker"},
            },
            "required": ["task_id", "status"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List tasks, optionally filtered by project, status, or timeframe. Use when the user asks about their tasks, what's pending, or what's overdue.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Filter by project UUID"},
                "status": {
                    "type": "string",
                    "enum": ["todo", "in_progress", "blocked", "done"],
                    "description": "Filter by status",
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["today", "this_week", "overdue"],
                    "description": "Filter timeframe",
                },
            },
        },
    },
    {
        "name": "list_projects",
        "description": "List the user's projects, optionally filtered by status. Use when they ask about their projects or need an overview.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "paused", "completed", "archived"],
                    "description": "Filter by project status",
                },
            },
        },
    },
    {
        "name": "query_project_knowledge",
        "description": "Search the project's knowledge base using semantic similarity. Use when the user asks about architecture decisions, client notes, PRD details, or any stored project knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "UUID of the project"},
                "search_query": {"type": "string", "description": "Natural language search query"},
            },
            "required": ["project_id", "search_query"],
        },
    },
    {
        "name": "set_reminder",
        "description": "Schedule a reminder message to be sent proactively to the user at a specified time. Use when the user asks to be reminded of something at a specific time or in X minutes/hours.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The reminder text to send to the user",
                },
                "remind_at": {
                    "type": "string",
                    "description": "The exact ISO 8601 datetime to trigger the reminder (e.g. '2026-08-20T22:30:00+01:00')",
                },
            },
            "required": ["message", "remind_at"],
        },
    },
    {
        "name": "set_preferred_name",
        "description": "Save or update the user's preferred name or nickname. Use when the user tells you what name they prefer to be called.",
        "parameters": {
            "type": "object",
            "properties": {
                "preferred_name": {
                    "type": "string",
                    "description": "The preferred name or nickname",
                },
            },
            "required": ["preferred_name"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper Resolvers (Resilient ID and Name Resolution)
# ═══════════════════════════════════════════════════════════════════════════════


async def _resolve_project_id(
    identifier: str | None,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> uuid.UUID | None:
    """
    Resolve a project UUID from a UUID string, project name, or default active project.
    Never throws an exception on bad input.
    """
    if not identifier:
        result = await db.execute(
            select(Project.id).where(and_(Project.user_id == user_id, Project.status == "active"))
        )
        active_projects = result.scalars().all()
        return active_projects[0] if len(active_projects) == 1 else None

    # 1. Try parsing as direct UUID
    try:
        parsed_uuid = uuid.UUID(identifier.strip())
        result = await db.execute(
            select(Project.id).where(and_(Project.id == parsed_uuid, Project.user_id == user_id))
        )
        if result.scalar_one_or_none():
            return parsed_uuid
    except (ValueError, AttributeError):
        pass

    # 2. Try exact or partial name match
    cleaned = identifier.strip().replace("_", " ").replace("-", " ")
    for suffix in ["uuid placeholder", "placeholder", "uuid", "id"]:
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()

    if cleaned:
        result = await db.execute(
            select(Project.id).where(
                and_(
                    Project.user_id == user_id,
                    Project.name.ilike(f"%{cleaned}%"),
                )
            )
        )
        matching = result.scalars().all()
        if matching:
            return matching[0]

    # 3. Fallback: if user has only 1 active project, default to it
    result = await db.execute(
        select(Project.id).where(and_(Project.user_id == user_id, Project.status == "active"))
    )
    active_projects = result.scalars().all()
    if len(active_projects) == 1:
        return active_projects[0]

    return None


async def _resolve_task(
    identifier: str,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> Task | None:
    """
    Resolve a Task from a UUID string or task title.
    Never throws an exception on bad input.
    """
    if not identifier:
        return None

    # 1. Try parsing as direct UUID
    try:
        parsed_uuid = uuid.UUID(identifier.strip())
        result = await db.execute(
            select(Task).where(and_(Task.id == parsed_uuid, Task.user_id == user_id))
        )
        task = result.scalar_one_or_none()
        if task:
            return task
    except (ValueError, AttributeError):
        pass

    # 2. Try partial title match
    cleaned = identifier.strip()
    result = await db.execute(
        select(Task).where(
            and_(
                Task.user_id == user_id,
                Task.title.ilike(f"%{cleaned}%"),
            )
        ).order_by(Task.created_at.desc())
    )
    matching = result.scalars().all()
    if matching:
        return matching[0]

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Handler Functions — safe async DB operations
# ═══════════════════════════════════════════════════════════════════════════════


async def handle_create_project(
    args: dict, user_id: uuid.UUID, db: AsyncSession
) -> dict[str, Any]:
    """Create a project and its profile in one atomic operation."""
    validated = CreateProjectArgs(**args)

    project = Project(
        user_id=user_id,
        name=validated.name,
        client_name=validated.client_name,
    )
    db.add(project)
    await db.flush()

    # Create associated profile
    profile = ProjectProfile(
        project_id=project.id,
        tech_stack=validated.tech_stack or [],
        primary_goal=validated.primary_goal,
        client_quirks=validated.client_quirks,
    )
    db.add(profile)
    await db.commit()

    return {
        "status": "created",
        "project_id": str(project.id),
        "name": validated.name,
        "client_name": validated.client_name,
        "message": f"Project '{validated.name}' created successfully (ID: {project.id}).",
    }


async def handle_update_project_profile(
    args: dict, user_id: uuid.UUID, db: AsyncSession
) -> dict[str, Any]:
    """Update a project's profile fields."""
    validated = UpdateProjectProfileArgs(**args)
    project_id = await _resolve_project_id(validated.project_id, user_id, db)

    if not project_id:
        return {
            "status": "error",
            "message": f"Could not find project '{validated.project_id}'. Use list_projects to see active projects.",
        }

    result = await db.execute(
        select(ProjectProfile).where(ProjectProfile.project_id == project_id)
    )
    profile = result.scalar_one_or_none()

    if not profile:
        profile = ProjectProfile(
            project_id=project_id,
            tech_stack=validated.tech_stack or [],
            repo_url=validated.repo_url,
            deployment_target=validated.deployment_target,
            client_quirks=validated.client_quirks,
            primary_goal=validated.primary_goal,
        )
        db.add(profile)
    else:
        if validated.tech_stack is not None:
            profile.tech_stack = validated.tech_stack
        if validated.repo_url is not None:
            profile.repo_url = validated.repo_url
        if validated.deployment_target is not None:
            profile.deployment_target = validated.deployment_target
        if validated.client_quirks is not None:
            profile.client_quirks = validated.client_quirks
        if validated.primary_goal is not None:
            profile.primary_goal = validated.primary_goal

    await db.commit()

    return {
        "status": "updated",
        "project_id": str(project_id),
        "message": "Project profile updated.",
    }


async def handle_create_task(
    args: dict, user_id: uuid.UUID, db: AsyncSession
) -> dict[str, Any]:
    """Create a task under a project."""
    validated = CreateTaskArgs(**args)
    project_id = await _resolve_project_id(validated.project_id, user_id, db)

    if not project_id:
        return {
            "status": "error",
            "message": f"Could not find project '{validated.project_id}'. Use list_projects to check project IDs.",
        }

    due_at = None
    if validated.due_at:
        try:
            due_at = datetime.fromisoformat(validated.due_at.replace("Z", "+00:00"))
        except ValueError:
            return {"status": "error", "message": f"Invalid date format: {validated.due_at}"}

    task = Task(
        project_id=project_id,
        user_id=user_id,
        title=validated.title,
        priority=validated.priority,
        due_at=due_at,
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    return {
        "status": "created",
        "task_id": str(task.id),
        "title": validated.title,
        "priority": validated.priority,
        "due_at": validated.due_at,
        "message": f"Task '{validated.title}' created (ID: {task.id}).",
    }


async def handle_update_task_status(
    args: dict, user_id: uuid.UUID, db: AsyncSession
) -> dict[str, Any]:
    """Update a task's status and optional blocker reason."""
    validated = UpdateTaskStatusArgs(**args)
    task = await _resolve_task(validated.task_id, user_id, db)

    if not task:
        return {
            "status": "error",
            "message": f"Task '{validated.task_id}' not found. Use list_tasks to see existing task IDs.",
        }

    task.status = validated.status
    if validated.status == "blocked" and validated.blocker_reason:
        task.blocker_reason = validated.blocker_reason
    elif validated.status != "blocked":
        task.blocker_reason = None

    if validated.status == "done":
        task.completed_at = datetime.now(timezone.utc)

    await db.commit()

    return {
        "status": "updated",
        "task_id": str(task.id),
        "new_status": validated.status,
        "message": f"Task '{task.title}' is now {validated.status}.",
    }


async def handle_list_tasks(
    args: dict, user_id: uuid.UUID, db: AsyncSession
) -> dict[str, Any]:
    """List tasks with optional filters."""
    validated = ListTasksArgs(**args)
    conditions = [Task.user_id == user_id]

    if validated.project_id:
        project_id = await _resolve_project_id(validated.project_id, user_id, db)
        if project_id:
            conditions.append(Task.project_id == project_id)
        else:
            return {
                "status": "error",
                "message": f"Could not find project matching '{validated.project_id}'.",
                "tasks": [],
            }

    if validated.status:
        conditions.append(Task.status == validated.status)

    now = datetime.now(timezone.utc)
    if validated.timeframe == "today":
        end_of_day = now.replace(hour=23, minute=59, second=59)
        conditions.append(Task.due_at <= end_of_day)
        conditions.append(Task.status != "done")
    elif validated.timeframe == "this_week":
        end_of_week = now + timedelta(days=(6 - now.weekday()))
        end_of_week = end_of_week.replace(hour=23, minute=59, second=59)
        conditions.append(Task.due_at <= end_of_week)
        conditions.append(Task.status != "done")
    elif validated.timeframe == "overdue":
        conditions.append(Task.due_at < now)
        conditions.append(Task.status != "done")

    result = await db.execute(
        select(Task).where(and_(*conditions)).order_by(Task.due_at.asc().nullslast())
    )
    tasks = result.scalars().all()

    task_list = [
        {
            "task_id": str(t.id),
            "project_id": str(t.project_id),
            "title": t.title,
            "status": t.status,
            "priority": t.priority,
            "due_at": t.due_at.isoformat() if t.due_at else None,
            "blocker_reason": t.blocker_reason,
        }
        for t in tasks
    ]

    return {
        "status": "ok",
        "count": len(task_list),
        "tasks": task_list,
    }


async def handle_list_projects(
    args: dict, user_id: uuid.UUID, db: AsyncSession
) -> dict[str, Any]:
    """List the user's projects with optional status filter."""
    validated = ListProjectsArgs(**args)
    conditions = [Project.user_id == user_id]

    if validated.status:
        conditions.append(Project.status == validated.status)

    result = await db.execute(
        select(Project).where(and_(*conditions)).order_by(Project.created_at.desc())
    )
    projects = result.scalars().all()

    project_list = []
    for p in projects:
        entry = {
            "project_id": str(p.id),
            "name": p.name,
            "client_name": p.client_name,
            "status": p.status,
            "created_at": p.created_at.isoformat(),
        }
        if p.profile:
            entry["tech_stack"] = p.profile.tech_stack
            entry["primary_goal"] = p.profile.primary_goal
        project_list.append(entry)

    return {
        "status": "ok",
        "count": len(project_list),
        "projects": project_list,
    }


async def handle_query_project_knowledge(
    args: dict, user_id: uuid.UUID, db: AsyncSession
) -> dict[str, Any]:
    """Semantic search over project knowledge using pgvector cosine similarity."""
    validated = QueryProjectKnowledgeArgs(**args)
    project_id = await _resolve_project_id(validated.project_id, user_id, db)

    if not project_id:
        return {
            "status": "error",
            "message": f"Could not find project '{validated.project_id}'.",
            "results": [],
        }

    # Generate query embedding with retrieval_query task type
    query_embedding = await generate_embedding(
        validated.search_query, task_type="RETRIEVAL_QUERY"
    )

    # Cosine similarity search via pgvector <=> operator
    result = await db.execute(
        select(ProjectKnowledge)
        .where(ProjectKnowledge.project_id == project_id)
        .order_by(ProjectKnowledge.embedding.cosine_distance(query_embedding))
        .limit(3)
    )
    chunks = result.scalars().all()

    knowledge_list = [
        {
            "title": k.title,
            "category": k.category,
            "content": k.content[:500],  # Truncate for context window efficiency
        }
        for k in chunks
    ]

    return {
        "status": "ok",
        "count": len(knowledge_list),
        "results": knowledge_list,
    }


async def handle_set_reminder(
    args: dict, user_id: uuid.UUID, db: AsyncSession
) -> dict[str, Any]:
    """Schedule a reminder and register its in-memory timer."""
    validated = SetReminderArgs(**args)
    try:
        remind_at = datetime.fromisoformat(validated.remind_at.replace("Z", "+00:00"))
        if remind_at.tzinfo is None:
            remind_at = remind_at.replace(tzinfo=timezone.utc)
    except Exception:
        return {
            "status": "error",
            "message": f"Invalid datetime format: {validated.remind_at}. Please provide a valid ISO 8601 string.",
        }

    # Fetch user phone number for scheduling delivery
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return {"status": "error", "message": "User not found."}

    reminder = Reminder(
        user_id=user_id,
        message=validated.message,
        remind_at=remind_at,
        status="pending",
    )
    db.add(reminder)
    await db.commit()
    await db.refresh(reminder)

    # Register in-memory timer
    schedule_reminder_timer(
        reminder_id=reminder.id,
        user_id=user_id,
        phone_number=user.phone_number,
        message=validated.message,
        remind_at=remind_at,
    )

    return {
        "status": "scheduled",
        "reminder_id": str(reminder.id),
        "message": validated.message,
        "remind_at": remind_at.isoformat(),
        "confirmation": f"Reminder set for {remind_at.isoformat()}: '{validated.message}'",
    }


async def handle_set_preferred_name(
    args: dict, user_id: uuid.UUID, db: AsyncSession
) -> dict[str, Any]:
    """Update user's preferred name in database."""
    validated = SetPreferredNameArgs(**args)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return {"status": "error", "message": "User not found."}

    user.preferred_name = validated.preferred_name.strip()
    await db.commit()

    return {
        "status": "updated",
        "preferred_name": user.preferred_name,
        "message": f"Preferred name updated to '{user.preferred_name}'.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool Router — maps tool names to handler functions
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "create_project": handle_create_project,
    "update_project_profile": handle_update_project_profile,
    "create_task": handle_create_task,
    "update_task_status": handle_update_task_status,
    "list_tasks": handle_list_tasks,
    "list_projects": handle_list_projects,
    "query_project_knowledge": handle_query_project_knowledge,
    "set_reminder": handle_set_reminder,
    "set_preferred_name": handle_set_preferred_name,
}


async def execute_tool(
    tool_name: str,
    tool_args: dict,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> str:
    """
    Route a tool call to its handler and return the JSON result.

    This is the single gateway between the LLM and the database.
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return json.dumps({"status": "error", "message": f"Unknown tool: {tool_name}"})

    try:
        result = await handler(tool_args, user_id, db)
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
        return json.dumps({"status": "error", "message": f"Tool execution failed: {str(e)}"})
