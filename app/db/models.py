"""
Orbit Backend — SQLAlchemy 2.0 async models.

All tables for the 3-tier knowledge architecture:
  • Users & ConversationLogs (Tier 1 — short-term memory)
  • Projects, ProjectProfiles, Tasks (Tier 2 — relational state)
  • ProjectKnowledge with pgvector embedding (Tier 3 — semantic RAG)
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


# ── Base ───────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ── Enums ──────────────────────────────────────────────────────────────────────

TASK_STATUS = Enum(
    "todo", "in_progress", "blocked", "done",
    name="task_status_enum",
    create_constraint=True,
)

TASK_PRIORITY = Enum(
    "low", "medium", "high", "urgent",
    name="task_priority_enum",
    create_constraint=True,
)

KNOWLEDGE_CATEGORY = Enum(
    "architecture", "api_contract", "client_note", "prd",
    "technical_decision", "gotcha", "general",
    name="knowledge_category_enum",
    create_constraint=True,
)

CONVERSATION_ROLE = Enum(
    "user", "assistant", "tool",
    name="conversation_role_enum",
    create_constraint=True,
)

PROJECT_STATUS = Enum(
    "active", "paused", "completed", "archived",
    name="project_status_enum",
    create_constraint=True,
)

REMINDER_STATUS = Enum(
    "pending", "delivered", "cancelled", "failed",
    name="reminder_status_enum",
    create_constraint=True,
)


# ── Users ──────────────────────────────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    phone_number: Mapped[str] = mapped_column(
        String(20), unique=True, nullable=False, index=True
    )
    first_name: Mapped[str | None] = mapped_column(String(100))
    preferred_name: Mapped[str | None] = mapped_column(String(100))
    timezone: Mapped[str | None] = mapped_column(String(50), default="UTC")
    onboarding_state: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationships
    projects: Mapped[list["Project"]] = relationship(back_populates="user", lazy="selectin")
    conversation_logs: Mapped[list["ConversationLog"]] = relationship(
        back_populates="user", lazy="selectin"
    )
    reminders: Mapped[list["Reminder"]] = relationship(
        back_populates="user", lazy="selectin"
    )


# ── Projects ───────────────────────────────────────────────────────────────────


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    client_name: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(
        PROJECT_STATUS, default="active", server_default="active"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="projects")
    profile: Mapped["ProjectProfile | None"] = relationship(
        back_populates="project", uselist=False, lazy="selectin"
    )
    tasks: Mapped[list["Task"]] = relationship(back_populates="project", lazy="selectin")
    knowledge: Mapped[list["ProjectKnowledge"]] = relationship(
        back_populates="project", lazy="selectin"
    )


# ── Project Profiles ──────────────────────────────────────────────────────────


class ProjectProfile(Base):
    __tablename__ = "project_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    tech_stack: Mapped[dict | None] = mapped_column(JSONB, default=list)
    repo_url: Mapped[str | None] = mapped_column(String(500))
    deployment_target: Mapped[str | None] = mapped_column(String(200))
    client_quirks: Mapped[str | None] = mapped_column(Text)
    primary_goal: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="profile")


# ── Tasks ──────────────────────────────────────────────────────────────────────


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(
        TASK_STATUS, default="todo", server_default="todo"
    )
    blocker_reason: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[str] = mapped_column(
        TASK_PRIORITY, default="medium", server_default="medium"
    )
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="tasks")

    __table_args__ = (
        Index("ix_tasks_project_status", "project_id", "status"),
        Index("ix_tasks_user_due", "user_id", "due_at"),
    )


# ── Project Knowledge (pgvector) ──────────────────────────────────────────────


class ProjectKnowledge(Base):
    __tablename__ = "project_knowledge"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    category: Mapped[str] = mapped_column(KNOWLEDGE_CATEGORY, nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(768))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="knowledge")

    __table_args__ = (
        Index(
            "ix_knowledge_embedding",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# ── Conversation Logs ─────────────────────────────────────────────────────────


class ConversationLog(Base):
    __tablename__ = "conversation_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(CONVERSATION_ROLE, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversation_logs")

    __table_args__ = (
        Index("ix_convlog_user_created", "user_id", "created_at"),
    )


# ── Reminders ─────────────────────────────────────────────────────────────────


class Reminder(Base):
    __tablename__ = "reminders"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    remind_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(
        REMINDER_STATUS, default="pending", server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="reminders")

    __table_args__ = (
        Index("ix_reminders_status_remind_at", "status", "remind_at"),
        Index("ix_reminders_user_id", "user_id"),
    )

