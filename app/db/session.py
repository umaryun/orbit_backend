"""
Orbit Backend — Async database engine, session factory, and initialization.
"""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.db.models import Base

logger = logging.getLogger(__name__)

settings = get_settings()

def _get_async_db_url(raw_url: str) -> str:
    """Ensure standard postgresql:// or postgres:// URLs use the asyncpg async driver."""
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return raw_url


db_url = _get_async_db_url(settings.database_url)

engine = create_async_engine(
    db_url,
    echo=(settings.environment == "development"),
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async session, auto-closes on exit."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create pgvector extension and all tables. For dev bootstrapping."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized — all tables created.")


async def dispose_engine() -> None:
    """Gracefully close the connection pool."""
    await engine.dispose()
    logger.info("Database engine disposed.")
