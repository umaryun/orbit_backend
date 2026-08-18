"""
Orbit Backend — FastAPI application entrypoint & lifecycle.

Startup: Initialize database, configure logging
Shutdown: Dispose engine, close HTTP clients
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.webhook import router as webhook_router
from app.config import get_settings
from app.db.session import dispose_engine, init_db
from app.services.whatsapp import close_client

settings = get_settings()

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Startup
    logger.info("🚀 Orbit Backend starting up (env: %s)", settings.environment)
    await init_db()
    logger.info("✅ Database initialized")

    yield

    # Shutdown
    logger.info("🛑 Orbit Backend shutting down")
    await close_client()
    await dispose_engine()
    logger.info("✅ Cleanup complete")


# ── App Factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Orbit — AI PM & Co-Developer",
    description="WhatsApp-based AI Product Manager backend for developers and freelancers.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None,
)

# ── CORS ───────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────

app.include_router(webhook_router)


@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "service": "orbit-backend",
        "version": "1.0.0",
        "environment": settings.environment,
    }


@app.get("/", tags=["system"])
async def root():
    """Root endpoint — basic service info."""
    return {
        "service": "Orbit — AI PM & Co-Developer",
        "version": "1.0.0",
        "docs": "/docs" if settings.environment == "development" else "disabled",
    }
