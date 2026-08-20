"""
Orbit Backend — Application configuration via Pydantic Settings.

All environment variables are validated at startup. Missing required keys
will crash the app immediately with a clear error message.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Supabase Postgres ──────────────────────────────────────────────
    database_url: str

    # ── WhatsApp Cloud API ─────────────────────────────────────────────
    whatsapp_token: str
    whatsapp_phone_number_id: str
    whatsapp_verify_token: str

    # ── Gemini (LLM + Embeddings) ──────────────────────────────────────
    gemini_api_key: str
    gemini_llm_model: str = "gemini-3-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    # ── Groq (Whisper) ─────────────────────────────────────────────────
    groq_api_key: str

    # ── App ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    environment: str = "development"

    # ── Tuning Knobs ───────────────────────────────────────────────────
    short_term_memory_limit: int = 15
    semantic_search_top_k: int = 3
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    max_tool_iterations: int = 5
    typing_delay_min_ms: int = 400
    typing_delay_max_ms: int = 800

    @property
    def whatsapp_api_url(self) -> str:
        return f"https://graph.facebook.com/v20.0/{self.whatsapp_phone_number_id}"


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — parsed once, reused everywhere."""
    return Settings()
