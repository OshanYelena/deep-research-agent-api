"""
Application configuration — loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── API Keys ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    EXA_API_KEY: str = ""
    EXA_BASE_URL: str = "https://api.exa.ai"

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL: str = "gpt-4o-mini"

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── CrewAI Limits ─────────────────────────────────────────────────────────
    AGENT_MAX_RPM: int = 150
    AGENT_MAX_ITER: int = 15

    # ── Job Store ─────────────────────────────────────────────────────────────
    # Max jobs kept in memory before oldest are evicted
    MAX_JOBS_IN_MEMORY: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
