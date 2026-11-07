from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server configuration, read from the environment (and .env in dev)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["gemini", "openai", "deepseek", "kimi"] = "gemini"
    gemini_model: str = "gemini-3.1-flash-lite"
    openai_model: str = "gpt-5.6-luna"
    deepseek_model: str = "deepseek-v4-flash"
    kimi_model: str = "kimi-k2.6"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    moonshot_api_key: str = ""
    max_output_tokens: int = 4096
    knowledge_dir: Path = Path(__file__).resolve().parent.parent / "knowledge"
    # Built client assets; when the directory exists the server serves them at /
    # (production image). In dev it doesn't exist and Vite serves the client.
    static_dir: Path = Path(__file__).resolve().parent.parent / "static"
    rate_limit: str = "20 per 10 minutes"
    # Sites allowed to embed the chat in an iframe (widget.js), as a CSP
    # frame-ancestors source list. 'self' covers the embed demo; add the
    # production site: "'self' https://ruudjuffermans.nl".
    frame_ancestors: str = "'self'"
    # Hard daily spend cap (USD); when reached the agent refuses politely
    # until midnight. 0 disables the cap.
    daily_budget_usd: float = 2.0
    # Bearer token for /api/admin/* (approvals and the admin panel's read
    # API). Empty disables the admin endpoints entirely.
    admin_token: str = ""
    # How long visitor questions, answers and feedback text stay in the
    # database; after that only the numbers remain. 0 = never store text.
    text_retention_days: int = 90
    # Optional OpenAI *admin* key (sk-admin-…) for the organisation Costs
    # API: OpenAI has no balance endpoint, so the panel shows billed spend
    # month-to-date instead. Empty = skipped.
    openai_admin_key: str = ""
    # Postgres for the data platform: checkpoints, approval queue, turn log,
    # budget ledger, knowledge vectors. Empty = no database — everything
    # in-memory, nothing survives a restart (tests, quick bare-metal dev).
    database_url: str = ""
    # Embedding provider for semantic search_knowledge — the two chat
    # providers that also expose an embeddings endpoint. Empty disables vector
    # search (term-overlap fallback); requires DATABASE_URL.
    embedding_provider: Literal["", "gemini", "openai"] = ""
    gemini_embedding_model: str = "gemini-embedding-001"
    openai_embedding_model: str = "text-embedding-3-small"

    @property
    def active_model(self) -> str:
        """The model id the configured provider will be called with."""
        return getattr(self, f"{self.llm_provider}_model")


@lru_cache
def get_settings() -> Settings:
    return Settings()
