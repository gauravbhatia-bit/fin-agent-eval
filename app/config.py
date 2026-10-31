"""Environment-backed application settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["local", "ci", "production"] = "local"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = ""
    gemini_judge_model: str = ""
    gemini_embed_model: str = ""
    llm_mode: Literal["live", "record", "replay"] = "live"
    llm_temperature: float = Field(default=0.3, ge=0, le=2)
    llm_max_output_tokens: int = Field(default=800, gt=0)
    pack: str = "eu_us_etf"
    rate_limit_per_min: int = Field(default=10, gt=0)
    daily_call_cap: int = Field(default=300, gt=0)
    max_input_chars: int = Field(default=1000, gt=0)
    allowed_origins: str = "http://localhost:8501"
    git_sha: str = "dev"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = ""
    backend_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
