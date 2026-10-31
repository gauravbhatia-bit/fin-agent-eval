"""Pydantic response models for the scaffold API."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response consumed by local and hosted deployment checks."""

    status: str
    version: str
    pack: str
    llm_mode: str
    model_chat_configured: bool


class VersionResponse(BaseModel):
    """Build and configured-model information for diagnostics."""

    git_sha: str
    app_version: str
    prompt_version: str | None
    gemini_model: str | None
    gemini_judge_model: str | None
    gemini_embed_model: str | None
