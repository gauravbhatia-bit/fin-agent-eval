"""FastAPI application entry point."""

from importlib.metadata import version
from typing import Annotated

from fastapi import Depends, FastAPI

from app.config import Settings, get_settings
from app.schemas import HealthResponse, VersionResponse

app = FastAPI(title="fin-agent-eval", version=version("fin-agent-eval"))


@app.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Return lightweight service health without making an LLM call."""
    return HealthResponse(
        status="ok",
        version=settings.git_sha,
        pack=settings.pack,
        llm_mode=settings.llm_mode,
    )


@app.get("/version", response_model=VersionResponse)
def get_version(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VersionResponse:
    """Return the current build and configured model names."""
    return VersionResponse(
        git_sha=settings.git_sha,
        app_version=version("fin-agent-eval"),
        prompt_version=None,
        gemini_model=settings.gemini_model or None,
        gemini_judge_model=settings.gemini_judge_model or None,
        gemini_embed_model=settings.gemini_embed_model or None,
    )
