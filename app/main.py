"""FastAPI application entry point."""

import json
from importlib.metadata import version
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as chat_router
from app.api.runtime import ChatRuntime, IpRateLimiter, SessionMemory
from app.config import Settings, get_settings
from app.llm.client import PROMPT_VERSION, DailyCallBudget, LLMClient
from app.schemas import HealthResponse, VersionResponse

LATEST_REPORT = Path(__file__).resolve().parents[1] / "reports" / "latest.json"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API and its process-local chat runtime."""
    app_settings = settings or get_settings()
    application = FastAPI(
        title="fin-agent-eval",
        version=version("fin-agent-eval"),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in app_settings.allowed_origins.split(",")
            if origin.strip()
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    llm_client = LLMClient(
        app_settings,
        budget=DailyCallBudget(app_settings.daily_call_cap),
    )
    application.state.runtime = ChatRuntime(
        settings=app_settings,
        llm_client=llm_client,
        session_memory=SessionMemory(),
        rate_limiter=IpRateLimiter(),
    )
    application.include_router(chat_router)

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return lightweight service health without making an LLM call."""
        return HealthResponse(
            status="ok",
            version=app_settings.git_sha,
            pack=app_settings.pack,
            llm_mode=app_settings.llm_mode,
        )

    @application.get("/version", response_model=VersionResponse)
    def get_version() -> VersionResponse:
        """Return the current build and configured model names."""
        return VersionResponse(
            git_sha=app_settings.git_sha,
            app_version=version("fin-agent-eval"),
            prompt_version=PROMPT_VERSION,
            gemini_model=app_settings.gemini_model or None,
            gemini_judge_model=app_settings.gemini_judge_model or None,
            gemini_embed_model=app_settings.gemini_embed_model or None,
        )

    @application.get("/eval/latest")
    def latest_evaluation() -> JSONResponse:
        """Return the locally written latest report summary, if one exists."""
        if not LATEST_REPORT.is_file():
            raise HTTPException(
                status_code=404,
                detail="No evaluation report is available yet.",
            )
        try:
            report = json.loads(LATEST_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(
                status_code=503,
                detail="The latest evaluation report is unavailable.",
            ) from error
        summary = {
            key: report[key]
            for key in (
                "git_sha",
                "prompt_version",
                "models",
                "pack",
                "mode",
                "timestamp",
                "n",
                "k",
                "metrics",
                "audit",
            )
            if key in report
        }
        return JSONResponse(content=summary)

    return application


app = create_app()
