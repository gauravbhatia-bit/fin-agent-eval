"""FastAPI application entry point."""

import json
from importlib.metadata import version
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.fees import router as fees_router
from app.api.routes import router as chat_router
from app.api.runtime import ChatRuntime, IpRateLimiter, SessionMemory
from app.config import Settings, get_settings
from app.llm.client import PROMPT_VERSION, DailyCallBudget, LLMClient
from app.schemas import HealthResponse, VersionResponse

LATEST_REPORT = Path(__file__).resolve().parents[1] / "reports" / "latest.json"
SAMPLE_REPORT = Path(__file__).resolve().parents[1] / "reports" / "sample_latest.json"
BASELINE_REPORT = Path(__file__).resolve().parents[1] / "evals" / "baseline.json"


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
    application.include_router(fees_router)

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
        """Return a dashboard-safe report summary or the clearly marked sample."""
        report_path = LATEST_REPORT if LATEST_REPORT.is_file() else SAMPLE_REPORT
        if not report_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="No evaluation report is available yet.",
            )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(
                status_code=503,
                detail="The latest evaluation report is unavailable.",
            ) from error
        if (
            not isinstance(report, dict)
            or not isinstance(baseline, dict)
            or not isinstance(report.get("metrics"), dict)
            or not isinstance(report.get("audit"), dict)
            or not isinstance(report.get("cases"), list)
            or not isinstance(baseline.get("metrics"), dict)
            or not isinstance(baseline.get("lower_is_better", []), list)
            or not isinstance(baseline.get("tolerance_pct_points"), (int, float))
            or (
                "previous_metrics" in report
                and not isinstance(report["previous_metrics"], dict)
            )
        ):
            raise HTTPException(
                status_code=503,
                detail="The latest evaluation report is unavailable.",
            )
        report_cases = report["cases"]
        for row in report_cases:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("case"), dict)
                or not isinstance(row.get("runs"), list)
                or any(not isinstance(run, dict) for run in row["runs"])
            ):
                raise HTTPException(
                    status_code=503,
                    detail="The latest evaluation report is unavailable.",
                )
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
                "previous_metrics",
            )
            if key in report
        }
        summary["cases"] = [
            {
                "id": row["case"].get("id", "unknown"),
                "category": row["case"].get("category", "unknown"),
                "question": row["case"].get("question", ""),
                "expected_decision": row["case"].get("expected_decision", "no_answer"),
                "pass": bool(row.get("pass", False)),
                "actual_decisions": [
                    run["decision"]
                    for run in row["runs"]
                    if isinstance(run.get("decision"), str)
                ],
            }
            for row in report_cases
        ]
        summary["baseline"] = {
            key: baseline[key]
            for key in (
                "placeholder",
                "tolerance_pct_points",
                "metrics",
                "lower_is_better",
            )
            if key in baseline
        }
        summary["is_sample"] = report_path == SAMPLE_REPORT or report.get(
            "is_sample", False
        )
        if "previous_metrics" in report:
            summary["previous_metrics"] = report["previous_metrics"]
        return JSONResponse(content=summary)

    return application


app = create_app()
