from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, create_app

client = TestClient(app)


def test_health_returns_service_status_and_build_sha() -> None:
    response = TestClient(create_app(Settings(_env_file=None))).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "dev",
        "pack": "eu_us_etf",
        "llm_mode": "live",
        "model_chat_configured": False,
    }


def test_health_reports_model_chat_configuration_without_exposing_key() -> None:
    client_with_key = TestClient(
        create_app(
            Settings(
                _env_file=None,
                gemini_api_key="placeholder",  # pragma: allowlist secret
            )
        )
    )

    response = client_with_key.get("/health")

    assert response.status_code == 200
    assert response.json()["model_chat_configured"] is True
    assert "placeholder" not in response.text


def test_replay_mode_does_not_report_live_model_chat_as_configured() -> None:
    replay_client = TestClient(
        create_app(
            Settings(
                _env_file=None,
                llm_mode="replay",
                gemini_api_key="placeholder",  # pragma: allowlist secret
            )
        )
    )

    assert replay_client.get("/health").json()["model_chat_configured"] is False


def test_version_returns_build_and_model_configuration() -> None:
    response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert payload["git_sha"] == "dev"
    assert payload["app_version"] == "0.1.0"
    assert payload["prompt_version"] == "v1"
    assert payload["gemini_model"] is None
    assert payload["gemini_judge_model"] is None
    assert payload["gemini_embed_model"] is None


def test_latest_evaluation_returns_not_found_without_report(
    monkeypatch,
    tmp_path,
) -> None:
    import app.main as main_module

    monkeypatch.setattr(
        main_module,
        "LATEST_REPORT",
        tmp_path / "missing.json",
    )
    monkeypatch.setattr(
        main_module,
        "SAMPLE_REPORT",
        tmp_path / "missing-sample.json",
    )

    response = client.get("/eval/latest")

    assert response.status_code == 404
    assert response.json()["detail"] == "No evaluation report is available yet."


def test_latest_evaluation_returns_dashboard_data_without_raw_traces(
    monkeypatch,
    tmp_path,
) -> None:
    import json

    import app.main as main_module

    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "git_sha": "abc",
                "n": 2,
                "k": 3,
                "metrics": {},
                "previous_metrics": {"decision_accuracy": {"value": 0.5}},
                "audit": {"status": "not_available"},
                "cases": [
                    {
                        "case": {
                            "id": "case-1",
                            "category": "answerable",
                            "question": "Question?",
                            "expected_decision": "answered",
                        },
                        "pass": False,
                        "runs": [{"decision": "refused"}],
                    }
                ],
                "traces": [{"answer": "omitted"}],
            }
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "placeholder": True,
                "tolerance_pct_points": 3,
                "metrics": {"decision_accuracy": 0.85},
                "lower_is_better": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main_module, "LATEST_REPORT", path)
    monkeypatch.setattr(main_module, "BASELINE_REPORT", baseline_path)

    response = client.get("/eval/latest")

    assert response.status_code == 200
    assert response.json() == {
        "git_sha": "abc",
        "n": 2,
        "k": 3,
        "metrics": {},
        "previous_metrics": {"decision_accuracy": {"value": 0.5}},
        "audit": {"status": "not_available"},
        "cases": [
            {
                "id": "case-1",
                "category": "answerable",
                "question": "Question?",
                "expected_decision": "answered",
                "pass": False,
                "actual_decisions": ["refused"],
            }
        ],
        "baseline": {
            "placeholder": True,
            "tolerance_pct_points": 3,
            "metrics": {"decision_accuracy": 0.85},
            "lower_is_better": [],
        },
        "is_sample": False,
    }


def test_latest_evaluation_uses_sample_report_when_no_run_exists(
    monkeypatch,
    tmp_path,
) -> None:
    import app.main as main_module

    monkeypatch.setattr(main_module, "LATEST_REPORT", tmp_path / "missing.json")
    monkeypatch.setattr(
        main_module,
        "SAMPLE_REPORT",
        Path(__file__).resolve().parents[2] / "reports" / "sample_latest.json",
    )

    response = client.get("/eval/latest")

    assert response.status_code == 200
    assert response.json()["is_sample"] is True
    assert len(response.json()["cases"]) == 4
    assert response.json()["cases"][0]["actual_decisions"] == ["answered", "answered"]
    assert response.json()["baseline"]["placeholder"] is True


def test_latest_evaluation_rejects_malformed_report(
    monkeypatch,
    tmp_path,
) -> None:
    import app.main as main_module

    report_path = tmp_path / "report.json"
    report_path.write_text("[]", encoding="utf-8")
    baseline_path = Path(__file__).resolve().parents[2] / "evals" / "baseline.json"
    monkeypatch.setattr(main_module, "LATEST_REPORT", report_path)
    monkeypatch.setattr(main_module, "BASELINE_REPORT", baseline_path)

    response = client.get("/eval/latest")

    assert response.status_code == 503
    assert response.json()["detail"] == "The latest evaluation report is unavailable."
