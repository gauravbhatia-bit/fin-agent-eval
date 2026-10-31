from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_service_status_and_build_sha() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "dev",
        "pack": "eu_us_etf",
        "llm_mode": "live",
    }


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

    response = client.get("/eval/latest")

    assert response.status_code == 404
    assert response.json()["detail"] == "No evaluation report is available yet."


def test_latest_evaluation_returns_only_the_report_summary(
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
                "traces": [{"answer": "omitted"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main_module, "LATEST_REPORT", path)

    response = client.get("/eval/latest")

    assert response.status_code == 200
    assert response.json() == {"git_sha": "abc", "n": 2, "k": 3, "metrics": {}}
