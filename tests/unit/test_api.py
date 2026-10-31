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
