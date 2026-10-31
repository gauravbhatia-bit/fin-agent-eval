from fastapi.testclient import TestClient

from app.api.runtime import IpRateLimiter, SessionMemory
from app.config import Settings
from app.main import create_app


def build_client(**overrides: object) -> TestClient:
    settings = Settings(_env_file=None, **overrides)
    return TestClient(create_app(settings))


def test_chat_returns_required_schema_for_rule_refusal() -> None:
    client = build_client()

    response = client.post(
        "/chat",
        json={"session_id": "test-session", "message": "Should I buy X?"},
    )

    assert response.status_code == 200
    assert set(response.json()) == {
        "decision",
        "answer",
        "tool_calls",
        "citations",
        "disclaimer",
        "trace_id",
    }
    assert response.json()["decision"] == "refused"
    assert response.json()["tool_calls"] == []
    assert response.json()["trace_id"]


def test_chat_rejects_empty_and_over_configured_length() -> None:
    client = build_client(max_input_chars=10)

    empty = client.post("/chat", json={"session_id": "s", "message": "  "})
    too_long = client.post(
        "/chat",
        json={"session_id": "s", "message": "x" * 11},
    )

    assert empty.status_code == 422
    assert too_long.status_code == 422


def test_chat_rate_limits_requests_per_client_ip() -> None:
    client = build_client(rate_limit_per_min=1)
    payload = {"session_id": "s", "message": "Should I buy X?"}

    assert client.post("/chat", json=payload).status_code == 200
    limited = client.post("/chat", json=payload)

    assert limited.status_code == 429
    assert "wait a minute" in limited.json()["detail"]


def test_chat_surfaces_provider_status_without_provider_error_body(
    monkeypatch,
) -> None:
    import app.api.routes as routes

    class ProviderError(Exception):
        code = 400
        status = "INVALID_ARGUMENT"

    def fail_graph(_state: object) -> None:
        raise ProviderError("provider response details stay private")

    monkeypatch.setattr(routes.agent_graph, "invoke", fail_graph)
    client = build_client()

    response = client.post(
        "/chat",
        json={"session_id": "test-session", "message": "What is an ETF?"},
    )

    assert response.status_code == 502
    assert "HTTP 400 (INVALID_ARGUMENT)" in response.json()["detail"]
    assert "provider response details stay private" not in response.json()["detail"]


def test_chat_explains_quota_failure_and_local_fee_calculator(
    monkeypatch,
) -> None:
    import app.api.routes as routes

    class ProviderError(Exception):
        code = 429
        status = "RESOURCE_EXHAUSTED"

    def fail_graph(_state: object) -> None:
        raise ProviderError("provider response details stay private")

    monkeypatch.setattr(routes.agent_graph, "invoke", fail_graph)
    client = build_client()

    response = client.post(
        "/chat",
        json={"session_id": "test-session", "message": "What is an ETF?"},
    )

    assert response.status_code == 502
    assert "AI chat is temporarily unavailable" in response.json()["detail"]
    assert "does not use Gemini" in response.json()["detail"]
    assert "provider response details stay private" not in response.json()["detail"]


def test_fee_endpoint_calculates_without_invoking_chat(monkeypatch) -> None:
    import app.agent.graph as graph

    def fail_graph(_state: object) -> None:
        raise AssertionError("fee endpoint must not invoke the LLM")

    monkeypatch.setattr(graph.agent_graph, "invoke", fail_graph)
    client = build_client()

    response = client.post(
        "/fees",
        json={"instrument_type": "etf", "side": "buy", "order_value": 2000},
    )

    assert response.status_code == 200
    assert response.json()["fee"] == 3.0
    assert response.json()["fee_pct"] == 0.15
    assert [part["amount"] for part in response.json()["breakdown"]] == [1, 2, 0]


def test_fee_endpoint_rejects_invalid_input() -> None:
    client = build_client()

    response = client.post(
        "/fees",
        json={"instrument_type": "crypto", "side": "buy", "order_value": 2000},
    )

    assert response.status_code == 422


def test_session_memory_keeps_six_turns_and_caps_size() -> None:
    memory = SessionMemory(max_turns=6, max_chars=500)
    for turn in range(7):
        memory.append("s", f"question {turn}", f"answer {turn}")

    history = memory.history("s")

    assert len(history) == 12
    assert history[0]["content"] == "question 1"
    assert history[-1]["content"] == "answer 6"


def test_session_memory_expires_idle_sessions(monkeypatch) -> None:
    import app.api.runtime as runtime_module

    now = [0.0]
    monkeypatch.setattr(runtime_module, "monotonic", lambda: now[0])
    memory = SessionMemory(ttl_seconds=30)
    memory.append("s", "question", "answer")

    now[0] = 31.0
    assert memory.history("s") == []


def test_session_memory_evicts_oldest_turns_for_size_cap() -> None:
    memory = SessionMemory(max_turns=6, max_chars=25)
    memory.append("s", "1234567890", "abcdefghij")
    memory.append("s", "klmnopqrst", "uvwxyz")

    history = memory.history("s")

    assert len(history) == 2
    assert history[0]["content"] == "klmnopqrst"


def test_session_memory_caps_number_of_sessions() -> None:
    memory = SessionMemory(max_sessions=1)
    memory.append("first", "question", "answer")
    memory.append("second", "question", "answer")

    assert memory.history("first") == []
    assert memory.history("second")


def test_rate_limiter_reclaims_expired_ip_slots(monkeypatch) -> None:
    import app.api.runtime as runtime_module

    now = [0.0]
    monkeypatch.setattr(runtime_module, "monotonic", lambda: now[0])
    limiter = IpRateLimiter(max_tracked_ips=1)

    assert limiter.allow("first", 1)
    assert not limiter.allow("second", 1)
    now[0] = 61.0
    assert limiter.allow("second", 1)
