import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from ui.client import BackendClient, BackendError
from ui.data import (
    confusion_matrix,
    consistency_by_category,
    failed_cases,
    metric_deltas,
    metric_rows,
)

ROOT = Path(__file__).resolve().parents[2]


def load_dashboard_sample() -> dict[str, Any]:
    report = json.loads((ROOT / "reports" / "sample_latest.json").read_text())
    baseline = json.loads((ROOT / "evals" / "baseline.json").read_text())
    report["baseline"] = baseline
    report["cases"] = [
        {
            "id": row["case"]["id"],
            "category": row["case"]["category"],
            "question": row["case"]["question"],
            "expected_decision": row["case"]["expected_decision"],
            "pass": row["pass"],
            "actual_decisions": [run["decision"] for run in row["runs"]],
        }
        for row in report["cases"]
    ]
    return report


def test_metric_rows_compare_values_using_baseline_direction_and_tolerance() -> None:
    report = load_dashboard_sample()

    rows = {row["metric"]: row for row in metric_rows(report)}

    assert rows["decision_accuracy"]["status"] == "PASS"
    assert rows["over_refusal_rate"]["status"] == "FAIL"
    assert rows["content_score"]["status"] == "N/A"
    assert rows["decision_accuracy"]["n"] == 4
    assert rows["decision_accuracy"]["k"] == 2


def test_confusion_matrix_counts_all_repeated_case_runs() -> None:
    matrix = confusion_matrix(load_dashboard_sample()["cases"])

    assert matrix[0] == {
        "expected": "answered",
        "answered": 3,
        "refused": 1,
        "escalated": 0,
        "no_answer": 0,
    }
    assert matrix[1]["refused"] == 2
    assert matrix[2]["escalated"] == 2


def test_consistency_by_category_and_failed_cases() -> None:
    cases = load_dashboard_sample()["cases"]

    consistency = consistency_by_category(cases)

    assert consistency == [
        {"category": "answerable", "consistency": 1.0, "cases": 1},
        {"category": "escalate", "consistency": 1.0, "cases": 1},
        {"category": "must_refuse", "consistency": 1.0, "cases": 1},
        {"category": "tool_needed", "consistency": 0.5, "cases": 1},
    ]
    assert [case["id"] for case in failed_cases(cases)] == ["sample-tool"]


def test_metric_deltas_skip_unavailable_values() -> None:
    rows = metric_deltas(
        {"decision_accuracy": {"value": 0.875}, "content_score": {"value": None}},
        {"decision_accuracy": {"value": 0.75}, "content_score": {"value": None}},
    )

    assert rows == [
        {
            "metric": "decision_accuracy",
            "previous": 0.75,
            "current": 0.875,
            "delta": pytest.approx(0.125),
        },
        {
            "metric": "content_score",
            "previous": None,
            "current": None,
            "delta": None,
        },
    ]


def test_health_check_retries_retryable_cold_start_errors() -> None:
    class ColdStartingClient(BackendClient):
        attempts = 0

        def health(self) -> dict[str, Any]:
            self.attempts += 1
            if self.attempts == 1:
                raise BackendError("Starting", retryable=True)
            return {"status": "ok"}

    client = ColdStartingClient("http://backend")
    current_time = 0.0
    retries: list[float] = []

    def clock() -> float:
        return current_time

    def advance(seconds: float) -> None:
        nonlocal current_time
        current_time += seconds

    result = client.wait_for_health(
        timeout_seconds=10,
        interval_seconds=2,
        on_retry=retries.append,
        monotonic=clock,
        sleep=advance,
    )

    assert result == {"status": "ok"}
    assert client.attempts == 2
    assert retries == [0]


def test_health_check_does_not_retry_non_retryable_errors() -> None:
    class BrokenClient(BackendClient):
        attempts = 0

        def health(self) -> dict[str, Any]:
            self.attempts += 1
            raise BackendError("Not found")

    client = BrokenClient("http://backend")

    with pytest.raises(BackendError, match="Not found"):
        client.wait_for_health(
            timeout_seconds=10,
            monotonic=lambda: 0,
            sleep=lambda _: None,
        )
    assert client.attempts == 1


def test_chat_uses_long_timeout_and_explains_backend_read_timeout() -> None:
    session = Mock()
    session.post.side_effect = requests.ReadTimeout("model response timed out")
    client = BackendClient(
        "http://backend",
        session=session,
        timeout_seconds=5,
        chat_timeout_seconds=180,
    )

    with pytest.raises(BackendError, match="timed out while waiting for the backend"):
        client.chat("session", "Calculate a fictional demo fee.")

    assert session.post.call_args.kwargs["timeout"] == (5, 180)


def test_chat_reports_connection_error_separately_from_timeout() -> None:
    session = Mock()
    session.post.side_effect = requests.ConnectionError("connection refused")
    client = BackendClient("http://backend", session=session)

    with pytest.raises(BackendError, match="chat service could not be reached"):
        client.chat("session", "Calculate a fictional demo fee.")


def test_backend_url_prefers_environment_over_streamlit_secret(monkeypatch) -> None:
    import ui.app as ui_app

    monkeypatch.setenv("BACKEND_URL", " https://api.example.test ")
    monkeypatch.setattr(ui_app.st, "secrets", {"BACKEND_URL": "https://secret.test"})

    assert ui_app._backend_url() == "https://api.example.test"


def test_backend_url_reads_streamlit_cloud_secret(monkeypatch) -> None:
    import ui.app as ui_app

    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.setattr(ui_app.st, "secrets", {"BACKEND_URL": "https://api.test"})

    assert ui_app._backend_url() == "https://api.test"


def test_backend_url_defaults_locally_when_streamlit_secrets_are_missing(
    monkeypatch,
) -> None:
    import ui.app as ui_app

    class MissingSecrets:
        def __getitem__(self, _key: str) -> str:
            raise FileNotFoundError("No secrets configured.")

    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.setattr(ui_app.st, "secrets", MissingSecrets())

    assert ui_app._backend_url() == "http://127.0.0.1:8000"


def test_fee_calculator_posts_structured_input_without_chat_timeout() -> None:
    session = Mock()
    response = Mock(ok=True)
    response.json.return_value = {"fee": 3.0, "fee_pct": 0.15, "breakdown": []}
    session.post.return_value = response
    client = BackendClient("http://backend", session=session, timeout_seconds=5)

    result = client.calculate_fee("etf", "buy", 2000)

    assert result["fee"] == 3.0
    session.post.assert_called_once_with(
        "http://backend/fees",
        json={"instrument_type": "etf", "side": "buy", "order_value": 2000},
        timeout=5,
    )
