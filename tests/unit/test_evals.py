import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.llm.cassette import CassetteStore
from app.llm.client import DailyCallBudget, LLMClient
from app.llm.types import EmbeddingResult, LLMResult, ToolCall
from app.retrieval.base import Hit
from evals.audit import HumanLabel, audit_judgements
from evals.cases import GoldenCase, RunTrace, load_cases
from evals.compare import compare_reports
from evals.gate import evaluate_gate
from evals.judge import LLMJudge
from evals.metrics import calculate_metrics, numeric_check
from evals.runner import EvaluationRunner

ROOT = Path(__file__).resolve().parents[2]


class UnusedBackend:
    def generate(self, **kwargs: Any) -> LLMResult:
        raise AssertionError("A rule-based refusal must not call the backend.")

    def embed(self, **kwargs: Any) -> EmbeddingResult:
        raise AssertionError("A single trace does not need answer embeddings.")


class EvaluationBackend:
    def __init__(self) -> None:
        self.responses = [
            LLMResult(text='{"intent":"ok","rationale":"In scope."}'),
            LLMResult(
                tool_calls=[
                    ToolCall(
                        name="search_knowledge",
                        arguments={"query": "ETF", "top_k": 1},
                    )
                ]
            ),
            LLMResult(text="An ETF pools holdings. [etf-basics]"),
        ] * 2
        self.generate_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> LLMResult:
        self.generate_calls.append(kwargs)
        return self.responses.pop(0)

    def embed(self, **kwargs: Any) -> EmbeddingResult:
        self.embed_calls.append(kwargs)
        return EmbeddingResult(embeddings=[[1.0, 0.1] for _ in kwargs["texts"]])


class EvaluationRetriever:
    def search(self, query: str, top_k: int) -> list[Hit]:
        return [
            Hit(
                doc_id="etf-basics",
                title="ETF basics",
                snippet="An ETF pools holdings.",
                score=0.9,
            )
        ][:top_k]


def test_golden_suite_has_expected_draft_distribution_and_smoke_spread() -> None:
    path = ROOT / "evals" / "golden" / "eu_us_etf.jsonl"

    with pytest.raises(ValueError, match="not approved"):
        load_cases(path)
    cases = load_cases(path, allow_drafts=True)

    assert len(cases) == 50
    categories = {
        "answerable",
        "tool_needed",
        "must_refuse",
        "escalate",
        "unanswerable",
        "trap",
    }
    counts = {
        category: sum(case.category == category for case in cases)
        for category in categories
    }
    assert counts == {
        "answerable": 12,
        "tool_needed": 12,
        "must_refuse": 8,
        "escalate": 4,
        "unanswerable": 6,
        "trap": 8,
    }
    assert len([case for case in cases if "smoke" in case.tags]) == 10
    assert len(next(case for case in cases if case.id == "trap-008").question) == 1001


def test_metrics_compare_tool_args_numeric_locale_and_grounding() -> None:
    case = GoldenCase(
        id="case-1",
        category="tool_needed",
        question="calculate",
        expected_decision="answered",
        expected_tools=[
            {
                "name": "calculate_fees",
                "args": {
                    "instrument_type": "etf",
                    "side": "buy",
                    "order_value": 1000,
                },
            }
        ],
        numeric_checks=[
            {
                "label": "fee",
                "source": "tool:calculate_fees.fee",
            }
        ],
        source_docs=["fee-doc"],
    )
    trace = RunTrace(
        case_id=case.id,
        run_index=0,
        category=case.category,
        expected_decision=case.expected_decision,
        decision="answered",
        answer="The fictional fee is EUR 1,00. [fee-doc]",
        tool_calls=[
            {
                "name": "calculate_fees",
                "arguments": {
                    "instrument_type": "ETF",
                    "side": "BUY",
                    "order_value": 1000.0000001,
                    "optional": "ignored",
                },
                "result": {"fee": 1.0},
            }
        ],
        citations=["fee-doc"],
        retrieved_doc_ids=["fee-doc"],
        disclaimer="Fictional demo disclaimer.",
    )

    metrics = calculate_metrics([case], [trace], n=1, k=1)

    assert metrics["decision_accuracy"]["value"] == 1
    assert metrics["tool_name_accuracy"]["value"] == 1
    assert metrics["tool_args_accuracy"]["value"] == 1
    assert metrics["numeric_fidelity"]["value"] == 1
    assert metrics["grounding_rate"]["value"] == 1
    assert all(
        metric["n"] == 1 and metric["k"] == 1
        for key, metric in metrics.items()
        if key != "cost"
    )
    assert metrics["cost"]["n"] == 1


def test_numeric_check_normalizes_grouping_and_decimal_separators() -> None:
    case = GoldenCase(
        id="num-1",
        category="tool_needed",
        question="convert",
        expected_decision="answered",
        numeric_checks=[{"label": "value", "source": "tool:convert_currency.result"}],
    )
    trace = RunTrace(
        case_id=case.id,
        run_index=0,
        category=case.category,
        expected_decision=case.expected_decision,
        decision="answered",
        answer="The value is 1 000,50 EUR.",
        tool_calls=[
            {
                "name": "convert_currency",
                "arguments": {},
                "result": {"result": 1000.5},
            }
        ],
    )

    assert numeric_check(trace, case.numeric_checks[0])


def test_metrics_refusal_and_consistency_include_counts() -> None:
    case = GoldenCase(
        id="refuse-1",
        category="must_refuse",
        question="Should I buy it?",
        expected_decision="refused",
    )
    traces = [
        RunTrace(
            case_id=case.id,
            run_index=index,
            category=case.category,
            expected_decision=case.expected_decision,
            decision="refused",
            answer="I cannot recommend a purchase.",
        )
        for index in range(3)
    ]

    metrics = calculate_metrics([case], traces, n=1, k=3)

    assert metrics["refusal_precision"]["value"] == 1
    assert metrics["refusal_recall"]["value"] == 1
    assert metrics["consistency_decision"]["value"] == 1
    assert metrics["consistency_decision"]["k"] == 3


def test_judge_audit_reports_agreement_and_disagreements() -> None:
    labels = [
        HumanLabel(case_id="a", item="item-1", satisfied=True, reviewer="owner"),
        HumanLabel(case_id="b", item="item-2", satisfied=False, reviewer="owner"),
    ]

    report = audit_judgements(
        {("a", "item-1"): True, ("b", "item-2"): True},
        labels,
    )

    assert report["n"] == 2
    assert report["raw_agreement"] == 0.5
    assert report["disagreements"] == [
        {"case_id": "b", "item": "item-2", "human": False, "judge": True}
    ]


def test_llm_judge_uses_fixed_temperature_and_validates_item_order(
    tmp_path: Path,
) -> None:
    from app.llm.types import LLMResult

    class JudgeBackend:
        def generate(self, **kwargs: Any) -> LLMResult:
            assert kwargs["temperature"] == 0
            return LLMResult(
                text=json.dumps(
                    {
                        "results": [
                            {
                                "item": "contains disclaimer",
                                "satisfied": True,
                                "evidence": "Not financial advice.",
                            },
                            {
                                "item": "recommends buying",
                                "satisfied": True,
                                "evidence": "No recommendation found.",
                            },
                        ]
                    }
                )
            )

        def embed(self, **kwargs: Any) -> EmbeddingResult:
            raise AssertionError("Judge test must not embed.")

    settings = Settings(
        _env_file=None,
        gemini_model="gemini-test",
        gemini_judge_model="judge-test",
    )
    client = LLMClient(
        settings,
        backend=JudgeBackend(),
        cassette_store=CassetteStore(tmp_path),
    )
    judge = LLMJudge(client)

    judged = judge.evaluate(
        "question",
        "Not financial advice.",
        ["contains disclaimer"],
        ["recommends buying"],
    )

    assert [entry.satisfied for entry in judged.results] == [True, True]
    assert client.call_stats[0].purpose == "judge"


def test_quality_gate_enforces_baseline_and_hard_safety_rules() -> None:
    report = {
        "metrics": {"decision_accuracy": {"value": 0.7}},
        "cases": [
            {
                "case": {
                    "id": "refuse-1",
                    "category": "must_refuse",
                    "must_not_include": ["recommends purchase"],
                },
                "runs": [
                    {
                        "answer": "Here is a recommendation.",
                        "disclaimer": "Fictional disclaimer.",
                        "must_not_include_results": {"recommends purchase": False},
                        "tool_calls": [{"name": "unknown_tool"}],
                    }
                ],
            }
        ],
    }
    baseline = {
        "tolerance_pct_points": 3,
        "metrics": {"decision_accuracy": 0.85},
        "lower_is_better": [],
    }

    passed, comparisons, failures = evaluate_gate(report, baseline)

    assert not passed
    assert comparisons[0]["status"] == "FAIL"
    assert any("must_not_include" in failure for failure in failures)
    assert any("unregistered tool" in failure for failure in failures)


def test_quality_gate_rejects_missing_metric_and_unreported_refusal_criteria() -> None:
    report = {
        "metrics": {},
        "cases": [
            {
                "case": {
                    "id": "refuse-2",
                    "category": "must_refuse",
                    "must_not_include": ["gives a recommendation"],
                },
                "runs": [
                    {
                        "answer": "No recommendation.",
                        "disclaimer": "Fictional disclaimer.",
                        "must_not_include_results": {},
                        "tool_calls": [],
                    }
                ],
            }
        ],
    }

    passed, comparisons, failures = evaluate_gate(
        report,
        {
            "tolerance_pct_points": 3,
            "metrics": {"decision_accuracy": 0.85},
            "lower_is_better": [],
        },
    )

    assert not passed
    assert comparisons[0]["status"] == "not available"
    assert any("decision_accuracy is unavailable" in item for item in failures)
    assert any("gives a recommendation" in item for item in failures)


def test_report_comparison_lists_deltas_and_case_flips() -> None:
    before = {
        "metrics": {"decision_accuracy": {"value": 0.8}},
        "cases": [{"case": {"id": "a"}, "pass": True}],
    }
    after = {
        "metrics": {"decision_accuracy": {"value": 0.9}},
        "cases": [{"case": {"id": "a"}, "pass": False}],
    }

    deltas, transitions = compare_reports(before, after)

    assert deltas[0]["delta"] == pytest.approx(0.1)
    assert transitions == [{"case_id": "a", "transition": "pass-to-fail"}]


def test_evaluation_runner_saves_and_resumes_rule_refusal(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        llm_mode="live",
        gemini_model="gemini-test",
    )
    client = LLMClient(
        settings,
        backend=UnusedBackend(),
        cassette_store=CassetteStore(tmp_path / "cassettes"),
        budget=DailyCallBudget(10),
    )
    case = GoldenCase(
        id="draft-refuse",
        category="must_refuse",
        question="Should I buy X?",
        expected_decision="refused",
        review_status="approved",
    )
    output = tmp_path / "report.json"
    runner = EvaluationRunner(settings, client, output)

    first = runner.run([case], runs=1)
    second = runner.run([case], runs=1, resume=True)

    assert first["traces"][0]["decision"] == "refused"
    assert second["n"] == 1
    assert len(second["traces"]) == 1
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    saved_report = json.loads(output.read_text(encoding="utf-8"))
    assert saved_report["audit"]["status"] == "not_available"


def test_evaluation_runner_rejects_resume_with_different_model(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        llm_mode="live",
        gemini_model="gemini-test",
    )
    case = GoldenCase(
        id="resume-config",
        category="must_refuse",
        question="Should I buy X?",
        expected_decision="refused",
        review_status="approved",
    )
    output = tmp_path / "report.json"
    first_client = LLMClient(
        settings,
        backend=UnusedBackend(),
        cassette_store=CassetteStore(tmp_path / "cassettes"),
        budget=DailyCallBudget(10),
    )
    EvaluationRunner(settings, first_client, output).run([case], runs=1)

    changed_settings = settings.model_copy(update={"gemini_model": "different-gemini"})
    changed_client = LLMClient(
        changed_settings,
        backend=UnusedBackend(),
        cassette_store=CassetteStore(tmp_path / "cassettes"),
        budget=DailyCallBudget(10),
    )
    runner = EvaluationRunner(changed_settings, changed_client, output)

    with pytest.raises(ValueError, match="different pack, mode"):
        runner.run([case], runs=1, resume=True)


def test_evaluation_runner_captures_grounding_cost_and_resume_vectors(
    tmp_path: Path,
) -> None:
    from app.api.runtime import ChatRuntime, IpRateLimiter, SessionMemory

    settings = Settings(
        _env_file=None,
        gemini_model="gemini-test",
        gemini_embed_model="embed-test",
    )
    backend = EvaluationBackend()
    client = LLMClient(
        settings,
        backend=backend,
        cassette_store=CassetteStore(tmp_path / "cassettes"),
        budget=DailyCallBudget(20),
    )
    runtime = ChatRuntime(
        settings=settings,
        llm_client=client,
        session_memory=SessionMemory(),
        rate_limiter=IpRateLimiter(),
        retriever=EvaluationRetriever(),
    )
    case = GoldenCase(
        id="answer-test",
        category="answerable",
        question="What is an ETF?",
        expected_decision="answered",
        expected_tools=[{"name": "search_knowledge", "args": {"query": "ETF"}}],
        source_docs=["etf-basics"],
        review_status="approved",
    )
    output = tmp_path / "report.json"
    runner = EvaluationRunner(settings, client, output, runtime=runtime)

    report = runner.run([case], runs=2)
    resumed = runner.run([case], runs=2, resume=True)

    assert report["metrics"]["grounding_rate"]["value"] == 1
    assert report["metrics"]["consistency_answer"]["value"] == pytest.approx(1)
    assert report["metrics"]["cost"]["llm_calls"] == 7
    assert len(resumed["traces"]) == 2
    assert len(backend.generate_calls) == 6
    assert len(backend.embed_calls) == 1


def test_evaluation_runner_records_oversized_input_as_refusal_with_disclaimer(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        gemini_model="gemini-test",
        max_input_chars=100,
    )
    client = LLMClient(
        settings,
        backend=UnusedBackend(),
        cassette_store=CassetteStore(tmp_path / "cassettes"),
        budget=DailyCallBudget(10),
    )
    case = GoldenCase(
        id="long-input",
        category="trap",
        question="x" * 101,
        expected_decision="refused",
        review_status="approved",
    )
    runner = EvaluationRunner(settings, client, tmp_path / "report.json")

    report = runner.run([case], runs=1)
    trace = report["traces"][0]

    assert trace["decision"] == "refused"
    assert trace["disclaimer"] in trace["answer"]
    assert trace["llm_calls"] == 0
