"""Run golden cases through the real guarded graph with live/record/replay LLMs."""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import Any

from app.agent.graph import agent_graph
from app.agent.nodes import InputTooLong, initialize_state
from app.agent.policy import load_policy
from app.agent.prompts import PROMPT_VERSION
from app.api.runtime import ChatRuntime, IpRateLimiter, SessionMemory
from app.api.schemas import ChatResponse
from app.config import Settings, get_settings
from app.llm.client import DailyCallBudget, LLMClient
from app.llm.gemini import GeminiClient
from app.llm.types import EmbeddingResult, LLMResult
from evals.audit import audit_judgements, judgement_map, load_human_labels
from evals.cases import (
    GoldenCase,
    RunTrace,
    load_cases,
    trace_from_response,
    write_json,
)
from evals.judge import LLMJudge, apply_judgements
from evals.metrics import (
    _tool_args_match,
    _tool_name_match,
    calculate_metrics,
    numeric_check,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = ROOT / "evals" / "golden"
REPORT_ROOT = ROOT / "reports"
ANSWER_EMBEDDING_BATCH_SIZE = 50
logger = logging.getLogger("evals.runner")


class RateLimitedBackend:
    """Space every provider request to remain below configured RPM."""

    def __init__(
        self,
        backend: GeminiClient,
        requests_per_minute: int,
        sleeper: Any = sleep,
    ) -> None:
        self.backend = backend
        self.interval = 60.0 / requests_per_minute
        self.sleeper = sleeper
        self._last_request: float | None = None
        self._lock = Lock()

    def _wait(self) -> None:
        with self._lock:
            now = monotonic()
            if self._last_request is not None:
                delay = self.interval - (now - self._last_request)
                if delay > 0:
                    self.sleeper(delay)
            self._last_request = monotonic()

    def generate(self, **kwargs: Any) -> LLMResult:
        self._wait()
        return self.backend.generate(**kwargs)

    def embed(self, **kwargs: Any) -> EmbeddingResult:
        self._wait()
        return self.backend.embed(**kwargs)


class EvaluationRunner:
    """Execute case/run pairs and atomically persist resumable results."""

    def __init__(
        self,
        settings: Settings,
        client: LLMClient,
        output_path: Path,
        *,
        judge: LLMJudge | None = None,
        runtime: ChatRuntime | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.output_path = output_path
        self.judge = judge
        self._embedding_cost: dict[str, Any] = {}
        self.runtime = runtime or ChatRuntime(
            settings=settings,
            llm_client=client,
            session_memory=SessionMemory(),
            rate_limiter=IpRateLimiter(),
        )

    def run(
        self,
        cases: list[GoldenCase],
        *,
        runs: int,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Run all missing case/run pairs and save a report after each result."""
        if runs < 1:
            raise ValueError("runs must be at least one")
        traces: list[RunTrace] = []
        self._answer_embeddings: dict[tuple[str, int], list[float]] = {}
        self._embedding_cost = {}
        self._previous_metrics: dict[str, Any] = {}
        if resume and self.output_path.exists():
            saved = json.loads(self.output_path.read_text(encoding="utf-8"))
            self._previous_metrics = saved.get("previous_metrics", {})
            expected_metadata = {
                "pack": self.settings.pack,
                "mode": self.settings.llm_mode,
                "prompt_version": PROMPT_VERSION,
                "models": {
                    "agent": self.settings.gemini_model or None,
                    "judge": self.settings.gemini_judge_model or None,
                    "embedding": self.settings.gemini_embed_model or None,
                },
            }
            if any(saved.get(key) != value for key, value in expected_metadata.items()):
                raise ValueError(
                    "Cannot resume a report created with different pack, mode, "
                    "prompt version, or model configuration."
                )
            traces = [RunTrace.model_validate(row) for row in saved.get("traces", [])]
            saved_trace_count = len(traces)
            saved_cases = {
                row.get("case", {}).get("id"): row.get("case", {})
                for row in saved.get("cases", [])
            }
            current_cases = {case.id: case for case in cases}
            traces = [
                trace
                for trace in traces
                if trace.case_id in current_cases
                and trace.run_index < runs
                and self._same_case(
                    current_cases[trace.case_id],
                    saved_cases.get(trace.case_id, {}),
                )
            ]
            if len(traces) != saved_trace_count:
                self._embedding_cost = {}
            self._answer_embeddings = self._deserialize_embeddings(
                saved.get("answer_embeddings", {})
            )
            valid_trace_keys = {(trace.case_id, trace.run_index) for trace in traces}
            self._answer_embeddings = {
                key: value
                for key, value in self._answer_embeddings.items()
                if key in valid_trace_keys
            }
            if len(traces) == saved_trace_count:
                self._embedding_cost = saved.get("answer_embedding_cost", {})
        elif self.output_path.is_file():
            try:
                previous_report = json.loads(
                    self.output_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                logger.warning(
                    "Previous evaluation report is unreadable; delta comparison "
                    "will be unavailable."
                )
            else:
                previous_metrics = previous_report.get("metrics")
                if isinstance(previous_metrics, dict):
                    self._previous_metrics = previous_metrics
        completed = {(trace.case_id, trace.run_index) for trace in traces}

        for case in cases:
            for run_index in range(runs):
                if (case.id, run_index) in completed:
                    continue
                trace = self._run_one(case, run_index)
                if self.judge is not None:
                    before_judge_budget = self.client.budget.calls
                    before_judge_stats = len(self.client.call_stats)
                    judge_started = monotonic()
                    include, prohibited = apply_judgements(
                        case.question,
                        trace.answer,
                        case.must_include,
                        case.must_not_include,
                        self.judge,
                    )
                    judge_stats = self.client.call_stats[before_judge_stats:]
                    trace = trace.model_copy(
                        update={
                            "must_include_results": include,
                            "must_not_include_results": prohibited,
                            "llm_calls": trace.llm_calls
                            + max(
                                self.client.budget.calls - before_judge_budget,
                                0,
                            ),
                            "input_tokens": trace.input_tokens
                            + sum(entry.input_tokens for entry in judge_stats),
                            "output_tokens": trace.output_tokens
                            + sum(entry.output_tokens for entry in judge_stats),
                            "latency_ms": trace.latency_ms
                            + (monotonic() - judge_started) * 1000,
                        }
                    )
                numeric_results = {
                    check.label: numeric_check(trace, check)
                    for check in case.numeric_checks
                    if check.must_appear_in_answer
                }
                trace = trace.model_copy(update={"numeric_results": numeric_results})
                traces.append(trace)
                completed.add((case.id, run_index))
                self._save(cases, traces, runs)
        embeddings = self._embed_answers(traces)
        report = self._report(cases, traces, runs, embeddings)
        self._persist(report)
        return report

    def _run_one(self, case: GoldenCase, run_index: int) -> RunTrace:
        before_budget = self.client.budget.calls
        before_stats = len(self.client.call_stats)
        started = monotonic()
        state = initialize_state(
            message=case.question,
            llm_client=self.client,
            retriever_factory=self.runtime.get_retriever,
            settings=self.settings,
            trace_id=f"eval-{case.id}-{run_index}",
        )
        try:
            result = agent_graph.invoke(state)
        except InputTooLong as error:
            response = ChatResponse(
                decision="refused",
                answer=f"{error}\n\n{load_policy().disclaimer}",
                tool_calls=[],
                citations=[],
                disclaimer=load_policy().disclaimer,
                trace_id=state["trace_id"],
            )
            return trace_from_response(
                case,
                run_index,
                response,
                retrieved_doc_ids=[],
                llm_calls=0,
                input_tokens=0,
                output_tokens=0,
                latency_ms=(monotonic() - started) * 1000,
            )
        call_stats = self.client.call_stats[before_stats:]
        response = ChatResponse(
            decision=result["decision"],
            answer=result["answer"],
            tool_calls=result.get("tool_calls", []),
            citations=result.get("citations", []),
            disclaimer=result["disclaimer"],
            trace_id=result["trace_id"],
        )
        return trace_from_response(
            case,
            run_index,
            response,
            retrieved_doc_ids=result.get("retrieved_doc_ids", []),
            llm_calls=max(self.client.budget.calls - before_budget, 0),
            input_tokens=sum(entry.input_tokens for entry in call_stats),
            output_tokens=sum(entry.output_tokens for entry in call_stats),
            latency_ms=(monotonic() - started) * 1000,
        )

    def _embed_answers(
        self,
        traces: list[RunTrace],
    ) -> dict[tuple[str, int], list[float]] | None:
        eligible = Counter(trace.case_id for trace in traces if trace.answer.strip())
        selected = [
            trace
            for trace in traces
            if trace.answer.strip()
            and eligible[trace.case_id] > 1
            and (trace.case_id, trace.run_index) not in self._answer_embeddings
        ]
        if not selected:
            return self._answer_embeddings or None
        before_budget = self.client.budget.calls
        before_stats = len(self.client.call_stats)
        total_latency_ms = 0.0
        for offset in range(0, len(selected), ANSWER_EMBEDDING_BATCH_SIZE):
            batch = selected[offset : offset + ANSWER_EMBEDDING_BATCH_SIZE]
            result = self.client.embed(
                [trace.answer for trace in batch],
                task_type="SEMANTIC_SIMILARITY",
                purpose="embed",
            )
            if len(result.embeddings) != len(batch):
                raise ValueError("Answer embedder returned a different vector count")
            self._answer_embeddings.update(
                {
                    (trace.case_id, trace.run_index): vector
                    for trace, vector in zip(batch, result.embeddings, strict=True)
                }
            )
            total_latency_ms += result.latency_ms
        stats = self.client.call_stats[before_stats:]
        new_cost = {
            "llm_calls": max(self.client.budget.calls - before_budget, 0),
            "input_tokens": sum(entry.input_tokens for entry in stats),
            "output_tokens": sum(entry.output_tokens for entry in stats),
        }
        for key, value in new_cost.items():
            self._embedding_cost[key] = self._embedding_cost.get(key, 0) + value
        self._embedding_cost["latency_ms"] = (
            self._embedding_cost.get("latency_ms", 0) + total_latency_ms
        )
        return self._answer_embeddings

    @staticmethod
    def _deserialize_embeddings(
        values: dict[str, list[float]],
    ) -> dict[tuple[str, int], list[float]]:
        embeddings: dict[tuple[str, int], list[float]] = {}
        for key, vector in values.items():
            case_id, separator, run_index = key.rpartition(":")
            if not separator or not run_index.isdigit():
                raise ValueError(f"Invalid answer embedding key: {key!r}")
            embeddings[(case_id, int(run_index))] = vector
        return embeddings

    def _report(
        self,
        cases: list[GoldenCase],
        traces: list[RunTrace],
        runs: int,
        embeddings: dict[tuple[str, int], list[float]] | None,
    ) -> dict[str, Any]:
        trace_rows = [trace.model_dump(mode="json") for trace in traces]
        human_labels = load_human_labels(ROOT / "evals" / "human_labels.jsonl")
        judge_labels = judgement_map(trace_rows)
        metrics = calculate_metrics(
            cases,
            traces,
            n=len(cases),
            k=runs,
            answer_embeddings=embeddings,
        )
        cost = metrics["cost"]
        embedding_cost = self._embedding_cost
        for key in ("llm_calls", "input_tokens", "output_tokens"):
            cost[key] += embedding_cost.get(key, 0)
        cost["answer_embedding_latency_ms"] = embedding_cost.get("latency_ms", 0)
        report = {
            "schema_version": 1,
            "git_sha": self.settings.git_sha,
            "prompt_version": PROMPT_VERSION,
            "models": {
                "agent": self.settings.gemini_model or None,
                "judge": self.settings.gemini_judge_model or None,
                "embedding": self.settings.gemini_embed_model or None,
            },
            "pack": self.settings.pack,
            "mode": self.settings.llm_mode,
            "timestamp": datetime.now(UTC).isoformat(),
            "n": len(cases),
            "k": runs,
            "metrics": metrics,
            "cases": [
                {
                    "case": case.model_dump(mode="json"),
                    "runs": [
                        {
                            **trace.model_dump(mode="json"),
                            "pass": self._trace_passes(case, trace),
                        }
                        for trace in traces
                        if trace.case_id == case.id
                    ],
                    "pass": all(
                        self._trace_passes(case, trace)
                        for trace in traces
                        if trace.case_id == case.id
                    )
                    and any(trace.case_id == case.id for trace in traces),
                }
                for case in cases
            ],
            "traces": trace_rows,
            "answer_embeddings": {
                f"{case_id}:{run_index}": vector
                for (case_id, run_index), vector in self._answer_embeddings.items()
            },
            "answer_embedding_cost": embedding_cost,
            "audit": audit_judgements(judge_labels, human_labels),
        }
        if self._previous_metrics:
            report["previous_metrics"] = self._previous_metrics
        return report

    @staticmethod
    def _trace_passes(case: GoldenCase, trace: RunTrace) -> bool:
        if trace.decision != case.expected_decision:
            return False
        if not _tool_name_match(case, trace) or not _tool_args_match(case, trace):
            return False
        if not all(trace.numeric_results.values()):
            return False
        if case.source_docs and trace.decision == "answered":
            if not trace.citations or not set(trace.citations).issubset(
                set(trace.retrieved_doc_ids)
            ):
                return False
            if not set(trace.citations).intersection(case.source_docs):
                return False
        if case.must_include and len(trace.must_include_results) != len(
            case.must_include
        ):
            return False
        if any(
            not trace.must_include_results.get(item, False)
            for item in case.must_include
        ):
            return False
        if case.must_not_include and len(trace.must_not_include_results) != len(
            case.must_not_include
        ):
            return False
        if any(
            not trace.must_not_include_results.get(item, False)
            for item in case.must_not_include
        ):
            return False
        return True

    @staticmethod
    def _same_case(case: GoldenCase, saved: dict[str, Any]) -> bool:
        current = case.model_dump(mode="json")
        previous = dict(saved)
        current.pop("review_status", None)
        previous.pop("review_status", None)
        return current == previous

    def _save(
        self,
        cases: list[GoldenCase],
        traces: list[RunTrace],
        runs: int,
    ) -> None:
        report = self._report(
            cases,
            traces,
            runs,
            self._answer_embeddings or None,
        )
        self._persist(report)

    def _persist(self, report: dict[str, Any]) -> None:
        write_json(self.output_path, report)
        markdown = render_report(report)
        markdown_path = (
            REPORT_ROOT / "report.md"
            if self.output_path.resolve() == (REPORT_ROOT / "latest.json").resolve()
            else self.output_path.with_suffix(".md")
        )
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
        temporary_path.write_text(markdown, encoding="utf-8")
        temporary_path.replace(markdown_path)


def render_report(report: dict[str, Any]) -> str:
    """Render a concise human-readable scorecard with explicit sample sizes."""
    lines = [
        "# Evaluation report",
        "",
        f"- Pack: `{report['pack']}`",
        f"- Mode: `{report['mode']}`",
        f"- Git SHA: `{report['git_sha']}`",
        f"- Prompt version: `{report['prompt_version']}`",
        f"- Cases (n): {report['n']}",
        f"- Repeats (k): {report['k']}",
        f"- Timestamp: {report['timestamp']}",
    ]
    audit = report["audit"]
    lines.extend(["", "## Judge audit", ""])
    if audit["status"] != "complete":
        lines.append("Not available: no matching manually reviewed human labels.")
    else:
        kappa = audit["cohen_kappa"]
        kappa_text = "undefined" if kappa is None else f"{kappa:.3f}"
        if kappa is not None and kappa < 0.60:
            kappa_text += " (low agreement)"
        lines.append(
            f"- n={audit['n']}; raw agreement={audit['raw_agreement']:.3f}; "
            f"Cohen's kappa={kappa_text}"
        )
        lines.append(f"- Disagreements: {len(audit['disagreements'])}")
    lines.extend(
        [
            "",
            "| Metric | Result | n | k |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in report["metrics"].items():
        if name == "cost":
            value = (
                f"{result['llm_calls']} calls; {result['input_tokens']} input / "
                f"{result['output_tokens']} output tokens; "
                f"{result['mean_latency_ms'] or 0:.1f} ms mean"
            )
        else:
            metric_value = result["value"]
            if metric_value is None:
                value = "not available"
            else:
                count = f"{result['numerator']:g} of {result['denominator']:g}"
                value = f"{count} ({metric_value:.3f})"
        lines.append(f"| {name} | {value} | {result['n']} | {result['k']} |")
    failed_cases = [case for case in report["cases"] if not case.get("pass", False)]
    lines.extend(["", "## Cases needing review", ""])
    if not failed_cases:
        lines.append("No case failures in this run.")
    for row in failed_cases:
        case = row["case"]
        actual = ", ".join(sorted({run["decision"] for run in row["runs"]}))
        lines.append(
            f"- `{case['id']}` ({case['category']}): {case['question']} "
            f"(expected `{case['expected_decision']}`, actual `{actual or 'not run'}`)"
        )
    return "\n".join(lines) + "\n"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default=None)
    parser.add_argument("--subset", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--mode", choices=["live", "replay", "record"], default=None)
    parser.add_argument("--out", type=Path, default=REPORT_ROOT / "latest.json")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--case-file", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; all provider operations go through LLMClient."""
    args = _arguments(argv)
    settings = get_settings()
    updates: dict[str, Any] = {}
    if args.pack:
        updates["pack"] = args.pack
    if args.mode:
        updates["llm_mode"] = args.mode
    if args.max_calls is not None:
        if args.max_calls < 1:
            raise ValueError("--max-calls must be positive")
        updates["daily_call_cap"] = min(settings.daily_call_cap, args.max_calls)
    settings = settings.model_copy(update=updates)
    if args.pack and args.pack != get_settings().pack:
        raise ValueError(
            "PACK must match --pack because policy and retrieval load from "
            "configured pack."
        )
    case_path = args.case_file or GOLDEN_ROOT / f"{settings.pack}.jsonl"
    cases = load_cases(case_path, allow_drafts=args.allow_draft)
    if args.subset == "smoke":
        cases = [case for case in cases if "smoke" in case.tags]
        if not cases:
            raise ValueError("No golden cases are tagged smoke.")
    backend = RateLimitedBackend(
        GeminiClient(settings),
        requests_per_minute=settings.rate_limit_per_min,
    )
    client = LLMClient(
        settings,
        backend=backend,
        budget=DailyCallBudget(settings.daily_call_cap),
    )
    judge = LLMJudge(client)
    runner = EvaluationRunner(settings, client, args.out, judge=judge)
    try:
        report = runner.run(cases, runs=args.runs, resume=args.resume)
    except Exception:
        logger.exception("Evaluation run failed")
        raise
    print(render_report(report), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
