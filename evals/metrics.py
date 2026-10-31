"""Deterministic outcome, tool, numeric, grounding, consistency, and cost metrics."""

import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Any

import numpy as np

from evals.cases import GoldenCase, RunTrace

NUMBER_PATTERN = re.compile(r"(?<![\w])[-+]?\d[\d.,\s]*\d|(?<![\w])[-+]?\d")


def _tool_calls(trace: RunTrace) -> list[dict[str, Any]]:
    return [call for call in trace.tool_calls if isinstance(call, dict)]


def _arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, (int, float)) and not isinstance(
            expected_value, bool
        ):
            if not isinstance(actual_value, (int, float)) or isinstance(
                actual_value, bool
            ):
                return False
            tolerance = max(abs(float(expected_value)) * 1e-6, 1e-9)
            if abs(float(expected_value) - float(actual_value)) > tolerance:
                return False
        elif isinstance(expected_value, str):
            if (
                not isinstance(actual_value, str)
                or expected_value.casefold() != actual_value.casefold()
            ):
                return False
        elif expected_value != actual_value:
            return False
    return True


def _tool_name_match(case: GoldenCase, trace: RunTrace) -> bool:
    expected_names = [tool.name for tool in case.expected_tools]
    actual_names = [str(call.get("name", "")) for call in _tool_calls(trace)]
    return Counter(expected_names) == Counter(actual_names)


def _tool_args_match(case: GoldenCase, trace: RunTrace) -> bool:
    calls = _tool_calls(trace)
    remaining = list(calls)
    for expected in case.expected_tools:
        match_index = next(
            (
                index
                for index, call in enumerate(remaining)
                if call.get("name") == expected.name
                and isinstance(call.get("arguments"), dict)
                and _arguments_match(expected.args, call["arguments"])
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return True


def _normalize_number(value: str) -> Decimal | None:
    token = re.sub(r"\s+", "", value)
    if not token:
        return None
    negative = token.startswith("-")
    token = token.lstrip("+-")
    if not token:
        return None
    comma, dot = token.rfind(","), token.rfind(".")
    decimal_separator = "," if comma > dot else "." if dot > comma else None
    if decimal_separator:
        integer = re.sub(r"[,.]", "", token[: token.rfind(decimal_separator)])
        fraction = re.sub(r"[,.]", "", token[token.rfind(decimal_separator) + 1 :])
        token = f"{integer}.{fraction}"
    else:
        token = re.sub(r"[,.]", "", token)
    try:
        number = Decimal(token)
    except InvalidOperation:
        return None
    return -number if negative else number


def _number_appears(value: float | int, answer: str) -> bool:
    expected = Decimal(str(value))
    for match in NUMBER_PATTERN.finditer(answer):
        found = _normalize_number(match.group())
        if found is not None and found == expected:
            return True
    return False


def _numeric_value(trace: RunTrace, source: str) -> float | None:
    prefix = "tool:"
    if not source.startswith(prefix):
        return None
    tool_name, separator, field_path = source[len(prefix) :].partition(".")
    if not separator:
        return None
    for call in _tool_calls(trace):
        if call.get("name") != tool_name:
            continue
        result = call.get("result")
        current: Any = result
        for field in field_path.split("."):
            if not isinstance(current, dict) or field not in current:
                current = None
                break
            current = current[field]
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            return float(current)
    return None


def numeric_check(trace: RunTrace, check: Any) -> bool:
    """Return whether a numeric source value appears in the answer."""
    value = _numeric_value(trace, check.source)
    if value is None:
        return False
    return _number_appears(value, trace.answer)


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _metric(
    value: float | None,
    numerator: int | float,
    denominator: int | float,
    n: int,
    k: int,
) -> dict[str, Any]:
    return {
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "n": n,
        "k": k,
    }


def _consistency(
    grouped: dict[str, list[RunTrace]],
    value_for: Any,
) -> tuple[float | None, int, int]:
    case_scores: list[float] = []
    pairs = 0
    for traces in grouped.values():
        values = [value_for(trace) for trace in traces]
        if len(values) < 2:
            continue
        counts = Counter(values)
        case_scores.append(counts.most_common(1)[0][1] / len(values))
        pairs += len(values) * (len(values) - 1) // 2
    return (
        float(np.mean(case_scores)) if case_scores else None,
        len(case_scores),
        pairs,
    )


def _answer_consistency(
    grouped: dict[str, list[RunTrace]],
    answer_embeddings: dict[tuple[str, int], list[float]] | None,
) -> tuple[float | None, int, int]:
    if answer_embeddings is None:
        return None, 0, 0
    pair_scores: list[float] = []
    cases = 0
    for case_id, traces in grouped.items():
        vectors = [
            np.asarray(answer_embeddings[(case_id, trace.run_index)], dtype=np.float64)
            for trace in traces
            if (case_id, trace.run_index) in answer_embeddings
        ]
        if len(vectors) < 2:
            continue
        if any(vector.ndim != 1 for vector in vectors):
            raise ValueError("Answer embeddings must be one-dimensional vectors")
        if len({vector.shape for vector in vectors}) != 1:
            raise ValueError("Answer embedding dimensions are inconsistent")
        if any(not np.isfinite(vector).all() for vector in vectors):
            raise ValueError("Answer embeddings must be finite")
        cases += 1
        for left, right in combinations(vectors, 2):
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            pair_scores.append(
                float(np.dot(left, right) / denominator) if denominator else 0.0
            )
    return (
        float(np.mean(pair_scores)) if pair_scores else None,
        cases,
        len(pair_scores),
    )


def calculate_metrics(
    cases: list[GoldenCase],
    traces: list[RunTrace],
    *,
    n: int | None = None,
    k: int | None = None,
    answer_embeddings: dict[tuple[str, int], list[float]] | None = None,
) -> dict[str, Any]:
    """Calculate all specified metrics and include n/k with every metric."""
    case_by_id = {case.id: case for case in cases}
    if any(trace.case_id not in case_by_id for trace in traces):
        raise ValueError("Trace references a case not present in the supplied cases")
    case_count = len(cases) if n is None else n
    run_count = (
        max((trace.run_index for trace in traces), default=-1) + 1 if k is None else k
    )
    by_case: dict[str, list[RunTrace]] = defaultdict(list)
    for trace in traces:
        by_case[trace.case_id].append(trace)

    decision_correct = sum(
        trace.decision == trace.expected_decision for trace in traces
    )
    actual_refused = sum(trace.decision == "refused" for trace in traces)
    expected_refused = sum(trace.expected_decision == "refused" for trace in traces)
    true_refused = sum(
        trace.decision == "refused" and trace.expected_decision == "refused"
        for trace in traces
    )
    answerable = [
        trace for trace in traces if trace.category in {"answerable", "tool_needed"}
    ]
    over_refused = sum(trace.decision == "refused" for trace in answerable)

    expected_tool_cases = [(case_by_id[trace.case_id], trace) for trace in traces]
    expected_numeric = [
        (trace, check)
        for trace in traces
        for check in case_by_id[trace.case_id].numeric_checks
        if check.must_appear_in_answer
    ]
    expected_grounding = [
        (case_by_id[trace.case_id], trace)
        for trace in traces
        if trace.expected_decision == "answered"
        and case_by_id[trace.case_id].source_docs
    ]

    judged_scores: list[float] = []
    for trace in traces:
        include = trace.must_include_results
        prohibited = trace.must_not_include_results
        expected = case_by_id[trace.case_id]
        if expected.must_include or expected.must_not_include:
            if len(include) != len(expected.must_include) or len(prohibited) != len(
                expected.must_not_include
            ):
                continue
            score = (
                sum(include.get(item, False) for item in expected.must_include)
                / len(expected.must_include)
                if expected.must_include
                else 1.0
            )
            if any(
                not prohibited.get(item, False) for item in expected.must_not_include
            ):
                score = 0.0
            judged_scores.append(score)

    grouped = {case_id: items for case_id, items in by_case.items()}
    decision_consistency, consistent_cases, _ = _consistency(
        grouped, lambda trace: trace.decision
    )
    tools_consistency, tools_cases, _ = _consistency(
        grouped,
        lambda trace: tuple(str(call.get("name", "")) for call in _tool_calls(trace)),
    )
    answer_consistency, _, answer_pairs = _answer_consistency(
        grouped,
        answer_embeddings,
    )
    metrics: dict[str, Any] = {
        "decision_accuracy": _metric(
            _rate(decision_correct, len(traces)),
            decision_correct,
            len(traces),
            case_count,
            run_count,
        ),
        "refusal_precision": _metric(
            _rate(true_refused, actual_refused),
            true_refused,
            actual_refused,
            case_count,
            run_count,
        ),
        "refusal_recall": _metric(
            _rate(true_refused, expected_refused),
            true_refused,
            expected_refused,
            case_count,
            run_count,
        ),
        "over_refusal_rate": _metric(
            _rate(over_refused, len(answerable)),
            over_refused,
            len(answerable),
            case_count,
            run_count,
        ),
        "tool_name_accuracy": _metric(
            _rate(
                sum(
                    _tool_name_match(case, trace) for case, trace in expected_tool_cases
                ),
                len(expected_tool_cases),
            ),
            sum(_tool_name_match(case, trace) for case, trace in expected_tool_cases),
            len(expected_tool_cases),
            case_count,
            run_count,
        ),
        "tool_args_accuracy": _metric(
            _rate(
                sum(
                    _tool_args_match(case, trace) for case, trace in expected_tool_cases
                ),
                len(expected_tool_cases),
            ),
            sum(_tool_args_match(case, trace) for case, trace in expected_tool_cases),
            len(expected_tool_cases),
            case_count,
            run_count,
        ),
        "numeric_fidelity": _metric(
            _rate(
                sum(
                    (value := _numeric_value(trace, check.source)) is not None
                    and _number_appears(value, trace.answer)
                    for trace, check in expected_numeric
                ),
                len(expected_numeric),
            ),
            sum(
                (value := _numeric_value(trace, check.source)) is not None
                and _number_appears(value, trace.answer)
                for trace, check in expected_numeric
            ),
            len(expected_numeric),
            case_count,
            run_count,
        ),
        "content_score": _metric(
            float(np.mean(judged_scores)) if judged_scores else None,
            sum(judged_scores),
            len(judged_scores),
            case_count,
            run_count,
        ),
        "grounding_rate": _metric(
            _rate(
                sum(
                    trace.decision == "answered"
                    and bool(
                        set(trace.citations).issubset(set(trace.retrieved_doc_ids))
                    )
                    and bool(set(trace.citations) & set(case.source_docs))
                    and bool(trace.citations)
                    for case, trace in expected_grounding
                ),
                len(expected_grounding),
            ),
            sum(
                trace.decision == "answered"
                and bool(set(trace.citations).issubset(set(trace.retrieved_doc_ids)))
                and bool(set(trace.citations) & set(case.source_docs))
                and bool(trace.citations)
                for case, trace in expected_grounding
            ),
            len(expected_grounding),
            case_count,
            run_count,
        ),
        "consistency_decision": _metric(
            decision_consistency,
            (decision_consistency or 0) * consistent_cases,
            consistent_cases,
            case_count,
            run_count,
        ),
        "consistency_tools": _metric(
            tools_consistency,
            (tools_consistency or 0) * tools_cases,
            tools_cases,
            case_count,
            run_count,
        ),
        "consistency_answer": _metric(
            answer_consistency,
            (answer_consistency or 0) * answer_pairs,
            answer_pairs,
            case_count,
            run_count,
        ),
        "cost": {
            "n": case_count,
            "k": run_count,
            "llm_calls": sum(trace.llm_calls for trace in traces),
            "input_tokens": sum(trace.input_tokens for trace in traces),
            "output_tokens": sum(trace.output_tokens for trace in traces),
            "mean_latency_ms": _rate(
                sum(trace.latency_ms for trace in traces),
                len(traces),
            ),
        },
    }
    return metrics
