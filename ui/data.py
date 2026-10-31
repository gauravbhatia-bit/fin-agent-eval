"""Pure data shaping for evaluation dashboard views."""

from collections import Counter, defaultdict
from typing import Any

DECISIONS = ("answered", "refused", "escalated", "no_answer")


def metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare reported metric values with the configured baseline."""
    baseline = report.get("baseline", {})
    targets = baseline.get("metrics", {})
    lower_is_better = set(baseline.get("lower_is_better", []))
    tolerance = float(baseline.get("tolerance_pct_points", 0)) / 100
    rows = []
    for name, target in targets.items():
        metric = report.get("metrics", {}).get(name, {})
        current = metric.get("value")
        if current is None:
            status = "N/A"
        else:
            passed = (
                float(current) <= float(target) + tolerance
                if name in lower_is_better
                else float(current) >= float(target) - tolerance
            )
            status = "PASS" if passed else "FAIL"
        rows.append(
            {
                "metric": name,
                "current": current,
                "baseline": target,
                "status": status,
                "n": metric.get("n", report.get("n", 0)),
                "k": metric.get("k", report.get("k", 0)),
            }
        )
    return rows


def confusion_matrix(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count expected versus observed decisions across case runs."""
    counts: Counter[tuple[str, str]] = Counter()
    for case in cases:
        expected = case.get("expected_decision")
        if expected not in DECISIONS:
            continue
        for actual in case.get("actual_decisions", []):
            if actual in DECISIONS:
                counts[(expected, actual)] += 1
    return [
        {
            "expected": expected,
            **{actual: counts[(expected, actual)] for actual in DECISIONS},
        }
        for expected in DECISIONS
    ]


def consistency_by_category(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average each case's modal decision share within its category."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        decisions = case.get("actual_decisions", [])
        category = case.get("category")
        if not isinstance(category, str) or len(decisions) < 2:
            continue
        frequencies = Counter(decisions)
        grouped[category].append(max(frequencies.values()) / len(decisions))
    return [
        {
            "category": category,
            "consistency": sum(scores) / len(scores),
            "cases": len(scores),
        }
        for category, scores in sorted(grouped.items())
    ]


def failed_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return case summaries marked failed by the evaluation runner."""
    return [case for case in cases if not case.get("pass", False)]


def metric_deltas(
    metrics: dict[str, Any],
    previous_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Calculate current-minus-previous deltas for available metrics."""
    rows = []
    for name, current in metrics.items():
        if not isinstance(current, dict) or name not in previous_metrics:
            continue
        previous = previous_metrics[name]
        if (
            not isinstance(previous, dict)
            or "value" not in current
            or "value" not in previous
        ):
            continue
        before, after = previous.get("value"), current.get("value")
        rows.append(
            {
                "metric": name,
                "previous": before,
                "current": after,
                "delta": (
                    float(after) - float(before)
                    if before is not None and after is not None
                    else None
                ),
            }
        )
    return rows
