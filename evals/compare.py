"""Compare two machine-readable evaluation reports."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

METRICS = (
    "decision_accuracy",
    "refusal_precision",
    "refusal_recall",
    "over_refusal_rate",
    "tool_name_accuracy",
    "tool_args_accuracy",
    "numeric_fidelity",
    "content_score",
    "grounding_rate",
    "consistency_decision",
    "consistency_tools",
    "consistency_answer",
)


def compare_reports(
    before: dict[str, Any],
    after: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Compute metric deltas and case pass/fail transitions."""
    deltas: list[dict[str, Any]] = []
    for metric in METRICS:
        first = before.get("metrics", {}).get(metric, {}).get("value")
        second = after.get("metrics", {}).get(metric, {}).get("value")
        deltas.append(
            {
                "metric": metric,
                "before": first,
                "after": second,
                "delta": (
                    float(second) - float(first)
                    if first is not None and second is not None
                    else None
                ),
            }
        )
    old_cases = {
        row["case"]["id"]: bool(row["pass"]) for row in before.get("cases", [])
    }
    new_cases = {row["case"]["id"]: bool(row["pass"]) for row in after.get("cases", [])}
    transitions = [
        {
            "case_id": case_id,
            "transition": "pass-to-fail" if old_cases[case_id] else "fail-to-pass",
        }
        for case_id in sorted(old_cases.keys() & new_cases.keys())
        if old_cases[case_id] != new_cases[case_id]
    ]
    return deltas, transitions


def render_comparison(
    deltas: list[dict[str, Any]],
    transitions: list[dict[str, str]],
) -> str:
    """Format metric deltas and case flips."""
    lines = [
        "| Metric | Before | After | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in deltas:
        values = (
            "N/A"
            if row["delta"] is None
            else (f"{row['before']:.3f} | {row['after']:.3f} | {row['delta']:+.3f}")
        )
        if row["delta"] is None:
            lines.append(f"| {row['metric']} | N/A | N/A | N/A |")
        else:
            lines.append(f"| {row['metric']} | {values} |")
    lines.extend(["", "## Case transitions", ""])
    if transitions:
        lines.extend(
            f"- `{row['case_id']}`: {row['transition']}" for row in transitions
        )
    else:
        lines.append("No pass/fail transitions among common cases.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    deltas, transitions = compare_reports(before, after)
    print(render_comparison(deltas, transitions), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
