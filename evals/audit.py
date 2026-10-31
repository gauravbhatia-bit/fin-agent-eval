"""Audit judge labels against manually reviewed human labels."""

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HumanLabel(BaseModel):
    """One human-reviewed binary label for one case criterion."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    item: str
    satisfied: bool
    reviewer: str = Field(min_length=1)


def load_human_labels(path: Path) -> list[HumanLabel]:
    """Read manually authored JSONL labels; never synthesize missing labels."""
    if not path.exists():
        return []
    labels: list[HumanLabel] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                label = HumanLabel.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid human label"
                ) from error
            key = (label.case_id, label.item)
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate human label {key}")
            seen.add(key)
            labels.append(label)
    return labels


def audit_judgements(
    judge_labels: dict[tuple[str, str], bool],
    human_labels: list[HumanLabel],
) -> dict[str, Any]:
    """Compute raw agreement and Cohen's kappa over matched labels."""
    matched = [
        (label, judge_labels[(label.case_id, label.item)])
        for label in human_labels
        if (label.case_id, label.item) in judge_labels
    ]
    if not matched:
        return {
            "status": "not_available",
            "n": 0,
            "raw_agreement": None,
            "cohen_kappa": None,
            "disagreements": [],
        }
    human = [label.satisfied for label, _ in matched]
    judged = [decision for _, decision in matched]
    count = len(matched)
    observed_agreement = sum(a == b for a, b in zip(human, judged, strict=True)) / count
    human_positive = sum(human) / count
    judge_positive = sum(judged) / count
    expected_agreement = human_positive * judge_positive + (1 - human_positive) * (
        1 - judge_positive
    )
    kappa = (
        (observed_agreement - expected_agreement) / (1 - expected_agreement)
        if expected_agreement < 1
        else math.nan
    )
    disagreements = [
        {
            "case_id": label.case_id,
            "item": label.item,
            "human": label.satisfied,
            "judge": decision,
        }
        for label, decision in matched
        if label.satisfied != decision
    ]
    return {
        "status": "complete",
        "n": len(matched),
        "raw_agreement": observed_agreement,
        "cohen_kappa": kappa if math.isfinite(kappa) else None,
        "disagreements": disagreements,
    }


def judgement_map(traces: list[dict[str, Any]]) -> dict[tuple[str, str], bool]:
    """Flatten trace-level judge outcomes for audit matching."""
    output: dict[tuple[str, str], bool] = {}
    for trace in traces:
        case_id = str(trace["case_id"])
        for key in ("must_include_results", "must_not_include_results"):
            output.update(
                {
                    (case_id, str(item)): bool(value)
                    for item, value in trace.get(key, {}).items()
                }
            )
    return output
