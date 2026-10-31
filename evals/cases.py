"""Validated golden evaluation case schema and JSONL loading."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.api.schemas import ChatResponse

Decision = Literal["answered", "refused", "escalated", "no_answer"]
Category = Literal[
    "answerable",
    "tool_needed",
    "must_refuse",
    "escalate",
    "unanswerable",
    "trap",
]


class ExpectedTool(BaseModel):
    """Expected deterministic tool invocation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, int | float | str | bool | None]


class NumericCheck(BaseModel):
    """A tool-output number expected to appear in the final answer."""

    model_config = ConfigDict(extra="forbid")

    label: str
    source: str
    must_appear_in_answer: bool = True


class GoldenCase(BaseModel):
    """Single human-reviewable test case for an agent turn."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: Category
    question: str = Field(min_length=1, max_length=10_000)
    expected_decision: Decision
    expected_tools: list[ExpectedTool] = Field(default_factory=list)
    numeric_checks: list[NumericCheck] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    source_docs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    review_status: Literal["draft", "approved"] = "draft"
    repeat_question: str | None = None
    repeat_count: int | None = Field(default=None, ge=1, le=10_000)

    @model_validator(mode="after")
    def expand_repeat_question(self) -> "GoldenCase":
        """Expand intentionally long boundary cases without huge JSONL lines."""
        if self.repeat_count is None and self.repeat_question is None:
            return self
        if not self.repeat_question or self.repeat_count is None:
            raise ValueError("repeat_question and repeat_count must be set together")
        self.question = self.repeat_question * self.repeat_count
        return self


class RunTrace(BaseModel):
    """One observed run of a golden case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    run_index: int = Field(ge=0)
    category: Category
    expected_decision: Decision
    decision: Decision
    answer: str
    tool_calls: list[dict[str, object]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    disclaimer: str = ""
    trace_id: str = ""
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    must_include_results: dict[str, bool] = Field(default_factory=dict)
    must_not_include_results: dict[str, bool] = Field(default_factory=dict)
    numeric_results: dict[str, bool] = Field(default_factory=dict)


def load_cases(path: Path, *, allow_drafts: bool = False) -> list[GoldenCase]:
    """Load JSONL cases, rejecting malformed, duplicate, or unapproved data."""
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                case = GoldenCase.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid golden case"
                ) from error
            if case.id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate case id {case.id!r}")
            if case.review_status != "approved" and not allow_drafts:
                raise ValueError(
                    f"{path}:{line_number}: case {case.id!r} is not approved; "
                    "review it or pass --allow-draft for development."
                )
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"No golden cases found in {path}")
    return cases


def write_json(path: Path, payload: object) -> None:
    """Persist JSON atomically so interrupted runs leave a valid resume file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def trace_from_response(
    case: GoldenCase,
    run_index: int,
    response: ChatResponse,
    *,
    retrieved_doc_ids: list[str],
    llm_calls: int,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
) -> RunTrace:
    """Create the stable trace shape consumed by metrics and reports."""
    return RunTrace(
        case_id=case.id,
        run_index=run_index,
        category=case.category,
        expected_decision=case.expected_decision,
        decision=response.decision,
        answer=response.answer,
        tool_calls=response.tool_calls,
        citations=response.citations,
        retrieved_doc_ids=retrieved_doc_ids,
        disclaimer=response.disclaimer,
        trace_id=response.trace_id,
        llm_calls=llm_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )
