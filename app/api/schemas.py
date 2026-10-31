"""Request and response models for chat endpoints."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    """A synthetic finance-education question and in-memory session key."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=10_000)

    @field_validator("session_id", "message")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        """Reject whitespace-only values and normalize leading/trailing spaces."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class ChatResponse(BaseModel):
    """Traceable outcome for one chat turn."""

    decision: Literal["answered", "refused", "escalated", "no_answer"]
    answer: str
    tool_calls: list[dict[str, Any]]
    citations: list[str]
    disclaimer: str
    trace_id: str
