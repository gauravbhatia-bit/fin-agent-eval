"""Deterministic local human-escalation stub."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EscalateInput(BaseModel):
    """Reason for a simulated human escalation."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        """Reject whitespace-only reasons."""
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value.strip()


class EscalateOutput(BaseModel):
    """A predictable local ticket identifier and explanatory message."""

    ticket_id: str
    message: str


def escalate_to_human(data: EscalateInput) -> EscalateOutput:
    """Return a hash-derived fake ticket; no person or service is contacted."""
    digest = hashlib.sha256(data.reason.casefold().encode("utf-8")).hexdigest()[:12]
    return EscalateOutput(
        ticket_id=f"DEMO-{digest}",
        message="This is a fictional local escalation stub; no human was contacted.",
    )
