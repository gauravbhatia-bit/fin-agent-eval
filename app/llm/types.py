"""Typed request and response values shared by LLM backends."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """A model-requested function call, not yet executed."""

    name: str
    arguments: dict[str, Any]


class LLMResult(BaseModel):
    """Normalized text/function-call response and usage metadata."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0


class GenerationRequest(BaseModel):
    """Stable request structure used for call cassettes."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    response_schema: dict[str, Any] | None = None
    purpose: str
    prompt_version: str


class EmbeddingRequest(BaseModel):
    """Stable embedding request structure used for call cassettes."""

    model_config = ConfigDict(extra="forbid")

    model: str
    texts: list[str]
    purpose: str
    task_type: str
    prompt_version: str


class EmbeddingResult(BaseModel):
    """Normalized embedding response."""

    embeddings: list[list[float]]
    input_tokens: int = 0
    latency_ms: float = 0
