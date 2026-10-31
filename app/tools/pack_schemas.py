"""Validated schemas for policy and knowledge-document pack files."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PolicyPack(BaseModel):
    """Safety and disclaimer settings for the fictional demo pack."""

    model_config = ConfigDict(extra="forbid")

    pack_name: Literal["eu_us_etf"]
    fictional_data: Literal[True]
    disclaimer: str = Field(min_length=10)
    refused_topics: list[str] = Field(min_length=1)
    escalation_topics: list[str] = Field(min_length=1)
    rule_patterns: dict[str, list[str]]


class DocumentMetadata(BaseModel):
    """Required front matter for every pack knowledge document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=3, max_length=120)
    fictional_data: bool
    source_url: str
    retrieved_on: str
    verified: Literal[False]

    @model_validator(mode="after")
    def validate_source_metadata(self) -> "DocumentMetadata":
        """Require source URL and retrieval date together when a source is recorded."""
        if bool(self.source_url) != bool(self.retrieved_on):
            raise ValueError("source_url and retrieved_on must be provided together")
        if self.retrieved_on:
            date.fromisoformat(self.retrieved_on)
        return self
