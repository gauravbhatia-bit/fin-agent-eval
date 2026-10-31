"""Deterministic local keyword search for the Phase 2 knowledge pack."""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.retrieval.base import Hit
from app.tools.common import pack_directory
from app.tools.pack_schemas import DocumentMetadata


class SearchInput(BaseModel):
    """A short query and bounded result count for knowledge search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject whitespace-only queries."""
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class SearchOutput(BaseModel):
    """Ranked local knowledge matches."""

    results: list[Hit]


class _Document(BaseModel):
    metadata: DocumentMetadata
    body: str


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.casefold()))


def _load_documents() -> list[_Document]:
    docs_dir = pack_directory() / "docs"
    documents: list[_Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        front_matter, body = _read_document(path)
        metadata = DocumentMetadata.model_validate(yaml.safe_load(front_matter))
        documents.append(
            _Document(
                metadata=metadata,
                body=body,
            )
        )
    return documents


def _read_document(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f"Invalid YAML front matter in {path}")
    return parts[1], parts[2].strip()


def search_knowledge(data: SearchInput) -> SearchOutput:
    """Search the fictional knowledge pack by deterministic token overlap."""
    query_tokens = _tokens(data.query)
    if not query_tokens:
        return SearchOutput(results=[])

    ranked: list[Hit] = []
    for document in _load_documents():
        document_tokens = _tokens(f"{document.metadata.title} {document.body}")
        overlap = len(query_tokens & document_tokens)
        if overlap:
            ranked.append(
                Hit(
                    doc_id=document.metadata.doc_id,
                    title=document.metadata.title,
                    snippet=" ".join(document.body.split())[:240],
                    score=overlap / len(query_tokens),
                )
            )

    ranked.sort(key=lambda hit: (-hit.score, hit.doc_id))
    return SearchOutput(results=ranked[: data.top_k])
