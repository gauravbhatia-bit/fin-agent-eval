"""Small retrieval interface shared by the search tool and later retrievers."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class Hit(BaseModel):
    """A ranked knowledge-document match."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    snippet: str
    score: float = Field(ge=0, le=1)


class KnowledgeDocument(BaseModel):
    """Validated knowledge document loaded from one domain pack."""

    doc_id: str
    title: str
    body: str


class Retriever(Protocol):
    """Interface implemented by pack-backed knowledge retrievers."""

    def search(self, query: str, top_k: int) -> list[Hit]:
        """Return at most `top_k` ranked document matches."""
