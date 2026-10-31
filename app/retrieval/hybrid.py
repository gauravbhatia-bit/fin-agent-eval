"""Hybrid BM25 and Gemini-embedding retrieval with a local FAISS index."""

import hashlib
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import faiss
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.retrieval.base import Hit, KnowledgeDocument
from app.tools.common import pack_directory
from app.tools.pack_schemas import DocumentMetadata
from app.tools.search import _read_document

Embedder = Callable[[Sequence[str], str], list[list[float]]]
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class _EmbeddingCache(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vectors: dict[str, list[float]]


class HybridRetriever:
    """Rank the local corpus with normalized BM25 and dense cosine scores."""

    def __init__(
        self,
        embedder: Embedder,
        embedding_model: str,
        cache_directory: Path | None = None,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        if not 0 <= bm25_weight <= 1 or not 0 <= dense_weight <= 1:
            raise ValueError("retrieval weights must be between zero and one")
        if bm25_weight + dense_weight == 0:
            raise ValueError("at least one retrieval weight must be nonzero")
        weight_total = bm25_weight + dense_weight
        self.bm25_weight = bm25_weight / weight_total
        self.dense_weight = dense_weight / weight_total
        self.embedder = embedder
        self.embedding_model = embedding_model
        self.pack_name = get_settings().pack
        self.cache_directory = cache_directory or (
            Path(__file__).resolve().parents[2]
            / ".cache"
            / "embeddings"
            / self.pack_name
        )
        self.documents = self._load_documents()
        if not self.documents:
            raise ValueError(
                f"No knowledge documents found for pack {self.pack_name!r}"
            )
        self._tokens = [
            self._tokenize(doc.title + " " + doc.body) for doc in self.documents
        ]
        self._bm25 = BM25Okapi(self._tokens)
        self._index: faiss.IndexFlatIP | None = None

    def search(self, query: str, top_k: int) -> list[Hit]:
        """Return at most `top_k` matches ranked by 0.5 BM25 and 0.5 dense score."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        bm25_scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float32)
        dense_scores = self._dense_scores(query)
        normalized_bm25 = self._minmax(bm25_scores)
        normalized_dense = self._minmax(dense_scores)
        combined = (
            self.bm25_weight * normalized_bm25 + self.dense_weight * normalized_dense
        )
        ordered = sorted(
            range(len(self.documents)),
            key=lambda index: (-float(combined[index]), self.documents[index].doc_id),
        )
        return [
            Hit(
                doc_id=self.documents[index].doc_id,
                title=self.documents[index].title,
                snippet=" ".join(self.documents[index].body.split())[:240],
                score=min(1.0, max(0.0, float(combined[index]))),
            )
            for index in ordered[:top_k]
        ]

    def _dense_scores(self, query: str) -> np.ndarray:
        index = self._get_index()
        query_vector = self._embed(
            [f"Represent this query for retrieving relevant documents: {query}"],
            "RETRIEVAL_QUERY",
        )[0]
        query_matrix = self._normalize_vectors([query_vector])
        if query_matrix.shape[1] != index.d:
            raise ValueError("Query and document embedding dimensions do not match")
        scores, indices = index.search(query_matrix, len(self.documents))
        ordered_scores = np.zeros(len(self.documents), dtype=np.float32)
        for score, index_value in zip(scores[0], indices[0], strict=True):
            if index_value >= 0:
                ordered_scores[int(index_value)] = score
        return ordered_scores

    def _get_index(self) -> faiss.IndexFlatIP:
        if self._index is not None:
            return self._index
        cache = self._read_cache()
        document_vectors: list[list[float] | None] = [None] * len(self.documents)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        for index, document in enumerate(self.documents):
            key = self._document_hash(document)
            if key in cache:
                document_vectors[index] = cache[key]
            else:
                missing_indices.append(index)
                missing_texts.append(
                    f"Represent this document for retrieval: "
                    f"{document.title}\n{document.body}"
                )
        if missing_texts:
            vectors = self._embed(missing_texts, "RETRIEVAL_DOCUMENT")
            for document_index, vector in zip(missing_indices, vectors, strict=True):
                document_vectors[document_index] = vector
                cache[self._document_hash(self.documents[document_index])] = vector
            self._write_cache(cache)
        complete_vectors = [vector for vector in document_vectors if vector is not None]
        if len(complete_vectors) != len(self.documents):
            raise ValueError("Embedding cache did not produce one vector per document")
        normalized = self._normalize_vectors(complete_vectors)
        index = faiss.IndexFlatIP(normalized.shape[1])
        index.add(normalized)
        self._index = index
        return index

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        vectors = self.embedder(texts, task_type)
        if len(vectors) != len(texts):
            raise ValueError("Embedding provider returned a different vector count")
        if not vectors or any(not vector for vector in vectors):
            raise ValueError("Embedding provider returned an empty vector")
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Embedding provider returned non-numeric vectors"
            ) from error
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Embedding provider returned invalid vectors")
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError("Embedding provider returned a zero vector")
        if any(len(vector) != matrix.shape[1] for vector in vectors):
            raise ValueError("Embedding provider returned inconsistent dimensions")
        return matrix.tolist()

    def _load_documents(self) -> list[KnowledgeDocument]:
        documents: list[KnowledgeDocument] = []
        docs_directory = pack_directory() / "docs"
        for path in sorted(docs_directory.glob("*.md")):
            front_matter, body = _read_document(path)
            metadata = DocumentMetadata.model_validate(yaml.safe_load(front_matter))
            documents.append(
                KnowledgeDocument(
                    doc_id=metadata.doc_id,
                    title=metadata.title,
                    body=body,
                )
            )
        return documents

    def _read_cache(self) -> dict[str, list[float]]:
        path = self._cache_path()
        if not path.exists():
            return {}
        payload = _EmbeddingCache.model_validate_json(path.read_text(encoding="utf-8"))
        return payload.vectors

    def _write_cache(self, cache: dict[str, list[float]]) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                delete=False,
            ) as file:
                file.write(
                    _EmbeddingCache(vectors=cache).model_dump_json(
                        indent=2,
                    )
                )
                file.flush()
                os.fsync(file.fileno())
                temporary_path = file.name
            os.replace(temporary_path, path)
        finally:
            if temporary_path and Path(temporary_path).exists():
                Path(temporary_path).unlink()

    def _cache_path(self) -> Path:
        model_hash = hashlib.sha256(self.embedding_model.encode("utf-8")).hexdigest()[
            :16
        ]
        return self.cache_directory / f"{model_hash}.json"

    @staticmethod
    def _document_hash(document: KnowledgeDocument) -> str:
        payload = f"{document.title}\n{document.body}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.casefold())

    @staticmethod
    def _normalize_vectors(vectors: list[list[float]]) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Embeddings must be a finite two-dimensional matrix")
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError("Embeddings must not contain zero vectors")
        faiss.normalize_L2(matrix)
        return matrix

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        minimum = float(scores.min())
        maximum = float(scores.max())
        if maximum == minimum:
            return np.full_like(scores, 0.5, dtype=np.float32)
        return (scores - minimum) / (maximum - minimum)
