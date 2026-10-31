from collections.abc import Sequence
from pathlib import Path

import pytest

from app.retrieval.hybrid import HybridRetriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def __call__(
        self,
        texts: Sequence[str],
        task_type: str,
    ) -> list[list[float]]:
        values = list(texts)
        self.calls.append((values, task_type))
        return [
            [1.0, float(index + 1) / (len(values) + 1)] for index in range(len(values))
        ]


def test_hybrid_retriever_returns_scored_hits_and_reuses_model_cache(
    tmp_path: Path,
) -> None:
    first_embedder = FakeEmbedder()
    retriever = HybridRetriever(
        first_embedder,
        embedding_model="fake-embed-v1",
        cache_directory=tmp_path,
    )

    first_hits = retriever.search("What is an ETF?", top_k=3)

    assert len(first_hits) == 3
    assert all(0 <= hit.score <= 1 for hit in first_hits)
    assert first_hits[0].doc_id
    assert first_embedder.calls[0][1] == "RETRIEVAL_DOCUMENT"
    assert first_embedder.calls[-1][1] == "RETRIEVAL_QUERY"

    second_embedder = FakeEmbedder()
    second = HybridRetriever(
        second_embedder,
        embedding_model="fake-embed-v1",
        cache_directory=tmp_path,
    )
    second.search("What is an ETF?", top_k=1)

    assert len(second_embedder.calls) == 1
    assert second_embedder.calls[0][1] == "RETRIEVAL_QUERY"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_hybrid_retriever_uses_distinct_cache_for_each_model(tmp_path: Path) -> None:
    retriever = HybridRetriever(
        FakeEmbedder(),
        embedding_model="fake-embed-v1",
        cache_directory=tmp_path,
    )
    retriever.search("ETF", top_k=1)
    other_model = HybridRetriever(
        FakeEmbedder(),
        embedding_model="fake-embed-v2",
        cache_directory=tmp_path,
    )
    other_model.search("ETF", top_k=1)

    assert len(list(tmp_path.glob("*.json"))) == 2


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[]],
        [[0.0, 0.0]],
        [[1.0, float("nan")]],
        [[1.0], [1.0, 2.0]],
    ],
)
def test_embedding_provider_results_are_validated(
    vectors: list[list[float]],
) -> None:
    retriever = HybridRetriever(
        lambda texts, task_type: vectors,
        embedding_model="fake",
        cache_directory=Path("."),
    )

    with pytest.raises(ValueError):
        retriever._embed(["text"], "RETRIEVAL_QUERY")


def test_invalid_retrieval_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(
            FakeEmbedder(),
            embedding_model="fake",
            cache_directory=Path("."),
            bm25_weight=2,
        )
