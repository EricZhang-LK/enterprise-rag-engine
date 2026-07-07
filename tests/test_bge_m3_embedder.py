from collections.abc import Sequence
from typing import Any

import pytest

from enterprise_rag_engine import BgeM3Embedder, EmbeddingRequest, SentenceTransformerEmbedder


class FakeSentenceTransformer:
    def __init__(self, vectors: Any) -> None:
        self._vectors = vectors
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
        batch_size: int,
    ) -> Any:
        """Record the encoding options so tests cover the adapter contract."""

        self.calls.append(
            {
                "sentences": tuple(sentences),
                "normalize_embeddings": normalize_embeddings,
                "batch_size": batch_size,
            }
        )
        return self._vectors


class FakeNumpyLikeArray:
    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self._rows]


def test_bge_m3_embedder_uses_default_model_and_returns_batch_result() -> None:
    model = FakeSentenceTransformer(vectors=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)))
    embedder = BgeM3Embedder(model=model, batch_size=16)

    result = embedder.embed(
        EmbeddingRequest(
            texts=("什么是企业级 RAG", "如何做 metadata filter"),
            normalize=True,
        )
    )

    assert result.model == "BAAI/bge-m3"
    assert result.dimension == 3
    assert result.embeddings == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    assert [item.model for item in result.results] == ["BAAI/bge-m3", "BAAI/bge-m3"]
    assert model.calls == [
        {
            "sentences": ("什么是企业级 RAG", "如何做 metadata filter"),
            "normalize_embeddings": True,
            "batch_size": 16,
        }
    ]


def test_sentence_transformer_embedder_respects_request_normalization() -> None:
    model = FakeSentenceTransformer(vectors=((1.0, 2.0),))
    embedder = SentenceTransformerEmbedder(
        model_name="BAAI/bge-m3",
        model=model,
        batch_size=8,
        normalize=True,
    )

    result = embedder.embed(EmbeddingRequest(texts=("raw vector",), normalize=False))

    assert result.embeddings == ((1.0, 2.0),)
    assert model.calls[0]["normalize_embeddings"] is False


def test_sentence_transformer_embedder_accepts_numpy_like_output() -> None:
    model = FakeSentenceTransformer(vectors=FakeNumpyLikeArray(((1, 2), (3, 4))))
    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-m3", model=model)

    result = embedder.embed(EmbeddingRequest(texts=("a", "b")))

    assert result.embeddings == ((1.0, 2.0), (3.0, 4.0))


def test_sentence_transformer_embedder_rejects_mismatched_output_count() -> None:
    model = FakeSentenceTransformer(vectors=((1.0, 2.0),))
    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-m3", model=model)

    with pytest.raises(ValueError, match="same number"):
        embedder.embed(EmbeddingRequest(texts=("a", "b")))
