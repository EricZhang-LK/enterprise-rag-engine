from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

from enterprise_rag_engine.interfaces import BaseEmbedder
from enterprise_rag_engine.models import (
    Embedding,
    EmbeddingBatchResult,
    EmbeddingRequest,
    EmbeddingResult,
)

BGE_M3_MODEL_NAME = "BAAI/bge-m3"


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embed text batches with a SentenceTransformer-compatible dense model."""

    def __init__(
        self,
        *,
        model_name: str,
        model: Any | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        if batch_size <= 0:
            msg = "batch_size must be greater than 0"
            raise ValueError(msg)
        self._model_name = model_name
        self._model = model if model is not None else self._load_model(model_name)
        self._batch_size = batch_size
        self._normalize = normalize

    def embed(self, request: EmbeddingRequest) -> EmbeddingBatchResult:
        """Convert request texts into normalized dense vectors."""

        normalize = request.normalize if request.normalize is not None else self._normalize
        raw_embeddings = self._model.encode(
            list(request.texts),
            normalize_embeddings=normalize,
            batch_size=self._batch_size,
        )
        embeddings = _coerce_embedding_rows(raw_embeddings)
        if len(embeddings) != len(request.texts):
            msg = "embedding output must contain the same number of rows as input texts"
            raise ValueError(msg)

        results = tuple(
            EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self._model_name,
                metadata=request.metadata,
            )
            for text, embedding in zip(request.texts, embeddings, strict=True)
        )
        return EmbeddingBatchResult.from_results(results=results, model=self._model_name)

    def _load_model(self, model_name: str) -> Any:
        try:
            module = import_module("sentence_transformers")
        except ImportError as exc:
            msg = (
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with `pip install enterprise-rag-engine[embeddings]`."
            )
            raise RuntimeError(msg) from exc
        sentence_transformer = module.__dict__["SentenceTransformer"]
        return sentence_transformer(model_name)


class BgeM3Embedder(SentenceTransformerEmbedder):
    """BGE-M3 dense embedding adapter for enterprise RAG retrieval."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        super().__init__(
            model_name=BGE_M3_MODEL_NAME,
            model=model,
            batch_size=batch_size,
            normalize=normalize,
        )


def _coerce_embedding_rows(raw_embeddings: Any) -> tuple[Embedding, ...]:
    """Normalize list, numpy-like, or dense-vector dict outputs into tuples."""

    if isinstance(raw_embeddings, Mapping):
        raw_embeddings = raw_embeddings.get("dense_vecs", raw_embeddings.get("embeddings"))
    if hasattr(raw_embeddings, "tolist"):
        raw_embeddings = raw_embeddings.tolist()
    if not isinstance(raw_embeddings, Sequence) or isinstance(raw_embeddings, str | bytes):
        msg = "embedding output must be a sequence of vector rows"
        raise TypeError(msg)
    return tuple(_coerce_embedding_row(row) for row in raw_embeddings)


def _coerce_embedding_row(raw_row: Any) -> Embedding:
    if hasattr(raw_row, "tolist"):
        raw_row = raw_row.tolist()
    if not isinstance(raw_row, Sequence) or isinstance(raw_row, str | bytes):
        msg = "each embedding row must be a sequence of numbers"
        raise TypeError(msg)
    return tuple(float(value) for value in raw_row)
