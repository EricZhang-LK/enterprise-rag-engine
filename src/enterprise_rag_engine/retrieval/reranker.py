from __future__ import annotations

import math
from collections.abc import Sequence
from importlib import import_module
from typing import Any

from enterprise_rag_engine.interfaces import BaseReranker
from enterprise_rag_engine.models import RetrievalResult

BGE_RERANKER_V2_M3_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker(BaseReranker):
    """Re-rank retrieved chunks with a CrossEncoder-compatible model."""

    def __init__(
        self,
        *,
        model_name: str = BGE_RERANKER_V2_M3_MODEL_NAME,
        model: Any | None = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            msg = "batch_size must be greater than 0"
            raise ValueError(msg)
        self._model_name = model_name
        self._model = model if model is not None else self._load_model(model_name)
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        """Return the identifier of the model producing re-ranking scores."""

        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        """Score query-chunk pairs and return the most relevant candidates."""

        if top_k < 1:
            msg = "top_k must be greater than 0"
            raise ValueError(msg)
        if not query.strip():
            msg = "query must not be blank"
            raise ValueError(msg)
        if not candidates:
            return ()

        pairs = [(query, candidate.chunk.content) for candidate in candidates]
        raw_scores = _coerce_scores(self._model.predict(pairs, batch_size=self._batch_size))
        if len(raw_scores) != len(candidates):
            msg = "cross-encoder output must contain the same number of scores as candidates"
            raise ValueError(msg)

        ranked = sorted(
            zip(candidates, raw_scores, strict=True),
            key=lambda item: (-item[1], item[0].chunk.id),
        )
        return tuple(
            candidate.model_copy(
                update={
                    "query": query,
                    "score": _sigmoid(raw_score),
                    "rank": rank,
                    "retriever": "cross_encoder",
                    "explanation": f"cross_encoder_raw_score={raw_score}",
                }
            )
            for rank, (candidate, raw_score) in enumerate(ranked[:top_k], start=1)
        )

    def _load_model(self, model_name: str) -> Any:
        try:
            module = import_module("sentence_transformers")
        except ImportError as exc:
            msg = (
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with `pip install enterprise-rag-engine[embeddings]`."
            )
            raise RuntimeError(msg) from exc
        cross_encoder = module.__dict__["CrossEncoder"]
        return cross_encoder(model_name)


def _coerce_scores(raw_scores: Any) -> tuple[float, ...]:
    """Convert list- and numpy-like CrossEncoder outputs into scalar logits."""

    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, str | bytes):
        msg = "cross-encoder output must be a sequence of scalar scores"
        raise TypeError(msg)
    return tuple(float(score) for score in raw_scores)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1 + exponent)
