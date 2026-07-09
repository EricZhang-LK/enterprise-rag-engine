from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from enterprise_rag_engine.interfaces import BaseRetriever
from enterprise_rag_engine.models import DocumentChunk, RetrievalResult

Tokenizer = Callable[[str], tuple[str, ...]]

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    chunk: DocumentChunk
    term_counts: Counter[str]
    length: int


class BM25Retriever(BaseRetriever):
    """In-memory BM25 retriever for lexical chunk recall."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if k1 <= 0:
            msg = "k1 must be greater than 0"
            raise ValueError(msg)
        if not 0 <= b <= 1:
            msg = "b must be in the range [0, 1]"
            raise ValueError(msg)
        self._k1 = k1
        self._b = b
        self._tokenizer = tokenizer or default_bm25_tokenizer
        self._indexed_chunks = tuple(self._index_chunk(chunk) for chunk in chunks)
        self._average_length = _average_length(item.length for item in self._indexed_chunks)
        self._document_frequency = self._build_document_frequency()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Return chunks ranked by BM25 lexical relevance."""

        if top_k < 1:
            msg = "top_k must be greater than 0"
            raise ValueError(msg)
        query_terms = self._tokenizer(query)
        if not query_terms:
            msg = "query must contain at least one searchable term"
            raise ValueError(msg)

        scored = [
            (self._score(query_terms, indexed), indexed)
            for indexed in self._indexed_chunks
            if _matches_filters(indexed.chunk, filters)
        ]
        ranked = sorted(
            ((score, indexed) for score, indexed in scored if score > 0),
            key=lambda item: (-item[0], item[1].chunk.id),
        )
        return tuple(
            RetrievalResult(
                query=query,
                chunk=indexed.chunk,
                score=round(score, 6),
                rank=rank,
                retriever="bm25",
                explanation=_explain_matches(query_terms, indexed.term_counts),
            )
            for rank, (score, indexed) in enumerate(ranked[:top_k], start=1)
        )

    def _index_chunk(self, chunk: DocumentChunk) -> _IndexedChunk:
        terms = self._tokenizer(chunk.content)
        return _IndexedChunk(chunk=chunk, term_counts=Counter(terms), length=len(terms))

    def _build_document_frequency(self) -> dict[str, int]:
        document_frequency: dict[str, int] = {}
        for indexed in self._indexed_chunks:
            for term in indexed.term_counts:
                document_frequency[term] = document_frequency.get(term, 0) + 1
        return document_frequency

    def _score(self, query_terms: Sequence[str], indexed: _IndexedChunk) -> float:
        if indexed.length == 0:
            return 0.0

        score = 0.0
        for term in set(query_terms):
            term_frequency = indexed.term_counts.get(term, 0)
            if term_frequency == 0:
                continue
            idf = self._inverse_document_frequency(term)
            denominator = term_frequency + self._k1 * (
                1 - self._b + self._b * indexed.length / self._average_length
            )
            score += idf * term_frequency * (self._k1 + 1) / denominator
        return score

    def _inverse_document_frequency(self, term: str) -> float:
        document_count = len(self._indexed_chunks)
        containing_count = self._document_frequency.get(term, 0)
        return math.log(1 + (document_count - containing_count + 0.5) / (containing_count + 0.5))


def default_bm25_tokenizer(text: str) -> tuple[str, ...]:
    """Tokenize English terms and CJK characters for deterministic lexical retrieval."""

    return tuple(match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))


def _average_length(lengths: Iterable[int]) -> float:
    values = tuple(lengths)
    if not values:
        return 1.0
    return max(sum(values) / len(values), 1.0)


def _matches_filters(chunk: DocumentChunk, filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    payload = chunk.metadata_payload()
    return all(str(payload.get(key)) == value for key, value in filters.items())


def _explain_matches(query_terms: Sequence[str], term_counts: Counter[str]) -> str:
    matched_terms = sorted({term for term in query_terms if term_counts.get(term, 0) > 0})
    return "matched_terms=" + ",".join(matched_terms)
