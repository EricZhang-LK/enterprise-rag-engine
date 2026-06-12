from dataclasses import dataclass
from statistics import mean

from enterprise_rag_engine.models import DocumentChunk


@dataclass(frozen=True)
class ChunkQualityMetrics:
    """Aggregate quality metrics for one chunking strategy."""

    chunk_count: int
    average_chars: float
    p95_chars: int
    average_tokens: float
    p95_tokens: int
    overlap_ratio: float
    context_complete_ratio: float

    def as_dict(self) -> dict[str, float]:
        """Return a flat metric dictionary for scripts and reports."""

        return {
            "chunk_count": float(self.chunk_count),
            "average_chars": self.average_chars,
            "p95_chars": float(self.p95_chars),
            "average_tokens": self.average_tokens,
            "p95_tokens": float(self.p95_tokens),
            "overlap_ratio": self.overlap_ratio,
            "context_complete_ratio": self.context_complete_ratio,
        }


def evaluate_chunks(chunks: tuple[DocumentChunk, ...]) -> ChunkQualityMetrics:
    """Evaluate length, overlap, and metadata completeness for chunks."""

    if not chunks:
        return ChunkQualityMetrics(
            chunk_count=0,
            average_chars=0.0,
            p95_chars=0,
            average_tokens=0.0,
            p95_tokens=0,
            overlap_ratio=0.0,
            context_complete_ratio=0.0,
        )

    char_lengths = tuple(chunk.character_count for chunk in chunks)
    token_lengths = tuple(chunk.metadata.token_count or 0 for chunk in chunks)
    return ChunkQualityMetrics(
        chunk_count=len(chunks),
        average_chars=round(mean(char_lengths), 3),
        p95_chars=_percentile_ceil(char_lengths, percentile=0.95),
        average_tokens=round(mean(token_lengths), 3),
        p95_tokens=_percentile_ceil(token_lengths, percentile=0.95),
        overlap_ratio=_overlap_ratio(chunks),
        context_complete_ratio=_context_complete_ratio(chunks),
    )


def _percentile_ceil(values: tuple[int, ...], *, percentile: float) -> int:
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * percentile + 0.999) - 1))
    return sorted_values[index]


def _overlap_ratio(chunks: tuple[DocumentChunk, ...]) -> float:
    total_chars = sum(chunk.character_count for chunk in chunks)
    if total_chars == 0:
        return 0.0

    overlap_chars = 0
    comparable_chunks = sorted(
        (chunk for chunk in chunks if chunk.start_char is not None and chunk.end_char is not None),
        key=lambda chunk: (chunk.document_id, chunk.start_char or 0, chunk.end_char or 0),
    )

    previous_by_document: dict[str, DocumentChunk] = {}
    for chunk in comparable_chunks:
        previous = previous_by_document.get(chunk.document_id)
        if previous is not None and previous.end_char is not None and chunk.start_char is not None:
            overlap_chars += max(0, previous.end_char - chunk.start_char)
        previous_by_document[chunk.document_id] = chunk

    return round(overlap_chars / total_chars, 6)


def _context_complete_ratio(chunks: tuple[DocumentChunk, ...]) -> float:
    complete_count = sum(1 for chunk in chunks if _has_required_context(chunk))
    return round(complete_count / len(chunks), 6)


def _has_required_context(chunk: DocumentChunk) -> bool:
    metadata = chunk.metadata
    return all(
        (
            bool(metadata.source_uri),
            bool(metadata.document_id),
            bool(metadata.content_hash),
            metadata.start_char is not None,
            metadata.end_char is not None,
            metadata.token_count is not None,
            metadata.splitter is not None,
        )
    )
