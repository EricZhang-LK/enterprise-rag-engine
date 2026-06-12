from dataclasses import dataclass
from hashlib import sha256

from enterprise_rag_engine.document_pipeline.splitters.recursive import RecursiveSplitter
from enterprise_rag_engine.document_pipeline.tokenization import TOKEN_PATTERN, TokenCounter
from enterprise_rag_engine.interfaces import BaseSplitter
from enterprise_rag_engine.models import (
    ChunkMetadata,
    ChunkRole,
    ChunkType,
    Document,
    DocumentChunk,
)

SENTENCE_BOUNDARIES = frozenset({".", "!", "?", "。", "！", "？", "\n"})


@dataclass(frozen=True)
class SemanticUnit:
    """A small text unit used for experimental topic-boundary detection."""

    content: str
    start: int
    end: int


class SemanticSplitter(BaseSplitter):
    """Experimental semantic splitter based on adjacent-unit lexical similarity.

    This splitter is deliberately lightweight: it does not download embedding models.
    It groups nearby sentences when their token sets overlap, then falls back to
    RecursiveSplitter whenever a semantic group is still too long.
    """

    def __init__(
        self,
        *,
        max_chars: int = 800,
        min_chars: int = 160,
        overlap_chars: int | None = None,
        similarity_threshold: float = 0.12,
    ) -> None:
        if max_chars < 1:
            msg = "max_chars must be greater than 0"
            raise ValueError(msg)
        if min_chars < 0:
            msg = "min_chars must be greater than or equal to 0"
            raise ValueError(msg)
        if min_chars >= max_chars:
            msg = "min_chars must be smaller than max_chars"
            raise ValueError(msg)
        actual_overlap_chars = (
            min(80, max(0, max_chars // 10)) if overlap_chars is None else overlap_chars
        )
        if actual_overlap_chars < 0:
            msg = "overlap_chars must be greater than or equal to 0"
            raise ValueError(msg)
        if actual_overlap_chars >= max_chars:
            msg = "overlap_chars must be smaller than max_chars"
            raise ValueError(msg)
        if not 0 <= similarity_threshold <= 1:
            msg = "similarity_threshold must be between 0 and 1"
            raise ValueError(msg)

        self.max_chars = max_chars
        self.min_chars = min_chars
        self.similarity_threshold = similarity_threshold
        self.token_counter = TokenCounter()
        self.fallback_splitter = RecursiveSplitter(
            max_chars=max_chars,
            overlap_chars=actual_overlap_chars,
        )

    def split(self, document: Document) -> tuple[DocumentChunk, ...]:
        if not document.content.strip():
            return ()

        units = _semantic_units(document.content)
        grouped_units = self._group_units(units)
        chunks: list[DocumentChunk] = []

        for unit in grouped_units:
            if len(unit.content) <= self.max_chars:
                chunks.append(
                    _build_chunk(
                        document=document,
                        unit=unit,
                        chunk_index=len(chunks),
                        splitter=self.__class__.__name__,
                        token_count=self.token_counter.count(unit.content),
                    )
                )
                continue
            chunks.extend(self._fallback_chunks(document=document, unit=unit))

        return tuple(chunks)

    def _group_units(self, units: tuple[SemanticUnit, ...]) -> tuple[SemanticUnit, ...]:
        grouped: list[SemanticUnit] = []
        current: SemanticUnit | None = None

        for unit in units:
            if current is None:
                current = unit
                continue

            candidate = _merge_units(current, unit)
            similarity = _lexical_similarity(current.content, unit.content)
            should_merge = (
                len(candidate.content) <= self.max_chars
                and (
                    len(current.content) < self.min_chars
                    or similarity >= self.similarity_threshold
                )
            )

            if should_merge:
                current = candidate
                continue

            grouped.append(current)
            current = unit

        if current is not None:
            grouped.append(current)

        return tuple(grouped)

    def _fallback_chunks(
        self,
        *,
        document: Document,
        unit: SemanticUnit,
    ) -> tuple[DocumentChunk, ...]:
        temp_document = Document(
            id=document.id,
            source_uri=document.source_uri,
            type=document.type,
            title=document.title,
            content=unit.content,
            metadata=document.metadata,
            created_at=document.created_at,
        )
        relative_chunks = self.fallback_splitter.split(temp_document)

        return tuple(
            _rebase_chunk(document=document, chunk=chunk, base_start=unit.start)
            for chunk in relative_chunks
        )


def _semantic_units(text: str) -> tuple[SemanticUnit, ...]:
    units: list[SemanticUnit] = []
    start = 0
    index = 0

    while index < len(text):
        if text.startswith("\n\n", index):
            units.extend(_trimmed_unit(text, start, index + 2))
            index += 2
            start = index
            continue

        if text[index] in SENTENCE_BOUNDARIES:
            units.extend(_trimmed_unit(text, start, index + 1))
            start = index + 1
        index += 1

    units.extend(_trimmed_unit(text, start, len(text)))
    return tuple(units)


def _trimmed_unit(text: str, start: int, end: int) -> tuple[SemanticUnit, ...]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return ()
    return (SemanticUnit(content=text[start:end], start=start, end=end),)


def _merge_units(left: SemanticUnit, right: SemanticUnit) -> SemanticUnit:
    content = f"{left.content}\n{right.content}"
    return SemanticUnit(content=content, start=left.start, end=right.end)


def _lexical_similarity(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _terms(text: str) -> set[str]:
    return {match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)}


def _build_chunk(
    *,
    document: Document,
    unit: SemanticUnit,
    chunk_index: int,
    splitter: str,
    token_count: int,
) -> DocumentChunk:
    metadata = ChunkMetadata(
        source_uri=document.source_uri,
        document_id=document.id,
        content_hash=_content_hash(unit.content),
        chunk_index=chunk_index,
        chunk_role=ChunkRole.STANDALONE,
        splitter=splitter,
        token_count=token_count,
        start_char=unit.start,
        end_char=unit.end,
    )
    return DocumentChunk(
        document_id=document.id,
        content=unit.content,
        chunk_type=ChunkType.TEXT,
        metadata=metadata,
        start_char=unit.start,
        end_char=unit.end,
    )


def _rebase_chunk(*, document: Document, chunk: DocumentChunk, base_start: int) -> DocumentChunk:
    start_char = None if chunk.start_char is None else base_start + chunk.start_char
    end_char = None if chunk.end_char is None else base_start + chunk.end_char
    metadata = ChunkMetadata(
        source_uri=document.source_uri,
        document_id=document.id,
        content_hash=_content_hash(chunk.content),
        chunk_index=chunk.metadata.chunk_index,
        chunk_count=chunk.metadata.chunk_count,
        chunk_role=ChunkRole.STANDALONE,
        splitter="SemanticSplitter",
        token_count=chunk.metadata.token_count,
        start_char=start_char,
        end_char=end_char,
    )
    return DocumentChunk(
        document_id=document.id,
        content=chunk.content,
        chunk_type=ChunkType.TEXT,
        metadata=metadata,
        start_char=start_char,
        end_char=end_char,
    )


def _content_hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()
