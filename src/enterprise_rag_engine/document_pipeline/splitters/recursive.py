from dataclasses import dataclass
from hashlib import sha256

from enterprise_rag_engine.document_pipeline.tokenization import TokenCounter
from enterprise_rag_engine.interfaces import BaseSplitter
from enterprise_rag_engine.models import (
    ChunkMetadata,
    ChunkRole,
    ChunkType,
    Document,
    DocumentChunk,
)

DEFAULT_SEPARATORS = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    ". ",
    "! ",
    "? ",
    "；",
    "; ",
    "，",
    ", ",
    " ",
)


@dataclass(frozen=True)
class TextSpan:
    """A slice of document text with source offsets kept for citation and debugging."""

    start: int
    end: int


class RecursiveSplitter(BaseSplitter):
    """Split a document with progressively weaker text boundaries.

    The splitter first tries high-level boundaries such as paragraphs, then falls back
    to lines, sentences, words, and finally fixed-width character windows. This keeps
    chunks readable while guaranteeing that very long text can still be indexed.
    """

    def __init__(
        self,
        *,
        max_chars: int = 800,
        overlap_chars: int = 120,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ) -> None:
        if max_chars < 1:
            msg = "max_chars must be greater than 0"
            raise ValueError(msg)
        if overlap_chars < 0:
            msg = "overlap_chars must be greater than or equal to 0"
            raise ValueError(msg)
        if overlap_chars >= max_chars:
            msg = "overlap_chars must be smaller than max_chars"
            raise ValueError(msg)
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.separators = separators
        self.token_counter = TokenCounter()

    def split(self, document: Document) -> tuple[DocumentChunk, ...]:
        text = document.content
        if not text.strip():
            return ()

        leaf_spans = self._split_span(text, TextSpan(start=0, end=len(text)), separator_index=0)
        chunk_spans = self._merge_spans(text, leaf_spans)

        return tuple(
            self._build_chunk(
                document=document,
                text=text,
                span=span,
                chunk_index=index,
                chunk_count=len(chunk_spans),
            )
            for index, span in enumerate(chunk_spans)
            if text[span.start : span.end].strip()
        )

    def _split_span(
        self,
        text: str,
        span: TextSpan,
        *,
        separator_index: int,
    ) -> tuple[TextSpan, ...]:
        if span.end - span.start <= self.max_chars:
            return (span,)

        if separator_index >= len(self.separators):
            return tuple(
                TextSpan(start=start, end=min(start + self.max_chars, span.end))
                for start in range(span.start, span.end, self.max_chars)
            )

        separator = self.separators[separator_index]
        child_spans = _split_span_by_separator(text, span, separator)
        if len(child_spans) == 1:
            return self._split_span(text, span, separator_index=separator_index + 1)

        result: list[TextSpan] = []
        for child in child_spans:
            if child.end > child.start:
                result.extend(
                    self._split_span(text, child, separator_index=separator_index + 1)
                )
        return tuple(result)

    def _merge_spans(self, text: str, spans: tuple[TextSpan, ...]) -> tuple[TextSpan, ...]:
        merged: list[TextSpan] = []
        current_start: int | None = None
        current_end: int | None = None

        for span in spans:
            if current_start is None or current_end is None:
                current_start = span.start
                current_end = span.end
                continue

            candidate_end = span.end
            if candidate_end - current_start <= self.max_chars:
                current_end = candidate_end
                continue

            merged.append(_trim_span(text, TextSpan(current_start, current_end)))
            current_start = max(current_end - self.overlap_chars, candidate_end - self.max_chars, 0)
            current_end = candidate_end

        if current_start is not None and current_end is not None:
            merged.append(_trim_span(text, TextSpan(current_start, current_end)))

        return tuple(span for span in merged if span.end > span.start)

    def _build_chunk(
        self,
        *,
        document: Document,
        text: str,
        span: TextSpan,
        chunk_index: int,
        chunk_count: int,
    ) -> DocumentChunk:
        content = text[span.start : span.end]
        metadata = ChunkMetadata(
            source_uri=document.source_uri,
            document_id=document.id,
            content_hash=_content_hash(content),
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            chunk_role=ChunkRole.STANDALONE,
            splitter=self.__class__.__name__,
            token_count=self.token_counter.count(content),
            start_char=span.start,
            end_char=span.end,
        )
        return DocumentChunk(
            document_id=document.id,
            content=content,
            chunk_type=ChunkType.TEXT,
            metadata=metadata,
            start_char=span.start,
            end_char=span.end,
        )


def _split_span_by_separator(text: str, span: TextSpan, separator: str) -> tuple[TextSpan, ...]:
    spans: list[TextSpan] = []
    start = span.start

    while start < span.end:
        index = text.find(separator, start, span.end)
        if index == -1:
            spans.append(TextSpan(start=start, end=span.end))
            break

        boundary = index + len(separator)
        spans.append(TextSpan(start=start, end=boundary))
        start = boundary

    return tuple(spans)


def _trim_span(text: str, span: TextSpan) -> TextSpan:
    start = span.start
    end = span.end

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    return TextSpan(start=start, end=end)


def _content_hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()
