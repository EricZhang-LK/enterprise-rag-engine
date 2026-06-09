from collections.abc import Callable
from hashlib import sha256
from time import perf_counter
from typing import Any

from pypdf import PdfReader

from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentType,
    ParseResult,
    ParseStatus,
)

ReaderFactory = Callable[[str], Any]


class PdfTextParser(BaseParser):
    """Extract text from PDF pages while preserving page-level metadata."""

    def __init__(self, reader_factory: ReaderFactory | None = None) -> None:
        self._reader_factory = reader_factory or PdfReader

    def parse(self, source_uri: str) -> ParseResult:
        started_at = perf_counter()
        errors: list[str] = []

        try:
            reader = self._reader_factory(source_uri)
        except Exception as exc:
            document = Document(source_uri=source_uri, type=DocumentType.PDF, content="")
            return ParseResult(
                document=document,
                status=ParseStatus.FAILED,
                errors=(f"Failed to read PDF: {exc}",),
                elapsed_ms=_elapsed_ms(started_at),
            )

        page_texts = _extract_page_texts(reader, errors)
        content = "\n\n".join(text for _, text in page_texts if text)
        metadata = _reader_metadata(reader)
        metadata["page_count"] = len(getattr(reader, "pages", ()))
        metadata["parser"] = self.__class__.__name__

        if not content and not errors:
            errors.append("No extractable text found. The PDF may require OCR.")

        document = Document(
            source_uri=source_uri,
            type=DocumentType.PDF,
            title=_metadata_title(metadata),
            content=content,
            metadata=metadata,
        )
        chunks = tuple(
            _page_chunk(source_uri=source_uri, document_id=document.id, page_number=page, text=text)
            for page, text in page_texts
            if text
        )

        return ParseResult(
            document=document,
            chunks=chunks,
            status=_status_for(errors=errors, content=content),
            errors=tuple(errors),
            elapsed_ms=_elapsed_ms(started_at),
        )


def _extract_page_texts(reader: Any, errors: list[str]) -> list[tuple[int, str]]:
    page_texts: list[tuple[int, str]] = []
    for index, page in enumerate(getattr(reader, "pages", ()), start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            errors.append(f"Failed to extract text from page {index}: {exc}")
            continue
        page_texts.append((index, _normalize_text(raw_text)))
    return page_texts


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _page_chunk(
    *,
    source_uri: str,
    document_id: str,
    page_number: int,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        document_id=document_id,
        content=text,
        metadata=ChunkMetadata(
            source_uri=source_uri,
            document_id=document_id,
            page_number=page_number,
            content_hash=sha256(text.encode("utf-8")).hexdigest(),
        ),
    )


def _reader_metadata(reader: Any) -> dict[str, Any]:
    raw_metadata = getattr(reader, "metadata", None)
    if raw_metadata is None:
        return {}

    metadata: dict[str, Any] = {}
    for key in ("title", "author", "creator", "producer", "subject"):
        value = getattr(raw_metadata, key, None)
        if value:
            metadata[key] = str(value)
    return metadata


def _metadata_title(metadata: dict[str, Any]) -> str | None:
    title = metadata.get("title")
    if title is None:
        return None
    return str(title)


def _status_for(*, errors: list[str], content: str) -> ParseStatus:
    if not errors:
        return ParseStatus.SUCCEEDED
    if content:
        return ParseStatus.PARTIAL
    return ParseStatus.FAILED


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
