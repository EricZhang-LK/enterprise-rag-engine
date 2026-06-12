from collections.abc import Callable
from hashlib import sha256
from time import perf_counter
from typing import Any

from docx import Document as DocxDocument

from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentType,
    ParseResult,
    ParseStatus,
)

DocxReaderFactory = Callable[[str], Any]


class DocxParser(BaseParser):
    """Parse Word documents and preserve basic heading paths."""

    def __init__(self, reader_factory: DocxReaderFactory | None = None) -> None:
        self._reader_factory = reader_factory or DocxDocument

    def parse(self, source_uri: str) -> ParseResult:
        started_at = perf_counter()
        try:
            raw_document = self._reader_factory(source_uri)
        except Exception as exc:
            document = Document(source_uri=source_uri, type=DocumentType.DOCX, content="")
            return ParseResult(
                document=document,
                status=ParseStatus.FAILED,
                errors=(f"Failed to read Docx: {exc}",),
                elapsed_ms=_elapsed_ms(started_at),
            )

        paragraphs = _extract_paragraphs(raw_document)
        content = "\n\n".join(paragraph.text for paragraph in paragraphs if paragraph.text)
        title = _first_heading(paragraphs)
        document = Document(
            source_uri=source_uri,
            type=DocumentType.DOCX,
            title=title,
            content=content,
            metadata={
                "parser": self.__class__.__name__,
                "paragraph_count": len(paragraphs),
            },
        )
        chunks = tuple(
            _paragraph_chunks(source_uri=source_uri, document=document, paragraphs=paragraphs)
        )
        return ParseResult(
            document=document,
            chunks=chunks,
            status=ParseStatus.SUCCEEDED if content else ParseStatus.FAILED,
            errors=() if content else ("No extractable text found in Docx.",),
            elapsed_ms=_elapsed_ms(started_at),
        )


class DocxParagraph:
    def __init__(self, text: str, style_name: str | None) -> None:
        self.text = text.strip()
        self.style_name = style_name


def _extract_paragraphs(raw_document: Any) -> list[DocxParagraph]:
    paragraphs: list[DocxParagraph] = []
    for paragraph in getattr(raw_document, "paragraphs", ()):
        text = str(getattr(paragraph, "text", "")).strip()
        style = getattr(paragraph, "style", None)
        style_name = getattr(style, "name", None)
        if text:
            paragraphs.append(DocxParagraph(text=text, style_name=style_name))
    return paragraphs


def _paragraph_chunks(
    *,
    source_uri: str,
    document: Document,
    paragraphs: list[DocxParagraph],
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    heading_stack: list[str] = []

    for chunk_index, paragraph in enumerate(paragraphs):
        heading = _heading_level(paragraph.style_name)
        if heading is not None:
            heading_stack = heading_stack[: heading - 1]
            heading_stack.append(paragraph.text)
        section_path = tuple(heading_stack)
        chunks.append(
            DocumentChunk(
                document_id=document.id,
                content=paragraph.text,
                metadata=ChunkMetadata(
                    source_uri=source_uri,
                    document_id=document.id,
                    section_path=section_path,
                    content_hash=sha256(paragraph.text.encode("utf-8")).hexdigest(),
                    chunk_index=chunk_index,
                    chunk_count=len(paragraphs),
                    splitter="DocxParser",
                    start_char=0,
                    end_char=len(paragraph.text),
                ),
                start_char=0,
                end_char=len(paragraph.text),
            )
        )

    return chunks


def _heading_level(style_name: str | None) -> int | None:
    if style_name is None:
        return None
    if not style_name.startswith("Heading "):
        return None
    suffix = style_name.removeprefix("Heading ")
    if not suffix.isdigit():
        return None
    level = int(suffix)
    if level < 1:
        return None
    return level


def _first_heading(paragraphs: list[DocxParagraph]) -> str | None:
    for paragraph in paragraphs:
        if _heading_level(paragraph.style_name) == 1:
            return paragraph.text
    return None


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
