from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Protocol

from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentType,
    ParseResult,
    ParseStatus,
)


class TextReader(Protocol):
    def __call__(self, source_uri: str) -> str: ...


class MarkdownParser(BaseParser):
    """Parse Markdown text and preserve heading paths for each section."""

    def __init__(self, text_reader: TextReader | None = None) -> None:
        self._text_reader = text_reader or _read_markdown_file

    def parse(self, source_uri: str) -> ParseResult:
        started_at = perf_counter()
        try:
            content = self._text_reader(source_uri)
        except OSError as exc:
            document = Document(source_uri=source_uri, type=DocumentType.MARKDOWN, content="")
            return ParseResult(
                document=document,
                status=ParseStatus.FAILED,
                errors=(f"Failed to read Markdown: {exc}",),
                elapsed_ms=_elapsed_ms(started_at),
            )

        title = _first_heading(content)
        document = Document(
            source_uri=source_uri,
            type=DocumentType.MARKDOWN,
            title=title,
            content=content,
            metadata={"parser": self.__class__.__name__},
        )
        chunks = tuple(_section_chunks(source_uri=source_uri, document=document))
        return ParseResult(
            document=document,
            chunks=chunks,
            status=ParseStatus.SUCCEEDED,
            elapsed_ms=_elapsed_ms(started_at),
        )


def _section_chunks(*, source_uri: str, document: Document) -> list[DocumentChunk]:
    sections: list[DocumentChunk] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_path: tuple[str, ...] = ()

    def flush() -> None:
        text = "\n".join(line for line in current_lines).strip()
        if not text:
            return
        sections.append(
            DocumentChunk(
                document_id=document.id,
                content=text,
                metadata=ChunkMetadata(
                    source_uri=source_uri,
                    document_id=document.id,
                    section_path=current_path,
                    content_hash=sha256(text.encode("utf-8")).hexdigest(),
                ),
            )
        )

    for line in document.content.splitlines():
        heading = _parse_heading(line)
        if heading is not None:
            flush()
            current_lines = [line]
            level, title = heading
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_path = tuple(heading_stack)
            continue
        current_lines.append(line)

    flush()
    return sections


def _read_markdown_file(source_uri: str) -> str:
    return Path(source_uri).read_text(encoding="utf-8")


def _parse_heading(line: str) -> tuple[int, str] | None:
    stripped = line.lstrip()
    prefix_length = len(stripped) - len(stripped.lstrip("#"))
    if prefix_length == 0 or prefix_length > 6:
        return None
    if len(stripped) <= prefix_length or stripped[prefix_length] != " ":
        return None
    title = stripped[prefix_length:].strip()
    if not title:
        return None
    return prefix_length, title


def _first_heading(content: str) -> str | None:
    for line in content.splitlines():
        heading = _parse_heading(line)
        if heading is not None:
            return heading[1]
    return None


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
