from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    UNKNOWN = "unknown"


class ChunkType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"


class ChunkRole(StrEnum):
    STANDALONE = "standalone"
    PARENT = "parent"
    CHILD = "child"


class ParseStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class OcrStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_uri: str
    type: DocumentType = DocumentType.UNKNOWN
    title: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def character_count(self) -> int:
        return len(self.content)


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_uri: str
    document_id: str
    page_number: int | None = Field(default=None, ge=1)
    end_page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = Field(default_factory=tuple)
    tenant_id: str | None = None
    content_hash: str | None = None
    chunk_index: int | None = Field(default=None, ge=0)
    chunk_count: int | None = Field(default=None, ge=1)
    chunk_role: ChunkRole = ChunkRole.STANDALONE
    splitter: str | None = None
    token_count: int | None = Field(default=None, ge=0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    has_table: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ranges(self) -> "ChunkMetadata":
        if (
            self.page_number is not None
            and self.end_page_number is not None
            and self.end_page_number < self.page_number
        ):
            msg = "end_page_number must be greater than or equal to page_number"
            raise ValueError(msg)
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            msg = "end_char must be greater than or equal to start_char"
            raise ValueError(msg)
        return self


class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    chunk_type: ChunkType = ChunkType.TEXT
    metadata: ChunkMetadata
    parent_id: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @property
    def character_count(self) -> int:
        return len(self.content)


class TableBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: tuple[tuple[str, ...], ...]
    caption: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        if not self.rows:
            return 0
        return max(len(row) for row in self.rows)

    @property
    def is_rectangular(self) -> bool:
        if not self.rows:
            return True
        return all(len(row) == self.column_count for row in self.rows)


class OCRResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    status: OcrStatus
    page_number: int | None = Field(default=None, ge=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    errors: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: Document
    chunks: tuple[DocumentChunk, ...] = Field(default_factory=tuple)
    status: ParseStatus
    errors: tuple[str, ...] = Field(default_factory=tuple)
    elapsed_ms: float = Field(ge=0)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    chunk: DocumentChunk
    score: float = Field(ge=0)
    rank: int = Field(ge=1)
    retriever: str
    explanation: str | None = None
