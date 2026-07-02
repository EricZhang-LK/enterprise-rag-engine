from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

Embedding = tuple[float, ...]
FilterValue = str | int | float | bool


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


class ParseProgressStage(StrEnum):
    QUEUED = "queued"
    STARTED = "started"
    CACHE_HIT = "cache_hit"
    SUCCEEDED = "succeeded"
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

    def metadata_payload(self) -> dict[str, Any]:
        """Return filterable metadata fields shared by vector-store adapters."""

        return {
            "chunk_id": self.id,
            "document_id": self.document_id,
            "chunk_type": self.chunk_type.value,
            "parent_id": self.parent_id,
            "source_uri": self.metadata.source_uri,
            "page_number": self.metadata.page_number,
            "end_page_number": self.metadata.end_page_number,
            "section_path": list(self.metadata.section_path),
            "tenant_id": self.metadata.tenant_id,
            "content_hash": self.metadata.content_hash,
            "chunk_index": self.metadata.chunk_index,
            "chunk_count": self.metadata.chunk_count,
            "chunk_role": self.metadata.chunk_role.value,
            "splitter": self.metadata.splitter,
            "token_count": self.metadata.token_count,
            "start_char": self.metadata.start_char,
            "end_char": self.metadata.end_char,
            "has_table": self.metadata.has_table,
            "metadata": self.metadata.metadata,
        }


class VectorStoreFilter(BaseModel):
    model_config = ConfigDict(frozen=True)

    conditions: tuple["VectorFilterCondition", ...] = Field(default_factory=tuple)

    @classmethod
    def empty(cls) -> "VectorStoreFilter":
        return cls()

    @classmethod
    def exact(cls, conditions: dict[str, FilterValue]) -> "VectorStoreFilter":
        return cls(
            conditions=tuple(
                VectorFilterCondition(key=key, value=value) for key, value in conditions.items()
            )
        )

    @classmethod
    def equals(cls, key: str, value: FilterValue) -> "VectorStoreFilter":
        return cls(conditions=(VectorFilterCondition(key=key, value=value),))

    @classmethod
    def metadata(
        cls,
        *,
        tenant_id: str | None = None,
        document_id: str | None = None,
        page_number: int | None = None,
        chunk_type: ChunkType | str | None = None,
    ) -> "VectorStoreFilter":
        conditions: list[VectorFilterCondition] = []
        if tenant_id is not None:
            conditions.append(VectorFilterCondition(key="tenant_id", value=tenant_id))
        if document_id is not None:
            conditions.append(VectorFilterCondition(key="document_id", value=document_id))
        if page_number is not None:
            conditions.extend(
                (
                    VectorFilterCondition(
                        key="page_number",
                        operator=VectorFilterOperator.LTE,
                        value=page_number,
                    ),
                    VectorFilterCondition(
                        key="end_page_number",
                        operator=VectorFilterOperator.GTE,
                        value=page_number,
                    ),
                )
            )
        if chunk_type is not None:
            chunk_type_value = chunk_type.value if isinstance(chunk_type, ChunkType) else chunk_type
            conditions.append(VectorFilterCondition(key="chunk_type", value=chunk_type_value))
        return cls(conditions=tuple(conditions))

    @property
    def is_empty(self) -> bool:
        return not self.conditions

    def matches_payload(self, payload: dict[str, Any]) -> bool:
        return all(condition.matches(payload) for condition in self.conditions)


class VectorFilterOperator(StrEnum):
    EQ = "eq"
    LTE = "lte"
    GTE = "gte"


class VectorFilterCondition(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    value: FilterValue
    operator: VectorFilterOperator = VectorFilterOperator.EQ

    def matches(self, payload: dict[str, Any]) -> bool:
        payload_value = _payload_value(payload, self.key)
        if self.operator is VectorFilterOperator.EQ:
            return bool(payload_value == self.value)
        if payload_value is None:
            return False
        if not isinstance(payload_value, int | float) or not isinstance(self.value, int | float):
            return False
        if self.operator is VectorFilterOperator.LTE:
            return payload_value <= self.value
        if self.operator is VectorFilterOperator.GTE:
            return payload_value >= self.value


def _payload_value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if key == "end_page_number" and value is None:
        return payload.get("page_number")
    return value


class VectorStoreRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    embedding: Embedding

    @model_validator(mode="after")
    def validate_embedding(self) -> "VectorStoreRecord":
        if not self.embedding:
            msg = "embedding must not be empty"
            raise ValueError(msg)
        return self


class VectorSearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_embedding: Embedding
    top_k: int = Field(gt=0)
    filters: VectorStoreFilter = Field(default_factory=VectorStoreFilter.empty)
    query_text: str | None = None

    @model_validator(mode="after")
    def validate_query_embedding(self) -> "VectorSearchRequest":
        if not self.query_embedding:
            msg = "query_embedding must not be empty"
            raise ValueError(msg)
        return self


class VectorStoreWriteResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    affected_count: int = Field(ge=0)


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


class ParseProgressEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    source_uri: str
    stage: ParseProgressStage
    progress: float = Field(ge=0, le=1)
    status: ParseStatus | None = None
    message: str | None = None
    elapsed_ms: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    chunk: DocumentChunk
    score: float = Field(ge=0)
    rank: int = Field(ge=1)
    retriever: str
    explanation: str | None = None
