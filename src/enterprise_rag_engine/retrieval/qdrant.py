from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any, Protocol, cast
from uuid import NAMESPACE_URL, uuid5

from enterprise_rag_engine.interfaces import BaseVectorStore
from enterprise_rag_engine.models import (
    ChunkMetadata,
    ChunkRole,
    ChunkType,
    DocumentChunk,
    Embedding,
    RetrievalResult,
    VectorFilterOperator,
    VectorSearchRequest,
    VectorStoreFilter,
    VectorStoreRecord,
    VectorStoreWriteResult,
)


class QdrantPoint(Protocol):
    payload: dict[str, Any]
    score: float


class QdrantClientProtocol(Protocol):
    def upsert(self, *, collection_name: str, points: list[dict[str, Any]]) -> None: ...

    def query_points(
        self,
        *,
        collection_name: str,
        query: Embedding,
        query_filter: dict[str, Any] | None,
        limit: int,
        with_payload: bool,
    ) -> Sequence[Any]: ...

    def delete(self, *, collection_name: str, points_selector: dict[str, Any]) -> None: ...


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-backed vector store for retrieval-ready document chunks.

    The adapter keeps Qdrant payloads explicit instead of storing opaque Pydantic
    JSON. That makes metadata filtering, debugging, and future migrations easier.
    """

    def __init__(
        self,
        *,
        collection_name: str,
        vector_size: int,
        client: QdrantClientProtocol | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if vector_size < 1:
            msg = "vector_size must be greater than 0"
            raise ValueError(msg)
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._client = client or _create_qdrant_client(url=url, api_key=api_key)

    def upsert(self, records: Sequence[VectorStoreRecord]) -> VectorStoreWriteResult:
        """Insert or update chunk vectors using stable point ids."""

        if not records:
            return VectorStoreWriteResult(affected_count=0)

        points: list[dict[str, Any]] = []
        for record in records:
            self._validate_embedding(record.embedding)
            points.append(
                {
                    "id": _point_id(record.chunk.id),
                    "vector": record.embedding,
                    "payload": _chunk_payload(record.chunk),
                }
            )
        self._client.upsert(collection_name=self._collection_name, points=points)
        return VectorStoreWriteResult(affected_count=len(points))

    def search(self, request: VectorSearchRequest) -> tuple[RetrievalResult, ...]:
        """Search Qdrant and restore scored points into domain retrieval results."""

        self._validate_embedding(request.query_embedding)

        points = cast(
            Sequence[QdrantPoint],
            self._client.query_points(
                collection_name=self._collection_name,
                query=request.query_embedding,
                query_filter=_metadata_filter(request.filters),
                limit=request.top_k,
                with_payload=True,
            ),
        )
        return tuple(
            RetrievalResult(
                query=request.query_text or "",
                chunk=_chunk_from_payload(_require_payload(point.payload)),
                score=point.score,
                rank=rank,
                retriever="qdrant",
            )
            for rank, point in enumerate(points, start=1)
        )

    def delete(self, filters: VectorStoreFilter) -> VectorStoreWriteResult:
        """Delete all chunk vectors matching the given filter."""

        self._client.delete(
            collection_name=self._collection_name,
            points_selector={"filter": _metadata_filter(filters)},
        )
        return VectorStoreWriteResult(affected_count=0)

    def _validate_embedding(self, embedding: Embedding) -> None:
        if len(embedding) != self._vector_size:
            msg = f"embedding dimension must be {self._vector_size}"
            raise ValueError(msg)


def _create_qdrant_client(*, url: str | None, api_key: str | None) -> QdrantClientProtocol:
    try:
        qdrant_client_module = import_module("qdrant_client")
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        msg = "Install qdrant-client or pass a Qdrant-compatible client instance."
        raise RuntimeError(msg) from exc
    qdrant_client: Any = qdrant_client_module.QdrantClient
    return cast(QdrantClientProtocol, qdrant_client(url=url, api_key=api_key))


def _point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))


def _chunk_payload(chunk: DocumentChunk) -> dict[str, Any]:
    metadata = chunk.metadata
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "chunk_type": chunk.chunk_type.value,
        "parent_id": chunk.parent_id,
        "source_uri": metadata.source_uri,
        "page_number": metadata.page_number,
        "end_page_number": metadata.end_page_number,
        "section_path": list(metadata.section_path),
        "tenant_id": metadata.tenant_id,
        "content_hash": metadata.content_hash,
        "chunk_index": metadata.chunk_index,
        "chunk_count": metadata.chunk_count,
        "chunk_role": metadata.chunk_role.value,
        "splitter": metadata.splitter,
        "token_count": metadata.token_count,
        "start_char": metadata.start_char,
        "end_char": metadata.end_char,
        "has_table": metadata.has_table,
        "metadata": metadata.metadata,
    }


def _chunk_from_payload(payload: dict[str, Any]) -> DocumentChunk:
    metadata = ChunkMetadata(
        source_uri=str(payload["source_uri"]),
        document_id=str(payload["document_id"]),
        page_number=_optional_int(payload.get("page_number")),
        end_page_number=_optional_int(payload.get("end_page_number")),
        section_path=tuple(str(item) for item in payload.get("section_path", [])),
        tenant_id=_optional_str(payload.get("tenant_id")),
        content_hash=_optional_str(payload.get("content_hash")),
        chunk_index=_optional_int(payload.get("chunk_index")),
        chunk_count=_optional_int(payload.get("chunk_count")),
        chunk_role=ChunkRole(str(payload.get("chunk_role", ChunkRole.STANDALONE.value))),
        splitter=_optional_str(payload.get("splitter")),
        token_count=_optional_int(payload.get("token_count")),
        start_char=_optional_int(payload.get("start_char")),
        end_char=_optional_int(payload.get("end_char")),
        has_table=bool(payload.get("has_table", False)),
        metadata=cast(dict[str, Any], payload.get("metadata", {})),
    )
    return DocumentChunk(
        id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        content=str(payload["content"]),
        chunk_type=ChunkType(str(payload.get("chunk_type", ChunkType.TEXT.value))),
        metadata=metadata,
        parent_id=_optional_str(payload.get("parent_id")),
        start_char=metadata.start_char,
        end_char=metadata.end_char,
    )


def _metadata_filter(filters: VectorStoreFilter) -> dict[str, Any] | None:
    if filters.is_empty:
        return None
    return {
        "must": [
            _qdrant_condition(condition.key, condition.operator, condition.value)
            for condition in filters.conditions
        ]
    }


def _qdrant_condition(
    key: str,
    operator: VectorFilterOperator,
    value: str | int | float | bool,
) -> dict[str, Any]:
    if operator is VectorFilterOperator.EQ:
        return {"key": key, "match": {"value": value}}
    return {"key": key, "range": {operator.value: value}}


def _require_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        msg = "Qdrant search result must include payload"
        raise ValueError(msg)
    return payload


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | float | str | bytes | bytearray):
        return int(value)
    msg = f"Expected int-compatible payload value, got {type(value).__name__}"
    raise TypeError(msg)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
