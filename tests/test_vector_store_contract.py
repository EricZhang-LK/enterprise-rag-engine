from collections.abc import Sequence

from enterprise_rag_engine import (
    BaseVectorStore,
    ChunkMetadata,
    ChunkType,
    DocumentChunk,
    RetrievalResult,
    VectorSearchRequest,
    VectorStoreFilter,
    VectorStoreRecord,
    VectorStoreWriteResult,
)


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.records: list[VectorStoreRecord] = []

    def upsert(self, records: Sequence[VectorStoreRecord]) -> VectorStoreWriteResult:
        self.records.extend(records)
        return VectorStoreWriteResult(affected_count=len(records))

    def search(self, request: VectorSearchRequest) -> tuple[RetrievalResult, ...]:
        filtered = [
            record
            for record in self.records
            if request.filters.matches_payload(record.chunk.metadata_payload())
        ]
        return tuple(
            RetrievalResult(
                query=request.query_text or "",
                chunk=record.chunk,
                score=1.0,
                rank=rank,
                retriever="fake",
            )
            for rank, record in enumerate(filtered[: request.top_k], start=1)
        )

    def delete(self, filters: VectorStoreFilter) -> VectorStoreWriteResult:
        before_count = len(self.records)
        self.records = [
            record
            for record in self.records
            if not filters.matches_payload(record.chunk.metadata_payload())
        ]
        return VectorStoreWriteResult(affected_count=before_count - len(self.records))


def test_vector_store_contract_supports_upsert_search_and_filtered_delete() -> None:
    store = FakeVectorStore()
    keep = VectorStoreRecord(chunk=_chunk("chunk-1", tenant_id="tenant-a"), embedding=(0.1, 0.2))
    delete = VectorStoreRecord(chunk=_chunk("chunk-2", tenant_id="tenant-b"), embedding=(0.3, 0.4))

    upsert_result = store.upsert((keep, delete))
    search_results = store.search(
        VectorSearchRequest(
            query_embedding=(0.9, 0.8),
            top_k=10,
            filters=VectorStoreFilter.equals("tenant_id", "tenant-a"),
            query_text="vector store contract",
        )
    )
    delete_result = store.delete(VectorStoreFilter.equals("tenant_id", "tenant-b"))

    assert upsert_result.affected_count == 2
    assert [result.chunk.id for result in search_results] == ["chunk-1"]
    assert search_results[0].query == "vector store contract"
    assert delete_result.affected_count == 1
    assert [record.chunk.id for record in store.records] == ["chunk-1"]


def test_vector_store_filter_matches_multiple_exact_conditions() -> None:
    filters = VectorStoreFilter.exact({"tenant_id": "tenant-a", "document_id": "doc-1"})

    assert filters.matches_payload({"tenant_id": "tenant-a", "document_id": "doc-1"})
    assert not filters.matches_payload({"tenant_id": "tenant-a", "document_id": "doc-2"})


def test_vector_store_filter_matches_tenant_document_page_and_chunk_type() -> None:
    filters = VectorStoreFilter.metadata(
        tenant_id="tenant-a",
        document_id="doc-1",
        page_number=3,
        chunk_type=ChunkType.TEXT,
    )

    assert filters.matches_payload(
        {
            "tenant_id": "tenant-a",
            "document_id": "doc-1",
            "page_number": 2,
            "end_page_number": 4,
            "chunk_type": "text",
        }
    )
    assert filters.matches_payload(
        {
            "tenant_id": "tenant-a",
            "document_id": "doc-1",
            "page_number": 3,
            "end_page_number": None,
            "chunk_type": "text",
        }
    )
    assert not filters.matches_payload(
        {
            "tenant_id": "tenant-b",
            "document_id": "doc-1",
            "page_number": 3,
            "end_page_number": 3,
            "chunk_type": "text",
        }
    )
    assert not filters.matches_payload(
        {
            "tenant_id": "tenant-a",
            "document_id": "doc-1",
            "page_number": 4,
            "end_page_number": 5,
            "chunk_type": "text",
        }
    )
    assert not filters.matches_payload(
        {
            "tenant_id": "tenant-a",
            "document_id": "doc-1",
            "page_number": 3,
            "end_page_number": 3,
            "chunk_type": "table",
        }
    )


def test_vector_search_request_rejects_invalid_top_k() -> None:
    try:
        VectorSearchRequest(query_embedding=(0.1,), top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("top_k=0 should be rejected")


def _chunk(chunk_id: str, *, tenant_id: str) -> DocumentChunk:
    metadata = ChunkMetadata(source_uri="demo.md", document_id="doc-1", tenant_id=tenant_id)
    return DocumentChunk(
        id=chunk_id,
        document_id="doc-1",
        content=f"content for {chunk_id}",
        metadata=metadata,
    )
