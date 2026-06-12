import pytest
from pydantic import ValidationError

from enterprise_rag_engine import (
    ChunkMetadata,
    ChunkRole,
    Document,
    DocumentChunk,
    DocumentType,
    ParseResult,
    ParseStatus,
    RetrievalResult,
)


def test_document_exposes_character_count() -> None:
    document = Document(
        source_uri="demo.pdf",
        type=DocumentType.PDF,
        title="Demo",
        content="hello rag",
    )

    assert document.character_count == 9


def test_parse_result_counts_chunks() -> None:
    document = Document(source_uri="demo.pdf", type=DocumentType.PDF, content="hello rag")
    metadata = ChunkMetadata(source_uri="demo.pdf", document_id=document.id, page_number=1)
    chunk = DocumentChunk(document_id=document.id, content="hello", metadata=metadata)
    result = ParseResult(
        document=document,
        chunks=(chunk,),
        status=ParseStatus.SUCCEEDED,
        elapsed_ms=12.5,
    )

    assert result.chunk_count == 1


def test_chunk_metadata_rejects_invalid_page_number() -> None:
    document = Document(source_uri="demo.pdf", type=DocumentType.PDF, content="hello rag")

    with pytest.raises(ValidationError):
        ChunkMetadata(source_uri="demo.pdf", document_id=document.id, page_number=0)


def test_chunk_metadata_exposes_enterprise_fields_with_defaults() -> None:
    document = Document(source_uri="demo.pdf", type=DocumentType.PDF, content="hello rag")

    metadata = ChunkMetadata(source_uri="demo.pdf", document_id=document.id)

    assert metadata.chunk_role is ChunkRole.STANDALONE
    assert metadata.has_table is False
    assert metadata.metadata == {}


def test_chunk_metadata_rejects_invalid_page_range() -> None:
    document = Document(source_uri="demo.pdf", type=DocumentType.PDF, content="hello rag")

    with pytest.raises(ValidationError):
        ChunkMetadata(
            source_uri="demo.pdf",
            document_id=document.id,
            page_number=2,
            end_page_number=1,
        )


def test_chunk_metadata_rejects_invalid_char_range() -> None:
    document = Document(source_uri="demo.pdf", type=DocumentType.PDF, content="hello rag")

    with pytest.raises(ValidationError):
        ChunkMetadata(
            source_uri="demo.pdf",
            document_id=document.id,
            start_char=10,
            end_char=2,
        )


def test_retrieval_result_requires_positive_rank() -> None:
    document = Document(source_uri="demo.pdf", type=DocumentType.PDF, content="hello rag")
    metadata = ChunkMetadata(source_uri="demo.pdf", document_id=document.id)
    chunk = DocumentChunk(document_id=document.id, content="hello", metadata=metadata)

    with pytest.raises(ValidationError):
        RetrievalResult(query="hello?", chunk=chunk, score=0.9, rank=0, retriever="dense")
