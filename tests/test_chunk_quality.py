from enterprise_rag_engine import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentType,
    RecursiveSplitter,
)
from enterprise_rag_engine.evals.chunk_quality import evaluate_chunks


def test_evaluate_chunks_returns_zero_metrics_for_empty_input() -> None:
    metrics = evaluate_chunks(())

    assert metrics.chunk_count == 0
    assert metrics.average_chars == 0.0
    assert metrics.context_complete_ratio == 0.0


def test_evaluate_chunks_calculates_length_and_context_completeness() -> None:
    document = Document(source_uri="demo.txt", type=DocumentType.TEXT, content="alpha beta gamma")

    chunks = RecursiveSplitter(max_chars=8, overlap_chars=0).split(document)
    metrics = evaluate_chunks(chunks)

    assert metrics.chunk_count == 3
    assert metrics.average_chars > 0
    assert metrics.p95_chars <= 8
    assert metrics.average_tokens > 0
    assert metrics.context_complete_ratio == 1.0


def test_evaluate_chunks_calculates_overlap_ratio() -> None:
    metadata = ChunkMetadata(
        source_uri="demo.txt",
        document_id="doc-1",
        content_hash="hash",
        splitter="test",
        token_count=2,
        start_char=0,
        end_char=10,
    )
    first = DocumentChunk(
        document_id="doc-1",
        content="0123456789",
        metadata=metadata,
        start_char=0,
        end_char=10,
    )
    second = DocumentChunk(
        document_id="doc-1",
        content="56789abcde",
        metadata=metadata.model_copy(update={"start_char": 5, "end_char": 15}),
        start_char=5,
        end_char=15,
    )

    metrics = evaluate_chunks((first, second))

    assert metrics.overlap_ratio == 0.25


def test_evaluate_chunks_detects_incomplete_context() -> None:
    metadata = ChunkMetadata(source_uri="demo.txt", document_id="doc-1")
    chunk = DocumentChunk(document_id="doc-1", content="hello", metadata=metadata)

    metrics = evaluate_chunks((chunk,))

    assert metrics.context_complete_ratio == 0.0
