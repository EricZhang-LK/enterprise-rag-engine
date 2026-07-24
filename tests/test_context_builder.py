import pytest

from enterprise_rag_engine import ChunkMetadata, DocumentChunk, RetrievalResult
from enterprise_rag_engine.generation import ContextBuilder


def test_context_builder_sorts_by_rank_and_deduplicates_chunk_ids() -> None:
    builder = ContextBuilder(max_characters=200)

    context = builder.build(
        (
            _result("chunk-2", content="Second evidence.", score=0.9, rank=2),
            _result("chunk-1", content="First evidence.", score=0.8, rank=1),
            _result("chunk-1", content="Duplicate evidence.", score=0.99, rank=3),
        )
    )

    assert [item.chunk.id for item in context.results] == ["chunk-1", "chunk-2"]
    assert context.deduplicated_count == 1
    assert context.text == "[chunk-1]\nFirst evidence.\n\n[chunk-2]\nSecond evidence."
    assert context.truncated is False


def test_context_builder_keeps_complete_higher_ranked_context_within_budget() -> None:
    builder = ContextBuilder(max_characters=18)

    context = builder.build(
        (
            _result("chunk-1", content="Alpha", score=0.9, rank=1),
            _result("chunk-2", content="Beta", score=0.8, rank=2),
        )
    )

    assert [item.chunk.id for item in context.results] == ["chunk-1"]
    assert context.text == "[chunk-1]\nAlpha"
    assert context.character_count == len(context.text)
    assert context.character_count <= 18
    assert context.truncated is True


def test_context_builder_handles_empty_results_and_rejects_invalid_budget() -> None:
    assert ContextBuilder(max_characters=10).build(()).text == ""

    with pytest.raises(ValueError, match="max_characters"):
        ContextBuilder(max_characters=0)


def _result(chunk_id: str, *, content: str, score: float, rank: int) -> RetrievalResult:
    return RetrievalResult(
        query="What is the evidence?",
        chunk=DocumentChunk(
            id=chunk_id,
            document_id="doc-1",
            content=content,
            metadata=ChunkMetadata(source_uri="handbook.md", document_id="doc-1"),
        ),
        score=score,
        rank=rank,
        retriever="hybrid_rerank",
    )
