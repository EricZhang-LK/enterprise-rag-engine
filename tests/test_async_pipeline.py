import time

import pytest

from enterprise_rag_engine import (
    AsyncDocumentPipeline,
    Document,
    DocumentType,
    ParseResult,
    ParseStatus,
)
from enterprise_rag_engine.interfaces import BaseParser


class SlowParser(BaseParser):
    def parse(self, source_uri: str) -> ParseResult:
        time.sleep(0.05)
        document = Document(source_uri=source_uri, type=DocumentType.TEXT, content=source_uri)
        return ParseResult(document=document, status=ParseStatus.SUCCEEDED, elapsed_ms=1.0)


class ExplodingParser(BaseParser):
    def parse(self, source_uri: str) -> ParseResult:
        raise RuntimeError(f"boom: {source_uri}")


@pytest.mark.anyio
async def test_async_document_pipeline_parses_one_source() -> None:
    pipeline = AsyncDocumentPipeline(SlowParser())

    result = await pipeline.parse("demo.txt")

    assert result.status is ParseStatus.SUCCEEDED
    assert result.document.content == "demo.txt"


@pytest.mark.anyio
async def test_async_document_pipeline_parses_batch_concurrently() -> None:
    pipeline = AsyncDocumentPipeline(SlowParser())
    started_at = time.perf_counter()

    results = await pipeline.parse_many(("a.txt", "b.txt", "c.txt"))

    elapsed = time.perf_counter() - started_at
    assert [result.document.source_uri for result in results] == ["a.txt", "b.txt", "c.txt"]
    assert elapsed < 0.14


@pytest.mark.anyio
async def test_async_document_pipeline_returns_failed_result_on_exception() -> None:
    pipeline = AsyncDocumentPipeline(ExplodingParser())

    result = await pipeline.parse("bad.txt")

    assert result.status is ParseStatus.FAILED
    assert result.document.source_uri == "bad.txt"
    assert "boom: bad.txt" in result.errors[0]


@pytest.mark.anyio
async def test_async_document_pipeline_accepts_empty_batch() -> None:
    pipeline = AsyncDocumentPipeline(SlowParser())

    assert await pipeline.parse_many(()) == ()
