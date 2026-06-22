import pytest

from enterprise_rag_engine import (
    AsyncDocumentPipeline,
    CacheManager,
    Document,
    DocumentType,
    ParseProgressEvent,
    ParseProgressStage,
    ParseResult,
    ParseStatus,
    file_content_hash,
)
from enterprise_rag_engine.interfaces import BaseParser


class CountingParser(BaseParser):
    def __init__(self) -> None:
        self.call_count = 0

    def parse(self, source_uri: str) -> ParseResult:
        self.call_count += 1
        document = Document(
            source_uri=source_uri,
            type=DocumentType.TEXT,
            content=f"parsed:{self.call_count}",
        )
        return ParseResult(document=document, status=ParseStatus.SUCCEEDED, elapsed_ms=1.0)


class FailingParser(BaseParser):
    def __init__(self) -> None:
        self.call_count = 0

    def parse(self, source_uri: str) -> ParseResult:
        self.call_count += 1
        document = Document(source_uri=source_uri, type=DocumentType.TEXT, content="")
        return ParseResult(
            document=document,
            status=ParseStatus.FAILED,
            errors=("failed",),
            elapsed_ms=1.0,
        )


def test_file_content_hash_returns_none_for_missing_file() -> None:
    assert file_content_hash("missing.txt") is None


@pytest.mark.anyio
async def test_cache_manager_invalidates_when_file_content_changes() -> None:
    hash_provider = MutableHashProvider("v1")
    parser = CountingParser()
    cache = CacheManager(hash_provider=hash_provider)
    pipeline = AsyncDocumentPipeline(parser, cache_manager=cache)

    first = await pipeline.parse("demo.txt")
    second = await pipeline.parse("demo.txt")
    hash_provider.value = "v2"
    third = await pipeline.parse("demo.txt")

    assert first.document.content == "parsed:1"
    assert second.document.content == "parsed:1"
    assert third.document.content == "parsed:2"
    assert parser.call_count == 2


@pytest.mark.anyio
async def test_cache_manager_does_not_cache_failed_results() -> None:
    parser = FailingParser()
    pipeline = AsyncDocumentPipeline(
        parser,
        cache_manager=CacheManager(hash_provider=lambda _source_uri: "same-hash"),
    )

    await pipeline.parse("bad.txt")
    await pipeline.parse("bad.txt")

    assert parser.call_count == 2


@pytest.mark.anyio
async def test_cache_manager_emits_cache_hit_progress_event() -> None:
    events: list[ParseProgressEvent] = []
    parser = CountingParser()
    pipeline = AsyncDocumentPipeline(
        parser,
        cache_manager=CacheManager(hash_provider=lambda _source_uri: "same-hash"),
        progress_handler=events.append,
    )

    await pipeline.parse("demo.txt", task_id="task-cache")
    await pipeline.parse("demo.txt", task_id="task-cache")

    assert parser.call_count == 1
    assert events[-1].stage is ParseProgressStage.CACHE_HIT
    assert events[-1].status is ParseStatus.SUCCEEDED


class MutableHashProvider:
    def __init__(self, value: str) -> None:
        self.value = value

    def __call__(self, source_uri: str) -> str:
        return f"{source_uri}:{self.value}"
