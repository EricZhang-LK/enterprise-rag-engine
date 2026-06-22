import time
from threading import Lock

import pytest

from enterprise_rag_engine import (
    AsyncDocumentPipeline,
    Document,
    DocumentType,
    ParseProgressEvent,
    ParseProgressStage,
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


class TrackingParser(BaseParser):
    def __init__(self, *, sleep_seconds: float = 0.04) -> None:
        self._sleep_seconds = sleep_seconds
        self._lock = Lock()
        self.active_count = 0
        self.max_active_count = 0

    def parse(self, source_uri: str) -> ParseResult:
        with self._lock:
            self.active_count += 1
            self.max_active_count = max(self.max_active_count, self.active_count)
        try:
            time.sleep(self._sleep_seconds)
            document = Document(source_uri=source_uri, type=DocumentType.TEXT, content=source_uri)
            return ParseResult(document=document, status=ParseStatus.SUCCEEDED, elapsed_ms=1.0)
        finally:
            with self._lock:
                self.active_count -= 1


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
async def test_async_document_pipeline_respects_max_concurrency() -> None:
    parser = TrackingParser()
    pipeline = AsyncDocumentPipeline(parser, max_concurrency=2, max_queue_size=2)

    results = await pipeline.parse_many(("a.txt", "b.txt", "c.txt", "d.txt", "e.txt"))

    assert [result.document.source_uri for result in results] == [
        "a.txt",
        "b.txt",
        "c.txt",
        "d.txt",
        "e.txt",
    ]
    assert parser.max_active_count <= 2


@pytest.mark.anyio
async def test_async_document_pipeline_small_queue_still_preserves_order() -> None:
    pipeline = AsyncDocumentPipeline(SlowParser(), max_concurrency=2, max_queue_size=1)

    results = await pipeline.parse_many(("a.txt", "b.txt", "c.txt"))

    assert [result.document.source_uri for result in results] == ["a.txt", "b.txt", "c.txt"]


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


def test_async_document_pipeline_rejects_invalid_concurrency_settings() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        AsyncDocumentPipeline(SlowParser(), max_concurrency=0)

    with pytest.raises(ValueError, match="max_queue_size"):
        AsyncDocumentPipeline(SlowParser(), max_queue_size=0)


@pytest.mark.anyio
async def test_async_document_pipeline_emits_progress_events_for_one_source() -> None:
    events: list[ParseProgressEvent] = []
    pipeline = AsyncDocumentPipeline(SlowParser(), progress_handler=events.append)

    result = await pipeline.parse("demo.txt", task_id="task-1")

    assert result.status is ParseStatus.SUCCEEDED
    assert [event.stage for event in events] == [
        ParseProgressStage.STARTED,
        ParseProgressStage.SUCCEEDED,
    ]
    assert {event.task_id for event in events} == {"task-1"}
    assert events[-1].metadata["chunk_count"] == 0


@pytest.mark.anyio
async def test_async_document_pipeline_emits_queued_events_for_batch() -> None:
    events: list[ParseProgressEvent] = []
    pipeline = AsyncDocumentPipeline(
        SlowParser(),
        max_concurrency=2,
        max_queue_size=1,
        progress_handler=events.append,
    )

    await pipeline.parse_many(("a.txt", "b.txt"))

    queued_events = [event for event in events if event.stage is ParseProgressStage.QUEUED]
    assert [event.source_uri for event in queued_events] == ["a.txt", "b.txt"]
    assert [event.metadata["queue_index"] for event in queued_events] == [0, 1]


@pytest.mark.anyio
async def test_async_document_pipeline_supports_async_progress_handler() -> None:
    events: list[ParseProgressEvent] = []

    async def collect_event(event: ParseProgressEvent) -> None:
        events.append(event)

    pipeline = AsyncDocumentPipeline(SlowParser(), progress_handler=collect_event)

    await pipeline.parse("demo.txt")

    assert events[-1].stage is ParseProgressStage.SUCCEEDED


@pytest.mark.anyio
async def test_async_document_pipeline_emits_failed_event_on_exception() -> None:
    events: list[ParseProgressEvent] = []
    pipeline = AsyncDocumentPipeline(ExplodingParser(), progress_handler=events.append)

    result = await pipeline.parse("bad.txt", task_id="task-bad")

    assert result.status is ParseStatus.FAILED
    assert events[-1].stage is ParseProgressStage.FAILED
    assert events[-1].status is ParseStatus.FAILED
    assert events[-1].task_id == "task-bad"
