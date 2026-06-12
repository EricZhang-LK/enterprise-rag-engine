import asyncio
from collections.abc import Sequence
from time import perf_counter

from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import Document, DocumentType, ParseResult, ParseStatus


class AsyncDocumentPipeline:
    """Async wrapper around synchronous document parsers.

    Most document parsers are CPU-bound or blocking I/O adapters. The pipeline runs
    them in a worker thread so FastAPI handlers can await parsing without blocking
    the event loop.
    """

    def __init__(self, parser: BaseParser) -> None:
        self._parser = parser

    async def parse(self, source_uri: str) -> ParseResult:
        """Parse one source without blocking the event loop."""

        started_at = perf_counter()
        try:
            return await asyncio.to_thread(self._parser.parse, source_uri)
        except Exception as exc:
            document = Document(source_uri=source_uri, type=DocumentType.UNKNOWN, content="")
            return ParseResult(
                document=document,
                status=ParseStatus.FAILED,
                errors=(f"Async parse failed: {exc}",),
                elapsed_ms=_elapsed_ms(started_at),
            )

    async def parse_many(self, source_uris: Sequence[str]) -> tuple[ParseResult, ...]:
        """Parse a batch of sources concurrently with fail-soft semantics."""

        tasks = tuple(self.parse(source_uri) for source_uri in source_uris)
        if not tasks:
            return ()
        return tuple(await asyncio.gather(*tasks))


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
