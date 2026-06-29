import argparse
import asyncio
import json
import time
from threading import Lock

from enterprise_rag_engine.evals.pipeline_benchmark import benchmark_pipeline
from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import Document, DocumentType, ParseResult, ParseStatus


class SyntheticParser(BaseParser):
    """Deterministic blocking parser used to benchmark pipeline scheduling."""

    def __init__(self, *, delay_ms: float, payload_kb: int) -> None:
        self._delay_seconds = delay_ms / 1000
        self._content = "R" * (payload_kb * 1024)
        self._lock = Lock()
        self.active_count = 0
        self.max_active_count = 0

    def parse(self, source_uri: str) -> ParseResult:
        with self._lock:
            self.active_count += 1
            self.max_active_count = max(self.max_active_count, self.active_count)
        started_at = time.perf_counter()
        try:
            time.sleep(self._delay_seconds)
            document = Document(
                source_uri=source_uri,
                type=DocumentType.TEXT,
                content=self._content,
            )
            return ParseResult(
                document=document,
                status=ParseStatus.SUCCEEDED,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000, 3),
            )
        finally:
            with self._lock:
                self.active_count -= 1


async def run(args: argparse.Namespace) -> None:
    parser = SyntheticParser(delay_ms=args.parser_delay_ms, payload_kb=args.payload_kb)
    source_uris = tuple(f"benchmark://document-{index}.txt" for index in range(args.files))
    metrics = await benchmark_pipeline(
        parser,
        source_uris,
        max_concurrency=args.concurrency,
        max_queue_size=args.queue_size,
    )
    payload = metrics.as_dict() | {"observed_max_concurrency": parser.max_active_count}
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Benchmark AsyncDocumentPipeline with deterministic synthetic files."
    )
    argument_parser.add_argument("--files", type=int, default=20)
    argument_parser.add_argument("--concurrency", type=int, default=4)
    argument_parser.add_argument("--queue-size", type=int, default=8)
    argument_parser.add_argument("--parser-delay-ms", type=float, default=50.0)
    argument_parser.add_argument("--payload-kb", type=int, default=64)
    return argument_parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.files, args.concurrency, args.queue_size, args.payload_kb) < 1:
        raise SystemExit("files, concurrency, queue-size, and payload-kb must be positive")
    if args.parser_delay_ms < 0:
        raise SystemExit("parser-delay-ms must not be negative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
