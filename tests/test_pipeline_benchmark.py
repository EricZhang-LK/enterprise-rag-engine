import time

import pytest

from enterprise_rag_engine.evals.pipeline_benchmark import benchmark_pipeline
from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import Document, DocumentType, ParseResult, ParseStatus


class BenchmarkParser(BaseParser):
    def parse(self, source_uri: str) -> ParseResult:
        time.sleep(0.002)
        document = Document(source_uri=source_uri, type=DocumentType.TEXT, content="benchmark")
        return ParseResult(
            document=document,
            status=ParseStatus.SUCCEEDED,
            elapsed_ms=2.0,
        )


@pytest.mark.anyio
async def test_benchmark_pipeline_reports_latency_throughput_and_memory() -> None:
    metrics = await benchmark_pipeline(
        BenchmarkParser(),
        tuple(f"benchmark://{index}" for index in range(20)),
        max_concurrency=4,
        max_queue_size=4,
    )

    assert metrics.file_count == 20
    assert metrics.succeeded_count == 20
    assert metrics.failed_count == 0
    assert metrics.throughput_files_per_second > 0
    assert 0 < metrics.p50_latency_ms <= metrics.p95_latency_ms
    assert metrics.p95_latency_ms <= metrics.max_latency_ms
    assert metrics.peak_memory_mb >= 0
