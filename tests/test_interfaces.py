from enterprise_rag_engine import (
    BaseEvaluator,
    BaseParser,
    BaseSplitter,
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentType,
    ParseResult,
    ParseStatus,
)


class IncompleteParser(BaseParser):
    pass


class FakeParser(BaseParser):
    def parse(self, source_uri: str) -> ParseResult:
        document = Document(source_uri=source_uri, type=DocumentType.TEXT, content="hello rag")
        return ParseResult(document=document, status=ParseStatus.SUCCEEDED, elapsed_ms=1.0)


class FakeSplitter(BaseSplitter):
    def split(self, document: Document) -> tuple[DocumentChunk, ...]:
        metadata = ChunkMetadata(source_uri=document.source_uri, document_id=document.id)
        chunk = DocumentChunk(document_id=document.id, content=document.content, metadata=metadata)
        return (chunk,)


class FakeEvaluator(BaseEvaluator):
    def evaluate(self) -> dict[str, float]:
        return {"recall_at_5": 1.0}


def run_parser(parser: BaseParser, source_uri: str) -> ParseResult:
    return parser.parse(source_uri)


def run_splitter(splitter: BaseSplitter, document: Document) -> tuple[DocumentChunk, ...]:
    return splitter.split(document)


def run_evaluator(evaluator: BaseEvaluator) -> dict[str, float]:
    return evaluator.evaluate()


def test_parser_abc_accepts_explicit_implementation() -> None:
    result = run_parser(FakeParser(), "demo.txt")

    assert result.document.source_uri == "demo.txt"
    assert result.status is ParseStatus.SUCCEEDED


def test_parser_abc_rejects_incomplete_implementation() -> None:
    try:
        IncompleteParser()  # type: ignore[abstract]
    except TypeError as exc:
        assert "abstract" in str(exc)
    else:
        raise AssertionError("IncompleteParser should not be instantiable")


def test_splitter_abc_accepts_explicit_implementation() -> None:
    document = Document(source_uri="demo.txt", type=DocumentType.TEXT, content="hello rag")
    chunks = run_splitter(FakeSplitter(), document)

    assert len(chunks) == 1
    assert chunks[0].document_id == document.id


def test_evaluator_abc_returns_metrics() -> None:
    metrics = run_evaluator(FakeEvaluator())

    assert metrics == {"recall_at_5": 1.0}
