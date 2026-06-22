from pathlib import Path

from fastapi.testclient import TestClient

from enterprise_rag_engine.api import create_app
from enterprise_rag_engine.api.repositories.tasks import InMemoryTaskRepository
from enterprise_rag_engine.api.services.documents import DocumentTaskService
from enterprise_rag_engine.interfaces import BaseParser
from enterprise_rag_engine.models import Document, DocumentType, ParseResult, ParseStatus


class InMemoryDocumentStorage:
    def __init__(self) -> None:
        self.content_by_uri: dict[str, bytes] = {}

    async def save(self, *, task_id: str, filename: str, content: bytes) -> str:
        source_uri = str(Path("virtual") / task_id / filename)
        self.content_by_uri[source_uri] = content
        return source_uri


class StaticParser(BaseParser):
    def parse(self, source_uri: str) -> ParseResult:
        document = Document(
            source_uri=source_uri,
            type=DocumentType.MARKDOWN,
            content="# Enterprise RAG\n\nUploaded content.",
        )
        return ParseResult(
            document=document,
            status=ParseStatus.SUCCEEDED,
            elapsed_ms=2.5,
        )


class StaticParserProvider:
    def __init__(self, parser: BaseParser) -> None:
        self._parser = parser

    def get(self, source_uri: str) -> BaseParser:
        return self._parser


def _test_client(*, max_upload_bytes: int = 1024) -> TestClient:
    service = DocumentTaskService(
        repository=InMemoryTaskRepository(),
        storage=InMemoryDocumentStorage(),
        parser_registry=StaticParserProvider(StaticParser()),
        max_upload_bytes=max_upload_bytes,
    )
    app = create_app()
    app.state.document_task_service = service
    return TestClient(app)


def test_upload_document_creates_background_parse_task() -> None:
    client = _test_client()

    upload_response = client.post(
        "/documents/upload",
        files={"file": ("demo.md", b"# Enterprise RAG", "text/markdown")},
    )

    assert upload_response.status_code == 202
    upload_payload = upload_response.json()
    assert upload_payload["filename"] == "demo.md"
    assert upload_payload["status"] == "queued"
    assert upload_payload["task_url"].endswith(f"/tasks/{upload_payload['task_id']}")

    task_response = client.get(f"/tasks/{upload_payload['task_id']}")
    assert task_response.status_code == 200
    task_payload = task_response.json()
    assert task_payload["task_id"] == upload_payload["task_id"]
    assert task_payload["filename"] == "demo.md"
    assert task_payload["status"] == "succeeded"
    assert task_payload["stage"] == "succeeded"
    assert task_payload["progress"] == 1.0
    assert task_payload["parse_status"] == "succeeded"
    assert task_payload["chunk_count"] == 0
    assert task_payload["elapsed_ms"] == 2.5
    assert task_payload["errors"] == []


def test_upload_document_rejects_unsupported_extension() -> None:
    client = _test_client()

    response = client.post(
        "/documents/upload",
        files={"file": ("demo.exe", b"not a document", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "VALIDATION_FAILED"
    assert "Unsupported document type" in response.json()["message"]


def test_upload_document_rejects_oversized_content() -> None:
    client = _test_client(max_upload_bytes=4)

    response = client.post(
        "/documents/upload",
        files={"file": ("demo.md", b"12345", "text/markdown")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error_code": "VALIDATION_FAILED",
        "message": "Uploaded document exceeds 4 bytes",
    }


def test_get_parse_task_returns_not_found_for_unknown_id() -> None:
    client = _test_client()

    response = client.get("/tasks/missing-task")

    assert response.status_code == 404
    assert response.json() == {
        "error_code": "RESOURCE_NOT_FOUND",
        "message": "Parse task not found: missing-task",
    }
