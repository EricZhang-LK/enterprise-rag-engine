from fastapi import APIRouter
from fastapi.testclient import TestClient

from enterprise_rag_engine.api import create_app
from enterprise_rag_engine.exceptions import ResourceNotFoundError


def test_application_error_returns_standard_payload() -> None:
    app = create_app()
    router = APIRouter()

    @router.get("/missing")
    def missing_resource() -> None:
        raise ResourceNotFoundError("Document not found")

    app.include_router(router)
    client = TestClient(app)

    response = client.get("/missing")

    assert response.status_code == 404
    assert response.json() == {
        "error_code": "RESOURCE_NOT_FOUND",
        "message": "Document not found",
    }
