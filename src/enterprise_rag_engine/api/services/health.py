from enterprise_rag_engine.api.schemas.health import HealthResponse
from enterprise_rag_engine.settings import get_settings


def get_health_status() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        service=settings.app_name,
        status="ok",
        version=settings.app_version,
    )
