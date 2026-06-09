from enterprise_rag_engine import __version__
from enterprise_rag_engine.api.schemas.health import HealthResponse


def get_health_status() -> HealthResponse:
    return HealthResponse(
        service="enterprise-rag-engine",
        status="ok",
        version=__version__,
    )
