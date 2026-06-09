from fastapi import FastAPI

from enterprise_rag_engine.api.routers.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG Engine",
        version="0.1.0",
        summary="Production-minded backend for enterprise retrieval-augmented generation.",
    )
    app.include_router(health_router)
    return app


app = create_app()
