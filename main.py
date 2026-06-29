from fastapi import FastAPI

from app.api.routes_ask import router as ask_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.core.config import get_settings
from app.core.logging import setup_logging


setup_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Agentic RAG system for technical document question answering",
    version="0.7.0",
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(ask_router)
app.include_router(documents_router)
