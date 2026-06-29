from fastapi import FastAPI

from app.api.routes_ask import router as ask_router
from app.api.routes_eval import router as eval_router
from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.core.config import get_settings
from app.core.logging import setup_logging


setup_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Agentic RAG and automatic evaluation system for technical documents.",
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(ask_router)
app.include_router(eval_router)
