from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    detail: str


class Citation(BaseModel):
    file_name: str
    page_num: int | None = None
    section: str = "unknown"
    chunk_id: str
    text_preview: str = ""
    score: float = 0.0
    rerank_score: float = 0.0


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    strategy: Literal["naive", "hybrid", "rerank", "agentic"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    strategy: str
    route_trace: list[str] = []
    latency_ms: float


class ParsedBlock(BaseModel):
    text: str
    block_type: str = "paragraph"
    metadata: dict[str, Any] = {}


class ParsedPage(BaseModel):
    page_num: int
    text: str
    blocks: list[ParsedBlock] = []
    metadata: dict[str, Any] = {}


class ParsedDocument(BaseModel):
    doc_id: str
    file_name: str
    file_type: Literal["pdf", "md", "txt"]
    pages: list[ParsedPage]
    metadata: dict[str, Any] = {}


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


class RetrievedChunk(DocumentChunk):
    score: float = 0.0
    rerank_score: float = 0.0
    rank: int = 0
    retriever: str = ""
