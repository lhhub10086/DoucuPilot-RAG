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
    question: str
    strategy: str = "agentic"
    chunk_strategy: str = "section"
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    strategy: str
    latency_ms: float
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    rewritten_question: str | None = None
    route_trace: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


class UploadResponse(BaseModel):
    status: str
    file_name: str
    chunk_strategy: str
    num_documents: int
    num_chunks: int
    message: str


class DocumentInfo(BaseModel):
    file_name: str
    file_type: str
    source_path: str
    parsed: bool = False
    doc_id: str | None = None
    page_count: int | None = None


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    count: int


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
