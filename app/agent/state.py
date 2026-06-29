from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    rewritten_question: Optional[str]
    question_type: Optional[str]
    documents: list[dict[str, Any]]
    reranked_documents: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    route_trace: list[str]
    need_rewrite: bool
    rewrite_count: int
    retrieval_passed: bool
    answer_supported: bool
    failure_reason: Optional[str]
    latency_ms: float
    retrieved_chunk_ids: list[str]
    reranked_chunk_ids: list[str]
