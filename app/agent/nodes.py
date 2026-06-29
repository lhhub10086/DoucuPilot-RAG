from dataclasses import dataclass

from app.agent.graders import (
    classify_question_type,
    grade_answer_support,
    grade_retrieval,
    rewrite_by_rule,
)
from app.agent.state import AgentState
from app.core.schemas import RetrievedChunk
from app.indexer import IndexManager
from app.llm.citation_answer import build_citation_answer
from app.reranker import BaseReranker, CrossEncoderReranker
from app.retriever import HybridRetriever, RerankRetriever


def append_trace(state: AgentState, message: str) -> None:
    state.setdefault("route_trace", []).append(message)


def chunks_to_dicts(chunks: list[RetrievedChunk]) -> list[dict]:
    return [chunk.model_dump() for chunk in chunks]


def dicts_to_chunks(items: list[dict]) -> list[RetrievedChunk]:
    return [RetrievedChunk(**item) for item in items]


@dataclass
class AgentNodes:
    index_manager: IndexManager
    reranker: BaseReranker | None = None
    top_k: int = 5

    def __post_init__(self) -> None:
        if self.reranker is None:
            self.reranker = CrossEncoderReranker()

    def classify_question(self, state: AgentState) -> AgentState:
        question_type = classify_question_type(state["question"])
        state["question_type"] = question_type
        append_trace(state, f"classify_question: {question_type}")
        return state

    def retrieve_documents(self, state: AgentState) -> AgentState:
        query = state.get("rewritten_question") or state["question"]
        candidates = HybridRetriever(self.index_manager).retrieve(query, top_k=max(self.top_k * 3, self.top_k))
        state["documents"] = chunks_to_dicts(candidates)
        state["retrieved_chunk_ids"] = [chunk.chunk_id for chunk in candidates]
        append_trace(state, f"retrieve_documents: {len(candidates)} candidates")
        return state

    def grade_documents(self, state: AgentState) -> AgentState:
        query = state.get("rewritten_question") or state["question"]
        documents = dicts_to_chunks(state.get("documents", []))
        if state.get("question_type") in {"summary", "method", "compare"} and documents:
            passed = max(document.score for document in documents) >= 0.005
            failure_reason = None if passed else "low_retrieval_confidence"
        else:
            passed, failure_reason = grade_retrieval(query, documents)
        state["retrieval_passed"] = passed
        state["need_rewrite"] = not passed and state.get("rewrite_count", 0) < 1
        if not passed:
            state["failure_reason"] = failure_reason
        append_trace(state, f"grade_documents: {'passed' if passed else 'failed'}")
        return state

    def rewrite_query(self, state: AgentState) -> AgentState:
        rewritten = rewrite_by_rule(state["question"], state.get("question_type"))
        state["rewritten_question"] = rewritten
        state["rewrite_count"] = state.get("rewrite_count", 0) + 1
        append_trace(state, f"rewrite_query: {rewritten}")
        return state

    def rerank_documents(self, state: AgentState) -> AgentState:
        query = state.get("rewritten_question") or state["question"]
        documents = dicts_to_chunks(state.get("documents", []))
        reranked = self.reranker.rerank(query, documents, top_k=self.top_k) if self.reranker else []
        reranked = RerankRetriever(self.index_manager, reranker=self.reranker)._apply_document_prior(query, reranked)
        reranked = reranked[: self.top_k]
        state["reranked_documents"] = chunks_to_dicts(reranked)
        state["reranked_chunk_ids"] = [chunk.chunk_id for chunk in reranked]
        before = ",".join(state.get("retrieved_chunk_ids", [])[: self.top_k])
        after = ",".join(state.get("reranked_chunk_ids", []))
        append_trace(state, f"rerank_documents: {before} -> {after}")
        return state

    def generate_answer(self, state: AgentState) -> AgentState:
        reranked = dicts_to_chunks(state.get("reranked_documents", []))
        answer, citations = build_citation_answer(state["question"], reranked)
        state["answer"] = answer
        state["citations"] = [citation.model_dump() for citation in citations]
        append_trace(state, "generate_answer")
        return state

    def grade_answer(self, state: AgentState) -> AgentState:
        supported, label = grade_answer_support(state.get("answer", ""), state.get("citations", []))
        state["answer_supported"] = supported
        if not supported:
            state["failure_reason"] = state.get("failure_reason") or "insufficient_context"
        append_trace(state, f"grade_answer: {label}")
        return state

    def fallback_response(self, state: AgentState) -> AgentState:
        state["answer"] = "当前文档中未找到充分依据。"
        state["citations"] = []
        state["failure_reason"] = state.get("failure_reason") or "insufficient_context"
        state.setdefault("retrieved_chunk_ids", [])
        state.setdefault("reranked_chunk_ids", [])
        append_trace(state, "fallback_response")
        return state
