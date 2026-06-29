from app.agent import run_agentic_rag
from app.core.schemas import RetrievedChunk
from app.indexer import IndexManager
from app.indexer.embedding import HashingEmbeddingProvider
from app.reranker.base import BaseReranker
from tests.test_retriever import build_chunks


class SimpleReranker(BaseReranker):
    def rerank(self, query: str, documents: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        reranked = []
        for document in documents:
            score = 0.2 if "BM25" in document.text or "metadata" in document.text else 0.00001
            reranked.append(
                RetrievedChunk(
                    **document.model_dump(exclude={"rerank_score", "rank", "retriever"}),
                    rerank_score=score,
                    retriever="rerank",
                )
            )
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        for rank, item in enumerate(reranked[:top_k], start=1):
            item.rank = rank
        return reranked[:top_k]


def build_manager() -> IndexManager:
    manager = IndexManager(embedding_provider=HashingEmbeddingProvider())
    manager.add_documents(build_chunks())
    return manager


def test_agentic_strategy_runs_with_route_trace() -> None:
    state = run_agentic_rag(
        "BM25 keyword retrieval 方法是什么？",
        index_manager=build_manager(),
        reranker=SimpleReranker(),
        top_k=2,
    )

    assert state["answer"]
    assert state["citations"]
    assert state["route_trace"]
    assert any(step.startswith("classify_question") for step in state["route_trace"])
    assert any(step.startswith("retrieve_documents") for step in state["route_trace"])
    assert any(step.startswith("grade_documents") for step in state["route_trace"])
    assert any(step.startswith("rerank_documents") for step in state["route_trace"])


def test_agentic_fallback_for_unanswerable_question() -> None:
    state = run_agentic_rag(
        "这个文档有没有提到火星移民计划？",
        index_manager=build_manager(),
        reranker=SimpleReranker(),
        top_k=2,
    )

    assert state["answer"] == "当前文档中未找到充分依据。"
    assert state["citations"] == []
    assert state["failure_reason"] in {"low_retrieval_confidence", "insufficient_context"}
    assert any("fallback_response" == step for step in state["route_trace"])
