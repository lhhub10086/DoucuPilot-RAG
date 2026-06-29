from app.core.schemas import RetrievedChunk
from app.indexer import IndexManager
from app.indexer.embedding import HashingEmbeddingProvider
from app.reranker.base import BaseReranker
from app.retriever.rerank_retriever import RerankRetriever
from tests.test_retriever import build_chunks


class KeywordReranker(BaseReranker):
    def rerank(self, query: str, documents: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        scored = []
        for document in documents:
            score = 10.0 if "BM25" in document.text else 0.0
            scored.append(
                RetrievedChunk(
                    **document.model_dump(exclude={"rerank_score", "rank", "retriever"}),
                    rerank_score=score,
                    retriever="rerank",
                )
            )
        scored.sort(key=lambda item: item.rerank_score, reverse=True)
        for rank, item in enumerate(scored[:top_k], start=1):
            item.rank = rank
        return scored[:top_k]


def test_rerank_retriever_returns_rerank_scores() -> None:
    manager = IndexManager(embedding_provider=HashingEmbeddingProvider())
    manager.add_documents(build_chunks())

    results = RerankRetriever(manager, reranker=KeywordReranker()).retrieve(
        "keyword retrieval",
        top_k=1,
        candidate_k=2,
    )

    assert results[0].metadata["file_name"] == "search.md"
    assert results[0].rerank_score >= 10.0
    assert results[0].retriever == "rerank"
