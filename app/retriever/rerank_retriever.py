from app.core.schemas import RetrievedChunk
from app.indexer import IndexManager
from app.indexer.vector_index import file_name_boost
from app.reranker import BaseReranker, CrossEncoderReranker
from app.retriever.hybrid_retriever import HybridRetriever


class RerankRetriever:
    def __init__(self, index_manager: IndexManager, reranker: BaseReranker | None = None) -> None:
        self.index_manager = index_manager
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve_candidates(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> list[RetrievedChunk]:
        return HybridRetriever(self.index_manager).retrieve(query, top_k=candidate_k or max(top_k * 3, top_k))

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> list[RetrievedChunk]:
        candidates = self.retrieve_candidates(query, top_k=top_k, candidate_k=candidate_k)
        reranked = self.reranker.rerank(query, candidates, top_k=len(candidates))
        reranked = self._apply_document_prior(query, reranked)
        for rank, item in enumerate(reranked[:top_k], start=1):
            item.rank = rank
        return reranked[:top_k]

    def _apply_document_prior(self, query: str, documents: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self._query_mentions_known_file(query):
            return documents

        doc_counts: dict[str, int] = {}
        for chunk in self.index_manager.vector_index.chunks:
            doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1

        documents.sort(
            key=lambda item: (
                item.rerank_score
                + (10.0 * file_name_boost(query, item))
                + (min(doc_counts.get(item.doc_id, 1), 100) / 1000.0)
            ),
            reverse=True,
        )
        return documents

    def _query_mentions_known_file(self, query: str) -> bool:
        lowered_query = query.lower()
        return any(
            str(chunk.metadata.get("file_name", "")).lower() in lowered_query
            for chunk in self.index_manager.vector_index.chunks
            if chunk.metadata.get("file_name")
        )
