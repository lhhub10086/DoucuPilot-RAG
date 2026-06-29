from app.core.config import get_settings
from app.core.schemas import RetrievedChunk
from app.indexer.vector_index import build_searchable_text
from app.reranker.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    provider_name = "sentence-transformers-cross-encoder"

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, documents: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not documents:
            return []

        pairs = [(query, build_searchable_text(document)) for document in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        reranked: list[RetrievedChunk] = []
        for document, score in zip(documents, scores, strict=True):
            reranked.append(
                RetrievedChunk(
                    **document.model_dump(exclude={"rerank_score", "rank", "retriever"}),
                    rerank_score=float(score),
                    retriever="rerank",
                )
            )

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        for rank, item in enumerate(reranked[:top_k], start=1):
            item.rank = rank
        return reranked[:top_k]
