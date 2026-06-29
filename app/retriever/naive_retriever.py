from app.core.schemas import RetrievedChunk
from app.indexer import IndexManager


class NaiveRetriever:
    def __init__(self, index_manager: IndexManager) -> None:
        self.index_manager = index_manager

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.index_manager.search_vector(query, top_k=top_k)
