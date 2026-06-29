from abc import ABC, abstractmethod

from app.core.schemas import RetrievedChunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Rerank retrieved candidate chunks."""
