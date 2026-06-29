from abc import ABC, abstractmethod

from app.core.schemas import DocumentChunk, ParsedDocument


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Split a parsed document into chunks."""
