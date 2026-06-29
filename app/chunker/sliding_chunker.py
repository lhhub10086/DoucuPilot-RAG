from app.chunker.base import BaseChunker
from app.core.schemas import DocumentChunk, ParsedDocument
from app.utils.text_utils import normalize_whitespace


class SlidingWindowChunker(BaseChunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        counter = 0
        step = self.chunk_size - self.chunk_overlap

        for page in document.pages:
            text = normalize_whitespace(page.text)
            if not text:
                continue
            start = 0
            while start < len(text):
                counter += 1
                piece = text[start : start + self.chunk_size].strip()
                if piece:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{document.doc_id}_p{page.page_num}_c{counter}",
                            doc_id=document.doc_id,
                            text=piece,
                            metadata={
                                "file_name": document.file_name,
                                "page_num": page.page_num,
                                "section": page.metadata.get("section", "unknown"),
                                "chunk_type": "sliding",
                            },
                        )
                    )
                start += step

        return chunks
