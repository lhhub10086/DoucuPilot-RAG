from app.chunker.base import BaseChunker
from app.chunker.section_chunker import SectionChunker
from app.chunker.sliding_chunker import SlidingWindowChunker


def get_chunker(strategy: str, chunk_size: int = 800, chunk_overlap: int = 120) -> BaseChunker:
    if strategy == "sliding":
        return SlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "section":
        return SectionChunker(max_chunk_size=chunk_size)
    raise ValueError(f"Unsupported chunk strategy: {strategy}")
