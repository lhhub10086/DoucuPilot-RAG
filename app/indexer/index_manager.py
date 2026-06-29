import json
from pathlib import Path

from app.core.config import get_settings
from app.core.schemas import DocumentChunk, RetrievedChunk
from app.indexer.bm25_index import BM25Index
from app.indexer.embedding import BaseEmbeddingProvider
from app.indexer.vector_index import VectorIndex


class IndexManager:
    def __init__(
        self,
        persist_dir: str | Path = "data/indexes",
        embedding_provider: BaseEmbeddingProvider | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.vector_index = VectorIndex(embedding_provider=embedding_provider)
        self.bm25_index = BM25Index()

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        self.vector_index.add_documents(chunks)
        self.bm25_index.add_documents(chunks)

    def search_vector(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.vector_index.search(query, top_k=top_k)

    def search_bm25(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.bm25_index.search(query, top_k=top_k)

    def search_hybrid(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        settings = get_settings()
        vector_results = self.search_vector(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        return self._rrf_fusion([vector_results, bm25_results], top_k=top_k, rrf_k=settings.rrf_k)

    def save(self, namespace: str = "default") -> None:
        target_dir = self.persist_dir / namespace
        self.vector_index.save(target_dir / "vector_index.json")
        self.bm25_index.save(target_dir / "bm25_index.json")

    @classmethod
    def load(
        cls,
        namespace: str = "default",
        persist_dir: str | Path = "data/indexes",
        embedding_provider: BaseEmbeddingProvider | None = None,
    ) -> "IndexManager":
        manager = cls(persist_dir=persist_dir)
        target_dir = Path(persist_dir) / namespace
        manager.vector_index = VectorIndex.load(
            target_dir / "vector_index.json",
            embedding_provider=embedding_provider,
        )
        manager.bm25_index = BM25Index.load(target_dir / "bm25_index.json")
        return manager

    @classmethod
    def from_chunk_dir(
        cls,
        chunk_dir: str | Path,
        namespace: str = "default",
        persist_dir: str | Path = "data/indexes",
        embedding_provider: BaseEmbeddingProvider | None = None,
    ) -> "IndexManager":
        manager = cls(persist_dir=persist_dir, embedding_provider=embedding_provider)
        chunks = load_chunks(chunk_dir)
        manager.add_documents(chunks)
        manager.save(namespace=namespace)
        return manager

    @staticmethod
    def _rrf_fusion(
        result_sets: list[list[RetrievedChunk]],
        top_k: int,
        rrf_k: int,
    ) -> list[RetrievedChunk]:
        by_chunk_id: dict[str, RetrievedChunk] = {}
        scores: dict[str, float] = {}

        for results in result_sets:
            for rank, chunk in enumerate(results, start=1):
                by_chunk_id.setdefault(chunk.chunk_id, chunk)
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

        fused = []
        for chunk_id, score in scores.items():
            chunk = by_chunk_id[chunk_id]
            fused.append(
                RetrievedChunk(
                    **chunk.model_dump(exclude={"score", "rank", "retriever"}),
                    score=score,
                    retriever="hybrid",
                )
            )
        fused.sort(key=lambda item: item.score, reverse=True)
        for rank, item in enumerate(fused[:top_k], start=1):
            item.rank = rank
        return fused[:top_k]


def load_chunks(chunk_dir: str | Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in sorted(Path(chunk_dir).glob("*_chunks.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks.extend(DocumentChunk(**item) for item in data)
    return chunks
