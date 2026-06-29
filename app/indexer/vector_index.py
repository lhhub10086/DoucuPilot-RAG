import json
from pathlib import Path

from app.core.config import get_settings
from app.core.schemas import DocumentChunk, RetrievedChunk
from app.indexer.embedding import (
    BaseEmbeddingProvider,
    HashingEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)

INDEX_VERSION = "metadata-aware-v1"


def build_searchable_text(chunk: DocumentChunk) -> str:
    return " ".join(
        [
            str(chunk.metadata.get("file_name", "")),
            str(chunk.metadata.get("section", "")),
            chunk.text,
        ]
    ).strip()


def file_name_boost(query: str, chunk: DocumentChunk) -> float:
    file_name = str(chunk.metadata.get("file_name", "")).lower()
    return 1.0 if file_name and file_name in query.lower() else 0.0


class VectorIndex:
    def __init__(self, embedding_provider: BaseEmbeddingProvider | None = None) -> None:
        settings = get_settings()
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            settings.embedding_model
        )
        self.chunks: list[DocumentChunk] = []
        self.vectors: list[list[float]] = []

    @property
    def provider_name(self) -> str:
        return self.embedding_provider.provider_name

    @property
    def model_name(self) -> str:
        return self.embedding_provider.model_name

    @property
    def dimensions(self) -> int:
        return self.embedding_provider.dimensions

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        self.chunks.extend(chunks)
        self.vectors.extend(self.embedding_provider.embed(build_searchable_text(chunk)) for chunk in chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.embedding_provider.embed(query)
        scored: list[RetrievedChunk] = []
        for chunk, vector in zip(self.chunks, self.vectors, strict=True):
            score = self._cosine(query_vector, vector) + file_name_boost(query, chunk)
            scored.append(
                RetrievedChunk(
                    **chunk.model_dump(),
                    score=score,
                    retriever="vector",
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        for rank, item in enumerate(scored[:top_k], start=1):
            item.rank = rank
        return scored[:top_k]

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_version": INDEX_VERSION,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "chunks": [chunk.model_dump() for chunk in self.chunks],
            "vectors": self.vectors,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embedding_provider: BaseEmbeddingProvider | None = None) -> "VectorIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        provider = embedding_provider
        if provider is None:
            if payload.get("provider_name") == "hashing":
                provider = HashingEmbeddingProvider(dimensions=payload["dimensions"])
            else:
                provider = SentenceTransformerEmbeddingProvider(payload["model_name"])
        index = cls(provider)
        index.chunks = [DocumentChunk(**chunk) for chunk in payload["chunks"]]
        index.vectors = payload["vectors"]
        return index

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))
