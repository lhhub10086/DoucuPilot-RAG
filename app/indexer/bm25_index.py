import json
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from app.core.schemas import DocumentChunk, RetrievedChunk
from app.indexer.vector_index import build_searchable_text, file_name_boost

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    words = [word.strip().lower() for word in jieba.lcut(text) if word.strip()]
    tokens: list[str] = []
    for word in words:
        tokens.extend(TOKEN_RE.findall(word) or [word])
    return tokens


class BM25Index:
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.corpus_tokens: list[list[str]] = []
        self.index: BM25Okapi | None = None

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        self.chunks.extend(chunks)
        self.corpus_tokens.extend(tokenize(build_searchable_text(chunk)) for chunk in chunks)
        self.index = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.index is None:
            return []

        scores = self.index.get_scores(tokenize(query))
        boosted_scores = [
            (index, float(score) + (10.0 * file_name_boost(query, self.chunks[index])))
            for index, score in enumerate(scores)
        ]
        ranked = sorted(boosted_scores, key=lambda item: item[1], reverse=True)
        results: list[RetrievedChunk] = []
        for rank, (chunk_index, score) in enumerate(ranked[:top_k], start=1):
            chunk = self.chunks[chunk_index]
            results.append(
                RetrievedChunk(
                    **chunk.model_dump(),
                    score=float(score),
                    rank=rank,
                    retriever="bm25",
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": [chunk.model_dump() for chunk in self.chunks]}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls()
        index.add_documents([DocumentChunk(**chunk) for chunk in payload["chunks"]])
        return index
