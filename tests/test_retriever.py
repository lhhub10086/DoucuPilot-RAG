from app.core.schemas import DocumentChunk
from app.indexer.embedding import HashingEmbeddingProvider
from app.indexer import IndexManager
from app.retriever import HybridRetriever, NaiveRetriever


def build_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="c1",
            doc_id="doc1",
            text="PDF parser preserves page number metadata.",
            metadata={"file_name": "parser.pdf", "page_num": 1, "section": "Parser"},
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="doc2",
            text="Hybrid search combines vector search and BM25 keyword retrieval.",
            metadata={"file_name": "search.md", "page_num": 1, "section": "Search"},
        ),
    ]


def test_naive_retriever_returns_metadata() -> None:
    manager = IndexManager(embedding_provider=HashingEmbeddingProvider())
    manager.add_documents(build_chunks())

    results = NaiveRetriever(manager).retrieve("page metadata", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["file_name"]
    assert results[0].chunk_id


def test_hybrid_retriever_returns_ranked_results() -> None:
    manager = IndexManager(embedding_provider=HashingEmbeddingProvider())
    manager.add_documents(build_chunks())

    results = HybridRetriever(manager).retrieve("BM25 keyword retrieval", top_k=2)

    assert results
    assert results[0].rank == 1
    assert results[0].score >= results[-1].score
