import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.agent import run_agentic_rag
from app.indexer import IndexManager
from app.indexer.vector_index import INDEX_VERSION
from app.llm.citation_answer import build_citation_answer
from app.retriever import HybridRetriever, NaiveRetriever, RerankRetriever


def load_or_build_manager(chunk_strategy: str) -> IndexManager:
    namespace = chunk_strategy
    index_dir = Path("data/indexes") / namespace
    settings = get_settings()
    vector_index_path = index_dir / "vector_index.json"
    bm25_index_path = index_dir / "bm25_index.json"
    if vector_index_path.exists() and bm25_index_path.exists():
        metadata = json.loads(vector_index_path.read_text(encoding="utf-8"))
        if (
            metadata.get("index_version") == INDEX_VERSION
            and metadata.get("provider_name") == "sentence-transformers"
            and metadata.get("model_name") == settings.embedding_model
        ):
            return IndexManager.load(namespace=namespace)
        print(
            "Existing vector index is missing or uses a different embedding model; rebuilding index.",
            file=sys.stderr,
        )

    chunk_dir = Path("data/parsed/chunks") / chunk_strategy
    if not chunk_dir.exists():
        raise FileNotFoundError(
            f"Chunk directory not found: {chunk_dir}. Run scripts/ingest_docs.py first."
        )
    return IndexManager.from_chunk_dir(chunk_dir, namespace=namespace)


def ask(question: str, strategy: str, top_k: int, chunk_strategy: str) -> dict[str, object]:
    start = time.perf_counter()
    manager = load_or_build_manager(chunk_strategy)

    if strategy == "agentic":
        state = run_agentic_rag(question, index_manager=manager, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "strategy": "agentic",
            "latency_ms": round(latency_ms, 2),
            "retrieved_chunk_ids": state.get("retrieved_chunk_ids", []),
            "reranked_chunk_ids": state.get("reranked_chunk_ids", []),
            "rewritten_question": state.get("rewritten_question"),
            "route_trace": state.get("route_trace", []),
            "failure_reason": state.get("failure_reason"),
        }

    if strategy == "naive":
        retrieved = NaiveRetriever(manager).retrieve(question, top_k=top_k)
        reranked = []
    elif strategy == "hybrid":
        retrieved = HybridRetriever(manager).retrieve(question, top_k=top_k)
        reranked = []
    elif strategy == "rerank":
        retriever = RerankRetriever(manager)
        retrieved = retriever.retrieve_candidates(question, top_k=top_k)
        reranked = retriever.retrieve(question, top_k=top_k)
    else:
        raise ValueError("Stage 4 supports naive, hybrid, and rerank strategies.")

    answer_chunks = reranked if strategy == "rerank" else retrieved
    answer, citations = build_citation_answer(question, answer_chunks)
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "answer": answer,
        "citations": [citation.model_dump() for citation in citations],
        "strategy": strategy,
        "latency_ms": round(latency_ms, 2),
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved],
        "reranked_chunk_ids": [chunk.chunk_id for chunk in reranked],
        "rewritten_question": None,
        "route_trace": [],
        "failure_reason": None,
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ask a question against local DocuPilot-RAG indexes.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--strategy", choices=["naive", "hybrid", "rerank", "agentic"], default="hybrid")
    parser.add_argument("--top_k", type=int, default=settings.top_k)
    parser.add_argument("--chunk_strategy", choices=["sliding", "section"], default="section")
    args = parser.parse_args()

    result = ask(args.question, args.strategy, args.top_k, args.chunk_strategy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
