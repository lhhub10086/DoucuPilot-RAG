from statistics import mean
from typing import Any

NO_ANSWER = "当前文档中未找到充分依据"


def _normalize(value: str | None) -> str:
    return (value or "").lower()


def _citation_docs(result: dict[str, Any], k: int | None = None) -> list[str]:
    citations = result.get("citations", [])
    if k is not None:
        citations = citations[:k]
    return [_normalize(citation.get("file_name")) for citation in citations]


def _candidate_docs(result: dict[str, Any], k: int | None = None) -> list[str]:
    docs = _citation_docs(result, k=k)
    retrieved = result.get("retrieved_documents", [])
    reranked = result.get("reranked_documents", [])
    for item in reranked + retrieved:
        file_name = item.get("metadata", {}).get("file_name") if isinstance(item, dict) else None
        docs.append(_normalize(file_name))
    return docs[:k] if k is not None else docs


def hit_rate_at_k(result: dict[str, Any], expected_doc: str, k: int = 5) -> float:
    if not expected_doc:
        return 0.0
    expected = _normalize(expected_doc)
    return 1.0 if any(expected == doc for doc in _candidate_docs(result, k=k)) else 0.0


def reciprocal_rank(result: dict[str, Any], expected_doc: str) -> float:
    if not expected_doc:
        return 0.0
    expected = _normalize(expected_doc)
    for rank, doc in enumerate(_candidate_docs(result), start=1):
        if doc == expected:
            return 1.0 / rank
    return 0.0


def citation_accuracy(result: dict[str, Any], expected_doc: str) -> float:
    if not expected_doc:
        return 0.0
    expected = _normalize(expected_doc)
    return 1.0 if any(expected == doc for doc in _citation_docs(result)) else 0.0


def keyword_coverage(result: dict[str, Any], expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    haystack = _normalize(result.get("answer", ""))
    for citation in result.get("citations", []):
        haystack += " " + _normalize(citation.get("text_preview", ""))
    hits = sum(1 for keyword in expected_keywords if _normalize(keyword) in haystack)
    return hits / len(expected_keywords)


def is_unanswerable(sample: dict[str, Any]) -> bool:
    return sample.get("question_type") == "unanswerable"


def fallback_accuracy(result: dict[str, Any], sample: dict[str, Any]) -> float:
    if not is_unanswerable(sample):
        return 0.0
    return 1.0 if NO_ANSWER in result.get("answer", "") else 0.0


def average_latency(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return mean(float(result.get("latency_ms", 0.0)) for result in results)


def score_sample(result: dict[str, Any], sample: dict[str, Any], k: int = 5) -> dict[str, float]:
    return {
        "hit_rate@5": hit_rate_at_k(result, sample.get("expected_doc", ""), k=k),
        "mrr": reciprocal_rank(result, sample.get("expected_doc", "")),
        "citation_accuracy": citation_accuracy(result, sample.get("expected_doc", "")),
        "keyword_coverage": keyword_coverage(result, sample.get("expected_keywords", [])),
        "fallback_accuracy": fallback_accuracy(result, sample),
        "latency_ms": float(result.get("latency_ms", 0.0)),
    }


def summarize_strategy(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    if not scored_results:
        return {
            "hit_rate@5": 0.0,
            "mrr": 0.0,
            "citation_accuracy": 0.0,
            "keyword_coverage": 0.0,
            "fallback_accuracy_on_unanswerable": 0.0,
            "avg_latency_ms": 0.0,
            "answerable_count": 0.0,
            "unanswerable_count": 0.0,
            "unanswerable_fallback_count": 0.0,
        }

    answerable = [item for item in scored_results if not is_unanswerable(item["sample"])]
    unanswerable = [item for item in scored_results if is_unanswerable(item["sample"])]
    answerable_metrics = [item["metrics"] for item in answerable]
    unanswerable_metrics = [item["metrics"] for item in unanswerable]
    all_metrics = [item["metrics"] for item in scored_results]

    def metric_mean(name: str, items: list[dict[str, float]]) -> float:
        return mean(item[name] for item in items) if items else 0.0

    return {
        "hit_rate@5": metric_mean("hit_rate@5", answerable_metrics),
        "mrr": metric_mean("mrr", answerable_metrics),
        "citation_accuracy": metric_mean("citation_accuracy", answerable_metrics),
        "keyword_coverage": metric_mean("keyword_coverage", answerable_metrics),
        "fallback_accuracy_on_unanswerable": metric_mean(
            "fallback_accuracy", unanswerable_metrics
        ),
        "avg_latency_ms": metric_mean("latency_ms", all_metrics),
        "answerable_count": float(len(answerable)),
        "unanswerable_count": float(len(unanswerable)),
        "unanswerable_fallback_count": sum(
            item["fallback_accuracy"] for item in unanswerable_metrics
        ),
    }


def compute_deepeval_metrics(*_: Any, **__: Any) -> dict[str, Any]:
    return {"status": "skipped", "reason": "DeepEval is not integrated in Stage 6."}


def compute_ragas_metrics(*_: Any, **__: Any) -> dict[str, Any]:
    return {"status": "skipped", "reason": "RAGAS is not integrated in Stage 6."}
