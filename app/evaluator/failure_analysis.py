from pathlib import Path
from typing import Any

from app.evaluator.metrics import is_unanswerable


def classify_failure(scored_item: dict[str, Any], latency_threshold_ms: float = 60000.0) -> str | None:
    sample = scored_item["sample"]
    result = scored_item["result"]
    metrics = scored_item["metrics"]

    if is_unanswerable(sample):
        return None if metrics["fallback_accuracy"] == 1.0 else "fallback_error"
    if metrics["hit_rate@5"] == 0.0:
        return "retrieval_miss"
    if metrics["citation_accuracy"] == 0.0:
        return "citation_error"
    if metrics["keyword_coverage"] < 0.3:
        return "keyword_miss"
    if result.get("latency_ms", 0.0) > latency_threshold_ms:
        return "latency_high"
    return None


def collect_failures(strategy_results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for strategy, items in strategy_results.items():
        for item in items:
            reason = classify_failure(item)
            if reason:
                result = item["result"]
                sample = item["sample"]
                failures.append(
                    {
                        "question_id": sample["id"],
                        "question": sample["question"],
                        "strategy": strategy,
                        "expected_doc": sample.get("expected_doc", ""),
                        "top_citations": [
                            citation.get("file_name", "unknown")
                            for citation in result.get("citations", [])[:3]
                        ],
                        "failure_reason": reason,
                    }
                )
    return failures


def write_failure_cases(failures: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Failure Cases", ""]
    if not failures:
        lines.extend(["No failure cases found.", ""])
    for failure in failures[:5]:
        lines.extend(
            [
                f"## {failure['question_id']} - {failure['strategy']}",
                "",
                f"- question: {failure['question']}",
                f"- expected_doc: {failure['expected_doc']}",
                f"- top citations: {', '.join(failure['top_citations']) or 'none'}",
                f"- failure_reason: {failure['failure_reason']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8-sig")
