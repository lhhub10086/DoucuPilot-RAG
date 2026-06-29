import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.evaluator.dataset import load_eval_dataset
from app.evaluator.failure_analysis import collect_failures, write_failure_cases
from app.evaluator.metrics import is_unanswerable, score_sample, summarize_strategy


def run_evaluation(
    eval_file: str,
    strategies: list[str],
    chunk_strategy: str = "section",
) -> dict[str, Any]:
    from scripts.ask_cli import ask

    dataset = load_eval_dataset(eval_file)
    strategy_results: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, dict[str, float]] = {}

    for strategy in strategies:
        scored_items: list[dict[str, Any]] = []
        for sample in dataset:
            result = ask(
                sample["question"],
                strategy=strategy,
                top_k=get_settings().top_k,
                chunk_strategy=chunk_strategy,
            )
            metrics = score_sample(result, sample, k=5)
            scored_items.append({"sample": sample, "result": result, "metrics": metrics})
        strategy_results[strategy] = scored_items
        summary[strategy] = summarize_strategy(scored_items)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"eval_{timestamp}.json"
    csv_path = output_dir / f"eval_{timestamp}.csv"

    payload = {
        "eval_file": eval_file,
        "chunk_strategy": chunk_strategy,
        "strategies": strategies,
        "summary": summary,
        "results": strategy_results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, strategy_results)

    failures = collect_failures(strategy_results)
    answerable_count = sum(1 for sample in dataset if not is_unanswerable(sample))
    unanswerable_count = sum(1 for sample in dataset if is_unanswerable(sample))
    write_eval_report(
        output_path="docs/eval_report.md",
        eval_file=eval_file,
        chunk_strategy=chunk_strategy,
        sample_count=len(dataset),
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
        strategies=strategies,
        summary=summary,
        failures=failures,
    )
    write_failure_cases(failures, "docs/failure_cases.md")

    return {
        "summary": summary,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "report_path": "docs/eval_report.md",
        "failure_cases_path": "docs/failure_cases.md",
        "failures": failures,
    }


def write_csv(path: Path, strategy_results: dict[str, list[dict[str, Any]]]) -> None:
    fieldnames = [
        "strategy",
        "question_id",
        "question",
        "hit_rate@5",
        "mrr",
        "citation_accuracy",
        "keyword_coverage",
        "fallback_accuracy",
        "latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for strategy, items in strategy_results.items():
            for item in items:
                row = {
                    "strategy": strategy,
                    "question_id": item["sample"]["id"],
                    "question": item["sample"]["question"],
                    **item["metrics"],
                }
                writer.writerow(row)


def write_eval_report(
    output_path: str | Path,
    eval_file: str,
    chunk_strategy: str,
    sample_count: int,
    answerable_count: int,
    unanswerable_count: int,
    strategies: list[str],
    summary: dict[str, dict[str, float]],
    failures: list[dict[str, Any]],
) -> None:
    settings = get_settings()
    lines = [
        "# Evaluation Report",
        "",
        "## Evaluation Setup",
        "",
        f"- documents: local `data/raw` corpus",
        f"- chunk_strategy: `{chunk_strategy}`",
        f"- eval_file: `{eval_file}`",
        f"- eval_samples: {sample_count}",
        f"- answerable_samples: {answerable_count}",
        f"- unanswerable_samples: {unanswerable_count}",
        f"- strategies: {', '.join(strategies)}",
        f"- embedding_model: `{settings.embedding_model}`",
        f"- reranker_model: `{settings.reranker_model}`",
        "",
        "## Strategy Comparison",
        "",
        "| strategy | hit_rate@5 | mrr | citation_accuracy | keyword_coverage | fallback_accuracy_on_unanswerable | avg_latency_ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in strategies:
        item = summary[strategy]
        lines.append(
            f"| {strategy} | {item['hit_rate@5']:.3f} | {item['mrr']:.3f} | "
            f"{item['citation_accuracy']:.3f} | {item['keyword_coverage']:.3f} | "
            f"{item['fallback_accuracy_on_unanswerable']:.3f} | {item['avg_latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Unanswerable Performance",
            "",
            "| strategy | unanswerable_total | fallback_count | forced_answer_count | fallback_accuracy_on_unanswerable |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for strategy in strategies:
        item = summary[strategy]
        total = int(item["unanswerable_count"])
        fallback_count = int(item["unanswerable_fallback_count"])
        forced_count = max(total - fallback_count, 0)
        lines.append(
            f"| {strategy} | {total} | {fallback_count} | {forced_count} | "
            f"{item['fallback_accuracy_on_unanswerable']:.3f} |"
        )

    lines.extend(["", "## Observations", ""])
    lines.extend(build_observations(summary, strategies))
    lines.extend(["", "## Failure Cases", ""])
    if not failures:
        lines.append("No failure cases found.")
    for failure in failures[:5]:
        lines.extend(
            [
                f"### {failure['question_id']} - {failure['strategy']}",
                "",
                f"- question: {failure['question']}",
                f"- expected_doc: {failure['expected_doc']}",
                f"- top citations: {', '.join(failure['top_citations']) or 'none'}",
                f"- failure_reason: {failure['failure_reason']}",
                "",
            ]
        )
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")


def build_observations(summary: dict[str, dict[str, float]], strategies: list[str]) -> list[str]:
    observations: list[str] = []
    if "naive" in summary and "hybrid" in summary:
        delta = summary["hybrid"]["hit_rate@5"] - summary["naive"]["hit_rate@5"]
        observations.append(f"- hybrid hit_rate@5 vs naive: {delta:+.3f}.")
    if "hybrid" in summary and "rerank" in summary:
        delta = summary["rerank"]["citation_accuracy"] - summary["hybrid"]["citation_accuracy"]
        observations.append(f"- rerank citation_accuracy vs hybrid: {delta:+.3f}.")
    if "agentic" in summary:
        observations.append(
            "- agentic fallback_accuracy_on_unanswerable: "
            f"{summary['agentic']['fallback_accuracy_on_unanswerable']:.3f}."
        )
    if strategies:
        slowest = max(strategies, key=lambda item: summary[item]["avg_latency_ms"])
        observations.append(f"- highest average latency: {slowest}.")
    return observations


def print_summary_table(summary: dict[str, dict[str, float]], strategies: list[str]) -> None:
    print("strategy comparison table")
    print(
        "strategy, hit_rate@5, mrr, citation_accuracy, keyword_coverage, "
        "fallback_accuracy_on_unanswerable, avg_latency_ms"
    )
    for strategy in strategies:
        item = summary[strategy]
        print(
            f"{strategy}, {item['hit_rate@5']:.3f}, {item['mrr']:.3f}, "
            f"{item['citation_accuracy']:.3f}, {item['keyword_coverage']:.3f}, "
            f"{item['fallback_accuracy_on_unanswerable']:.3f}, {item['avg_latency_ms']:.1f}"
        )


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
