import json
from pathlib import Path

from app.evaluator.dataset import load_eval_dataset
from app.evaluator.metrics import (
    citation_accuracy,
    fallback_accuracy,
    hit_rate_at_k,
    keyword_coverage,
    reciprocal_rank,
)
from app.evaluator.run_eval import run_evaluation


def sample_result() -> dict:
    return {
        "answer": "CMD3 uses Cross-Modal Decoupled Deformable Distillation.",
        "citations": [
            {
                "file_name": "paper.pdf",
                "text_preview": "Cross-Modal Decoupled Deformable Distillation for EEG-fNIRS.",
            }
        ],
        "retrieved_documents": [
            {"metadata": {"file_name": "paper.pdf"}},
        ],
        "reranked_documents": [],
        "latency_ms": 123.0,
    }


def test_load_eval_dataset_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "question",
                "expected_doc": "paper.pdf",
                "expected_keywords": ["CMD3"],
            }
        )
        + "\n\n",
        encoding="utf-8",
    )

    samples = load_eval_dataset(path)

    assert len(samples) == 1
    assert samples[0]["id"] == "q1"


def test_metrics() -> None:
    result = sample_result()

    assert hit_rate_at_k(result, "paper.pdf") == 1.0
    assert reciprocal_rank(result, "paper.pdf") == 1.0
    assert citation_accuracy(result, "paper.pdf") == 1.0
    assert keyword_coverage(result, ["CMD3", "EEG-fNIRS"]) == 1.0
    assert fallback_accuracy(
        {"answer": "当前文档中未找到充分依据。"},
        {"expected_doc": "", "question_type": "unanswerable"},
    ) == 1.0
    assert fallback_accuracy(
        {"answer": "当前文档中未找到充分依据。"},
        {"expected_doc": "paper.pdf", "question_type": "fact"},
    ) == 0.0


def test_run_eval_small_sample(monkeypatch, tmp_path: Path) -> None:
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "CMD3?",
                "expected_doc": "paper.pdf",
                "expected_keywords": ["CMD3"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_ask(question: str, strategy: str, top_k: int, chunk_strategy: str) -> dict:
        result = sample_result()
        result["strategy"] = strategy
        result["retrieved_chunk_ids"] = ["c1"]
        result["reranked_chunk_ids"] = []
        return result

    import scripts.ask_cli

    monkeypatch.setattr(scripts.ask_cli, "ask", fake_ask)
    monkeypatch.chdir(tmp_path)

    result = run_evaluation(str(eval_file), ["naive"], chunk_strategy="section")

    assert result["summary"]["naive"]["hit_rate@5"] == 1.0
    assert result["summary"]["naive"]["fallback_accuracy_on_unanswerable"] == 0.0
    assert Path(result["report_path"]).exists()
