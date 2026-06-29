import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"id", "question", "expected_doc", "expected_keywords"}


def load_eval_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {dataset_path}")

    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {dataset_path}:{line_number}: {exc}") from exc

        missing = REQUIRED_FIELDS - set(sample)
        if missing:
            raise ValueError(
                f"Missing required fields at {dataset_path}:{line_number}: {sorted(missing)}"
            )
        if not isinstance(sample["expected_keywords"], list):
            raise ValueError(f"expected_keywords must be a list at {dataset_path}:{line_number}")
        samples.append(sample)

    return samples
