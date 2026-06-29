import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


def iter_document_files(input_path: str | Path) -> Iterable[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        return

    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield file_path


def write_json_model(model: BaseModel, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def write_json(data: object, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
