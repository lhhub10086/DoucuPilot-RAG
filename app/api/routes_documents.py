import json
from pathlib import Path

from fastapi import APIRouter

from app.core.schemas import DocumentInfo, DocumentsResponse
from app.utils.file_utils import SUPPORTED_EXTENSIONS

router = APIRouter(tags=["documents"])

RAW_DIR = Path("data/raw")
PARSED_DIR = Path("data/parsed")


@router.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    parsed_by_name = _load_parsed_metadata()
    documents: list[DocumentInfo] = []

    if RAW_DIR.exists():
        for path in sorted(RAW_DIR.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            parsed = parsed_by_name.get(path.name, {})
            documents.append(
                DocumentInfo(
                    file_name=path.name,
                    file_type=path.suffix.lower().lstrip("."),
                    source_path=str(path),
                    parsed=bool(parsed),
                    doc_id=parsed.get("doc_id"),
                    page_count=parsed.get("page_count"),
                )
            )

    return DocumentsResponse(documents=documents, count=len(documents))


def _load_parsed_metadata() -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    if not PARSED_DIR.exists():
        return metadata

    for path in sorted(PARSED_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        file_name = data.get("file_name")
        if not file_name:
            continue
        metadata[file_name] = {
            "doc_id": data.get("doc_id"),
            "page_count": len(data.get("pages", [])),
        }
    return metadata
