import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.schemas import UploadResponse
from app.indexer import IndexManager
from app.utils.file_utils import SUPPORTED_EXTENSIONS
from scripts.ingest_docs import ingest_documents

router = APIRouter(tags=["ingest"])

SUPPORTED_CHUNK_STRATEGIES = {"section", "sliding"}
RAW_DIR = Path("data/raw")


@router.post("/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("section"),
    rebuild_index: bool = Form(True),
) -> UploadResponse:
    if chunk_strategy not in SUPPORTED_CHUNK_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported chunk_strategy: {chunk_strategy}",
        )

    file_name = Path(file.filename or "").name
    suffix = Path(file_name).suffix.lower()
    if not file_name or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported file type: {suffix or 'unknown'}",
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_path = RAW_DIR / file_name
    try:
        with target_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to save uploaded file: {exc}",
        ) from exc
    finally:
        file.file.close()

    try:
        parsed_count, chunk_count = ingest_documents(str(target_path), chunk_strategy)
        if rebuild_index:
            chunk_dir = Path("data/parsed/chunks") / chunk_strategy
            IndexManager.from_chunk_dir(chunk_dir, namespace=chunk_strategy)
            try:
                from scripts import ask_cli

                ask_cli._MANAGER_CACHE.pop(chunk_strategy, None)
            except Exception:
                pass
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to ingest uploaded file: {exc}",
        ) from exc

    if parsed_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded file could not be parsed",
        )

    return UploadResponse(
        status="indexed" if rebuild_index else "uploaded",
        file_name=file_name,
        chunk_strategy=chunk_strategy,
        num_documents=parsed_count,
        num_chunks=chunk_count,
        message="document uploaded and indexed successfully"
        if rebuild_index
        else "document uploaded and parsed successfully",
    )
