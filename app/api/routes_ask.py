from fastapi import APIRouter, HTTPException, status

from app.core.schemas import AskRequest, AskResponse
from scripts.ask_cli import ask as run_ask

router = APIRouter(tags=["ask"])

SUPPORTED_STRATEGIES = {"naive", "hybrid", "rerank", "agentic"}
SUPPORTED_CHUNK_STRATEGIES = {"section", "sliding"}


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question must not be empty",
        )
    if request.strategy not in SUPPORTED_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported strategy: {request.strategy}",
        )
    if request.chunk_strategy not in SUPPORTED_CHUNK_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported chunk_strategy: {request.chunk_strategy}",
        )

    try:
        result = run_ask(
            request.question,
            strategy=request.strategy,
            top_k=request.top_k,
            chunk_strategy=request.chunk_strategy,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"index or chunks not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if request.strategy == "agentic":
        result.setdefault("route_trace", [])
    return result
