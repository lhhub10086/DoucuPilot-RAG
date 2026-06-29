from fastapi import APIRouter, HTTPException, status

from app.core.schemas import AskRequest, AskResponse

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(_: AskRequest) -> AskResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Ask endpoint will be implemented in later RAG stages.",
    )
