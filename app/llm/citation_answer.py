from app.core.schemas import Citation, RetrievedChunk

MIN_RERANK_CONFIDENCE = 0.0001


def build_citation_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    if not chunks:
        return "当前文档中未找到充分依据。", []

    citations = [
        Citation(
            file_name=chunk.metadata.get("file_name", "unknown"),
            page_num=chunk.metadata.get("page_num"),
            section=chunk.metadata.get("section", "unknown"),
            chunk_id=chunk.chunk_id,
            text_preview=chunk.text[:160],
            score=round(float(chunk.score), 6),
            rerank_score=round(float(chunk.rerank_score), 6),
        )
        for chunk in chunks
    ]
    has_rerank_scores = any(chunk.rerank_score != 0.0 for chunk in chunks)
    if has_rerank_scores and max(chunk.rerank_score for chunk in chunks) < MIN_RERANK_CONFIDENCE:
        return "当前文档中未找到充分依据。", citations

    evidence = " ".join(chunk.text.strip() for chunk in chunks[:2] if chunk.text.strip())
    answer = (
        f"基于检索与排序结果，问题“{question}”的主要相关依据是：{evidence} "
        f"[来源: {citations[0].file_name}, section={citations[0].section}, chunk={citations[0].chunk_id}]"
    )
    return answer, citations
