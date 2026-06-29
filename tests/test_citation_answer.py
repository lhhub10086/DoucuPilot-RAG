from app.core.schemas import RetrievedChunk
from app.llm.citation_answer import build_citation_answer


def test_low_confidence_rerank_returns_conservative_answer() -> None:
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="doc1",
        text="This chunk is unrelated to Mars migration.",
        metadata={"file_name": "paper.pdf", "page_num": 1, "section": "Intro"},
        score=0.01,
        rerank_score=0.00001,
        retriever="rerank",
    )

    answer, citations = build_citation_answer("有没有提到火星移民计划？", [chunk])

    assert answer == "当前文档中未找到充分依据。"
    assert citations[0].chunk_id == "c1"


def test_citation_contains_standard_fields() -> None:
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="doc1",
        text="A relevant evidence chunk.",
        metadata={"file_name": "paper.pdf", "page_num": 1, "section": "Method"},
        score=0.5,
        rerank_score=0.2,
        retriever="rerank",
    )

    _, citations = build_citation_answer("核心方法是什么？", [chunk])
    data = citations[0].model_dump()

    assert {
        "file_name",
        "page_num",
        "section",
        "chunk_id",
        "text_preview",
        "score",
        "rerank_score",
    }.issubset(data.keys())
