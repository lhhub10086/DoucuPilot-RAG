from app.chunker.section_chunker import SectionChunker
from app.chunker.sliding_chunker import SlidingWindowChunker
from app.core.schemas import ParsedBlock, ParsedDocument, ParsedPage


def build_document() -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc1",
        file_name="sample.md",
        file_type="md",
        pages=[
            ParsedPage(
                page_num=1,
                text="Title body " * 30,
                blocks=[
                    ParsedBlock(text="Intro", block_type="heading", metadata={"section": "Intro"}),
                    ParsedBlock(text="Body content", metadata={"section": "Intro"}),
                ],
            )
        ],
        metadata={"title": "sample", "source_path": "sample.md"},
    )


def test_sliding_chunker_metadata() -> None:
    chunks = SlidingWindowChunker(chunk_size=40, chunk_overlap=10).chunk(build_document())

    assert chunks
    assert chunks[0].metadata["file_name"] == "sample.md"
    assert chunks[0].metadata["page_num"] == 1
    assert chunks[0].metadata["chunk_type"] == "sliding"


def test_section_chunker_metadata() -> None:
    chunks = SectionChunker().chunk(build_document())

    assert len(chunks) == 1
    assert chunks[0].metadata["section"] == "Intro"
    assert chunks[0].metadata["chunk_type"] == "section"
