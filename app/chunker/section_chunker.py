from app.chunker.base import BaseChunker
from app.core.schemas import DocumentChunk, ParsedBlock, ParsedDocument
from app.utils.text_utils import normalize_whitespace


class SectionChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 1200) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        counter = 0

        for page in document.pages:
            section = page.metadata.get("section", "unknown")
            buffer: list[str] = []

            for block in page.blocks or [ParsedBlock(text=page.text)]:
                if block.block_type == "heading":
                    if buffer:
                        counter = self._flush(document, page.page_num, section, buffer, chunks, counter)
                        buffer = []
                    section = block.text
                    continue

                text = normalize_whitespace(block.text)
                if not text:
                    continue
                if sum(len(item) for item in buffer) + len(text) > self.max_chunk_size and buffer:
                    counter = self._flush(document, page.page_num, section, buffer, chunks, counter)
                    buffer = []
                buffer.append(text)

            if buffer:
                counter = self._flush(document, page.page_num, section, buffer, chunks, counter)

        return chunks

    @staticmethod
    def _flush(
        document: ParsedDocument,
        page_num: int,
        section: str,
        buffer: list[str],
        chunks: list[DocumentChunk],
        counter: int,
    ) -> int:
        counter += 1
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.doc_id}_p{page_num}_s{counter}",
                doc_id=document.doc_id,
                text="\n".join(buffer).strip(),
                metadata={
                    "file_name": document.file_name,
                    "page_num": page_num,
                    "section": section or "unknown",
                    "chunk_type": "section",
                },
            )
        )
        return counter
