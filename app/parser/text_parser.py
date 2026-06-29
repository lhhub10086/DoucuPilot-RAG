from pathlib import Path

from app.core.schemas import ParsedBlock, ParsedDocument, ParsedPage
from app.parser.base import BaseParser, build_doc_id


class TextParser(BaseParser):
    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        blocks = [ParsedBlock(text=paragraph, metadata={"section": "unknown"}) for paragraph in paragraphs]

        return ParsedDocument(
            doc_id=build_doc_id(path),
            file_name=path.name,
            file_type="txt",
            pages=[ParsedPage(page_num=1, text=text, blocks=blocks, metadata={})],
            metadata={"title": path.stem, "source_path": str(path)},
        )
