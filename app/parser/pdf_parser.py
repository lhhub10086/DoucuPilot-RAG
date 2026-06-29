import logging
from pathlib import Path

import fitz

from app.core.schemas import ParsedBlock, ParsedDocument, ParsedPage
from app.parser.base import BaseParser, build_doc_id

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        pages: list[ParsedPage] = []

        try:
            with fitz.open(path) as pdf:
                for index, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    blocks = [
                        ParsedBlock(text=line.strip(), metadata={"page_num": index})
                        for line in text.splitlines()
                        if line.strip()
                    ]
                    pages.append(
                        ParsedPage(
                            page_num=index,
                            text=text,
                            blocks=blocks,
                            metadata={"source": "pymupdf"},
                        )
                    )
        except Exception:
            logger.exception("Failed to parse PDF: %s", path)
            raise

        return ParsedDocument(
            doc_id=build_doc_id(path),
            file_name=path.name,
            file_type="pdf",
            pages=pages,
            metadata={"title": path.stem, "source_path": str(path)},
        )
