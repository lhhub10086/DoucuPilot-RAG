from pathlib import Path

from app.core.schemas import ParsedDocument
from app.parser.base import BaseParser


class MinerUAdapter(BaseParser):
    def parse(self, file_path: str | Path) -> ParsedDocument:
        raise NotImplementedError(
            "MinerU integration is reserved for the enhanced parser stage. "
            "Use PDFParser for the first runnable version."
        )
