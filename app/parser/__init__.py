from pathlib import Path

from app.parser.base import BaseParser
from app.parser.markdown_parser import MarkdownParser
from app.parser.pdf_parser import PDFParser
from app.parser.text_parser import TextParser


def get_parser(file_path: str | Path) -> BaseParser:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return PDFParser()
    if suffix in {".md", ".markdown"}:
        return MarkdownParser()
    if suffix == ".txt":
        return TextParser()
    raise ValueError(f"Unsupported file type: {suffix}")
