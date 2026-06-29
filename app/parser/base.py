from abc import ABC, abstractmethod
import hashlib
from pathlib import Path

from app.core.schemas import ParsedDocument


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse a source file into the normalized ParsedDocument schema."""


def build_doc_id(file_path: str | Path) -> str:
    path = Path(file_path)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"
