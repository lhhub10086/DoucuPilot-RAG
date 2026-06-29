import re
from pathlib import Path

from app.core.schemas import ParsedBlock, ParsedDocument, ParsedPage
from app.parser.base import BaseParser, build_doc_id

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


class MarkdownParser(BaseParser):
    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        blocks: list[ParsedBlock] = []
        current_section = "unknown"

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = HEADING_RE.match(stripped)
            if match:
                current_section = match.group(2).strip()
                blocks.append(
                    ParsedBlock(
                        text=current_section,
                        block_type="heading",
                        metadata={"level": len(match.group(1)), "section": current_section},
                    )
                )
            else:
                blocks.append(
                    ParsedBlock(
                        text=stripped,
                        block_type="paragraph",
                        metadata={"section": current_section},
                    )
                )

        return ParsedDocument(
            doc_id=build_doc_id(path),
            file_name=path.name,
            file_type="md",
            pages=[ParsedPage(page_num=1, text=text, blocks=blocks, metadata={"section": "markdown"})],
            metadata={"title": blocks[0].text if blocks else path.stem, "source_path": str(path)},
        )
