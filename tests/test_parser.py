from pathlib import Path

import fitz

from app.parser import get_parser


def test_text_parser(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    document = get_parser(file_path).parse(file_path)

    assert document.file_type == "txt"
    assert document.pages[0].blocks[0].text == "First paragraph."


def test_markdown_parser(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Intro\n\nMarkdown body.", encoding="utf-8")

    document = get_parser(file_path).parse(file_path)

    assert document.file_type == "md"
    assert document.pages[0].blocks[0].block_type == "heading"
    assert document.pages[0].blocks[1].metadata["section"] == "Intro"


def test_pdf_parser(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "PDF body text")
    pdf.save(file_path)
    pdf.close()

    document = get_parser(file_path).parse(file_path)

    assert document.file_type == "pdf"
    assert document.pages[0].page_num == 1
    assert "PDF body text" in document.pages[0].text
