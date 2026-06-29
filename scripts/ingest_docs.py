import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.chunker import get_chunker
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.parser import get_parser
from app.utils.file_utils import iter_document_files, write_json, write_json_model


logger = logging.getLogger(__name__)


def ingest_documents(input_path: str, chunk_strategy: str) -> tuple[int, int]:
    settings = get_settings()
    output_dir = Path("data/parsed")
    chunks_dir = output_dir / "chunks" / chunk_strategy
    parsed_count = 0
    chunk_count = 0

    chunker = get_chunker(
        chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    for file_path in iter_document_files(input_path):
        try:
            document = get_parser(file_path).parse(file_path)
            chunks = chunker.chunk(document)
            write_json_model(document, output_dir / f"{document.doc_id}.json")
            write_json(
                [chunk.model_dump() for chunk in chunks],
                chunks_dir / f"{document.doc_id}_chunks.json",
            )
            parsed_count += 1
            chunk_count += len(chunks)
            logger.info("Parsed %s into %s chunks", file_path, len(chunks))
        except Exception as exc:
            logger.exception("Failed to ingest %s: %s", file_path, exc)

    return parsed_count, chunk_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse documents and generate chunks.")
    parser.add_argument("--input", default="data/raw", help="Input file or directory.")
    parser.add_argument(
        "--chunk_strategy",
        choices=["sliding", "section"],
        default="sliding",
        help="Chunking strategy.",
    )
    args = parser.parse_args()

    setup_logging()
    parsed_count, chunk_count = ingest_documents(args.input, args.chunk_strategy)
    print(f"parsed documents: {parsed_count}")
    print(f"generated chunks: {chunk_count}")


if __name__ == "__main__":
    main()
