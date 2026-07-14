from __future__ import annotations

"""Ingest PDFs into the pipeline: extract, anonymize, chunk, and index."""

import sys
from pathlib import Path

from rag_pipeline.core.initializer import initialize
from rag_pipeline.core.pipeline.stages import ingest_pdf


def main() -> None:
    if not initialize():
        print("FATAL: RAG pipeline initialization failed. Exiting.")
        sys.exit(1)

    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/pdfs")
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    for pdf_path in pdf_files:
        print(f"Ingesting {pdf_path.name}...")
        result = ingest_pdf(pdf_path)
        print(f"  {result}")


if __name__ == "__main__":
    main()