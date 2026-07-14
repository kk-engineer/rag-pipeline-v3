from __future__ import annotations

from pathlib import Path

from rag_pipeline.core.models import Chunk


def build_source_list(chunks: list[Chunk], verified_chunk_ids: set[str]) -> str:
    verified = [c for c in chunks if c.chunk_id in verified_chunk_ids]
    if not verified:
        return ""

    seen = set()
    lines = []
    for i, c in enumerate(verified, 1):
        doc_name = c.filename or (Path(c.doc_id).name if c.doc_id else "?")
        key = (doc_name, c.page_number)
        if key not in seen:
            seen.add(key)
            lines.append(f"[{i}] ({doc_name}, page {c.page_number})")

    return "\n".join(f"{line}\n" for line in lines)
