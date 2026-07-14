from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline.config.settings import get_settings

logger = logging.getLogger(__name__)


class IngestionLedger:

    def __init__(self, db_path: str = ""):
        settings = get_settings()
        self._path = Path(db_path or settings.paths.ledger_db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                filename TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                doc_version_uuid TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0
            )
        """)
        self._conn.commit()
        logger.info("IngestionLedger initialised; path=%s", self._path)

    @staticmethod
    def compute_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def is_ingested(self, filename: str, file_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT file_hash FROM documents WHERE filename = ?", (filename,)
        ).fetchone()
        return row is not None and row[0] == file_hash

    def record_ingestion(self, filename: str, file_hash: str, doc_version_uuid: str, chunk_count: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (filename, file_hash, doc_version_uuid, ingested_at, chunk_count) VALUES (?, ?, ?, ?, ?)",
            (filename, file_hash, doc_version_uuid, datetime.now(timezone.utc).isoformat(), chunk_count),
        )
        self._conn.commit()
        logger.info("Recorded ingestion: %s (%s chunks, hash=%s)", filename, chunk_count, file_hash[:12])

    def get_status(self, filename: str) -> dict | None:
        row = self._conn.execute(
            "SELECT filename, file_hash, doc_version_uuid, ingested_at, chunk_count FROM documents WHERE filename = ?",
            (filename,),
        ).fetchone()
        if row is None:
            return None
        return {"filename": row[0], "file_hash": row[1], "doc_version_uuid": row[2],
                "ingested_at": row[3], "chunk_count": row[4]}
