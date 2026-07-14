from __future__ import annotations

import logging

import chromadb
from rank_bm25 import BM25Okapi

from rag_pipeline.config.settings import get_settings
from rag_pipeline.core.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


class BM25Retriever:

    def __init__(self):
        self._corpus: list[str] = []
        self._chunks: list[Chunk] = []
        self._bm25 = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._corpus = [c.content for c in self._chunks]
        self._bm25 = BM25Okapi([doc.split() for doc in self._corpus])
        logger.info("BM25 indexed %s chunks", len(chunks))

    def retrieve(self, query: str, k: int = 0) -> RetrievalResult:
        if not self._bm25:
            return RetrievalResult(chunks=[], scores=[], method="bm25")

        settings = get_settings()
        top_k = k if k > 0 else settings.thresholds.default_retrieval_k
        scores = self._bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        chunks = [self._chunks[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        logger.info("BM25 retrieved %s chunks for query=%s", len(chunks), query[:50])
        return RetrievalResult(chunks=chunks, scores=top_scores, method="bm25")


_bm25_instance: BM25Retriever | None = None


def get_bm25() -> BM25Retriever:
    global _bm25_instance
    if _bm25_instance is not None:
        return _bm25_instance

    settings = get_settings()
    client = chromadb.PersistentClient(
        path=settings.paths.chroma_db_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=settings.paths.chroma_collection_name)
    data = collection.get(include=["documents", "metadatas"])

    chunks = []
    if data and data.get("ids"):
        default_tenant = settings.machine.default_tenant_id
        for i, doc_id in enumerate(data["ids"]):
            content = data["documents"][i] if data.get("documents") else ""
            meta = data["metadatas"][i] if data.get("metadatas") else {}
            chunks.append(Chunk(
                chunk_id=doc_id,
                content=content,
                doc_id=meta.get("doc_id", ""),
                page_number=meta.get("page_number", 0),
                chunk_index=meta.get("chunk_index", 0),
                filename=meta.get("filename", ""),
                tenant_id=meta.get("tenant_id", default_tenant),
                doc_version_uuid=meta.get("doc_version_uuid", ""),
            ))

    _bm25_instance = BM25Retriever()
    _bm25_instance.index(chunks)
    logger.info("BM25 built from Chroma corpus: %s chunks", len(chunks))
    return _bm25_instance
