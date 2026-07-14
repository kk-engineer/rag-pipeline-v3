from __future__ import annotations

import logging

from rag_pipeline.adapters.retrieval.bm25_retriever import BM25Retriever
from rag_pipeline.adapters.retrieval.chroma_retriever import ChromaRetriever
from rag_pipeline.config.settings import get_settings
from rag_pipeline.core.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:

    def __init__(self, dense_retriever: ChromaRetriever, bm25_retriever: BM25Retriever):
        self._dense = dense_retriever
        self._sparse = bm25_retriever
        settings = get_settings()
        self._rrf_k = settings.thresholds.rrf_constant

    def index(self, chunks: list[Chunk]) -> None:
        self._dense.index(chunks)
        self._sparse.index(chunks)

    def retrieve(self, query: str, k: int = 0) -> RetrievalResult:
        settings = get_settings()
        top_k = k if k > 0 else settings.thresholds.default_retrieval_k
        dense_r = self._dense.retrieve(query, k=top_k)
        sparse_r = self._sparse.retrieve(query, k=top_k)
        fused = self._rrf_fuse(dense_r, sparse_r, top_k)
        logger.info("Hybrid retrieved %s chunks (dense=%s, sparse=%s)", len(fused.chunks), len(dense_r.chunks), len(sparse_r.chunks))
        return fused

    def _rrf_fuse(self, dense: RetrievalResult, sparse: RetrievalResult, k: int) -> RetrievalResult:
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, tuple[Chunk, float]] = {}

        for rank, chunk in enumerate(dense.chunks):
            chunk_map[chunk.chunk_id] = (chunk, dense.scores[rank] if rank < len(dense.scores) else 0.0)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        for rank, chunk in enumerate(sparse.chunks):
            chunk_map[chunk.chunk_id] = (chunk, sparse.scores[rank] if rank < len(sparse.scores) else 0.0)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]
        return RetrievalResult(
            chunks=[chunk_map[cid][0] for cid in sorted_ids],
            scores=[rrf_scores[cid] for cid in sorted_ids],
            method="hybrid_rrf",
        )
