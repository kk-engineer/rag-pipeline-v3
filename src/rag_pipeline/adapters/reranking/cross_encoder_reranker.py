from __future__ import annotations

import math

from sentence_transformers import CrossEncoder

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.core.models import RerankedResult, RetrievalResult
from rag_pipeline.utils.model_loading import resolve_model_path

logger = get_logger()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:

    def __init__(self, model_name: str = ""):
        settings = get_settings()
        self._model_name = model_name or settings.models.reranker
        model_path = resolve_model_path(settings, self._model_name)
        self._model = CrossEncoder(model_path)
        logger.info(f"CrossEncoderReranker model={self._model_name}")

    def rerank(self, query: str, results: RetrievalResult) -> RerankedResult:
        if not results.chunks:
            return RerankedResult(chunks=[], scores=[], method="cross_encoder")

        pairs = [(query, c.content) for c in results.chunks]
        logits = self._model.predict(pairs)
        scores = [_sigmoid(float(s)) for s in logits]
        scored = sorted(zip(results.chunks, scores), key=lambda x: x[1], reverse=True)
        logger.info(f"Reranked {len(scored)} chunks, top score={scored[0][1]:.4f}")
        return RerankedResult(
            chunks=[s[0] for s in scored],
            scores=[float(s[1]) for s in scored],
            method="cross_encoder",
        )
