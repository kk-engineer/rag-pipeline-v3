from __future__ import annotations

import logging

from rag_pipeline.config.settings import get_settings
from rag_pipeline.core.models import RerankedResult

logger = logging.getLogger(__name__)


class CircuitBreaker:

    def __init__(self, threshold: float | None = None):
        settings = get_settings()
        self._threshold = threshold if threshold is not None else settings.thresholds.retrieval_circuit_breaker
        logger.info("CircuitBreaker threshold=%s", self._threshold)

    def check(self, reranked: RerankedResult) -> dict:
        top_score = reranked.scores[0] if reranked.scores else 0.0
        passed = top_score >= self._threshold
        if not passed:
            logger.warning("Circuit breaker triggered: top_score=%.3f < threshold=%.3f", top_score, self._threshold)
        return {
            "passed": passed,
            "top_score": top_score,
            "threshold": self._threshold,
            "reason": "" if passed else f"Circuit breaker: top score {top_score:.3f} < threshold {self._threshold}",
        }
