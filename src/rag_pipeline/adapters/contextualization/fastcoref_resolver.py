from __future__ import annotations

import logging

from rag_pipeline.config.settings import get_settings
from rag_pipeline.core.models import ContextualizedQuery

logger = logging.getLogger(__name__)


class FastCorefResolver:

    def __init__(self, model_name: str = ""):
        settings = get_settings()
        self._model_name = model_name or settings.models.coreference
        self._model = None
        logger.info("FastCorefResolver configured; model=%s", self._model_name)

    def _lazy_load(self):
        if self._model is None:
            from fastcoref import FCoref
            self._model = FCoref()
            logger.info("fastcoref model loaded")

    def contextualize(self, query: str, conversation_history: list[dict[str, str]] | None = None) -> ContextualizedQuery:
        if not conversation_history:
            return ContextualizedQuery(original_query=query, resolved_query=query, coref_confidence=1.0)

        self._lazy_load()
        full_text = self._build_conversation_text(query, conversation_history)
        preds = self._model.predict(texts=[full_text])
        resolved = self._apply_coref(query, preds)
        confidence = self._compute_confidence(preds)
        logger.info("Coref resolved: conf=%.3f, original=%s, resolved=%s", confidence, query[:50], resolved[:50])
        return ContextualizedQuery(
            original_query=query,
            resolved_query=resolved,
            coref_confidence=confidence,
            conversation_history=conversation_history,
        )

    def _build_conversation_text(self, query: str, history: list[dict[str, str]]) -> str:
        parts = []
        for turn in history:
            parts.append(f"User: {turn.get('user', '')}")
            parts.append(f"Assistant: {turn.get('assistant', '')}")
        parts.append(f"User: {query}")
        return "\n".join(parts)

    def _apply_coref(self, query: str, preds) -> str:
        try:
            clusters = preds[0].get_clusters()
            if not clusters:
                return query
            resolved = query
            for cluster in clusters:
                if len(cluster) > 1:
                    representative = cluster[-1]
                    for mention in cluster[:-1]:
                        resolved = resolved.replace(mention, representative, 1)
            return resolved
        except Exception:
            return query

    @staticmethod
    def _compute_confidence(preds) -> float:
        try:
            clusters = preds[0].get_clusters()
            if not clusters:
                return 1.0
            return max(0.5, 1.0 - (len(clusters) * 0.1))
        except Exception:
            return 0.5
