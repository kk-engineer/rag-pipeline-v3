from __future__ import annotations

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.core.models import RoutingDecision

logger = get_logger()


class ZeroShotIntentClassifier:

    def __init__(self, confidence_floor: float | None = None):
        settings = get_settings()
        self._confidence_floor = confidence_floor if confidence_floor is not None else settings.thresholds.intent_confidence_floor
        logger.info(f"ZeroShotIntentClassifier floor={self._confidence_floor}")

    def classify(self, query: str) -> RoutingDecision:
        q = query.lower().strip()

        for kw in ["ignore previous", "forget instructions", "system prompt",
                     "you are now", "act as", "jailbreak", "bypass"]:
            if kw in q:
                logger.warning(f"Adversarial query detected: {query[:60]}")
                return RoutingDecision(intent="adversarial", confidence=0.95,
                                       is_in_scope=False, routed_to="safety_refusal")

        for kw in ["hello", "hi ", "hey", "greetings", "good morning", "good evening"]:
            if q.startswith(kw) or q == kw:
                logger.info(f"Greeting detected: {query[:40]}")
                return RoutingDecision(intent="greeting", confidence=0.9,
                                       is_in_scope=False, routed_to="greeting_response")

        logger.info(f"Classified as document_qa: {query[:40]}")
        return RoutingDecision(intent="document_qa", confidence=0.85,
                               is_in_scope=True, routed_to="retrieval")
