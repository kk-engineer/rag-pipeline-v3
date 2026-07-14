from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from rag_pipeline.config.settings import get_logger, get_settings

logger = get_logger()


class PresidioPIIAdapter:

    def __init__(self, enabled: bool | None = None):
        settings = get_settings()
        self._enabled = enabled if enabled is not None else settings.safety.pii_enabled
        self._language = settings.safety.pii_language
        self._analyzer = None
        self._anonymizer = None
        if self._enabled:
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
        logger.info(f"PresidioPIIAdapter enabled={self._enabled}")

    def analyze(self, text: str) -> dict:
        if not self._enabled or not self._analyzer:
            return []
        results = self._analyzer.analyze(text=text, language=self._language)
        pii_list = [{"entity_type": r.entity_type, "start": r.start, "end": r.end,
                      "score": r.score, "text": text[r.start:r.end]} for r in results]
        if pii_list:
            logger.info(f"Found {len(pii_list)} PII entities in text ({len(text)} chars)")
        return pii_list

    def anonymize(self, text: str) -> str:
        if not self._enabled or not self._analyzer:
            return text
        analyzer_results = self._analyzer.analyze(text=text, language=self._language)
        if analyzer_results:
            logger.info(f"Anonymizing {len(analyzer_results)} PII entities")
        result = self._anonymizer.anonymize(text=text, analyzer_results=analyzer_results)
        return result.text
