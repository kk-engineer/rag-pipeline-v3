from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.core.models import SafetyVerdict
from rag_pipeline.utils.model_loading import resolve_model_path

logger = get_logger()


class PromptGuardGate:

    def __init__(self, model_name: str = "", threshold: float | None = None, enabled: bool | None = None):
        settings = get_settings()
        self._model_name = model_name or settings.models.prompt_guard
        self._threshold = threshold if threshold is not None else settings.thresholds.injection_confidence_floor
        self._enabled = enabled if enabled is not None else settings.safety.prompt_guard_enabled
        self._max_length = settings.thresholds.max_tokenizer_length
        self._model = None
        self._tokenizer = None
        logger.info(f"PromptGuardGate enabled={self._enabled} model={self._model_name} threshold={self._threshold}")

    def _lazy_load(self):
        if self._model is None:
            settings = get_settings()
            model_path = resolve_model_path(settings, self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            logger.info(f"PromptGuard model loaded: {self._model_name}")

    def check(self, text: str) -> SafetyVerdict:
        if not self._enabled:
            return SafetyVerdict(passed=True, gate_name="promptguard", score=1.0,
                                 reason="PromptGuard disabled in config")
        self._lazy_load()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=self._max_length)
        outputs = self._model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        injection_score = probs[0][1].item()
        passed = injection_score < self._threshold
        if not passed:
            logger.warning(f"Prompt injection detected: score={injection_score:.3f}")
        return SafetyVerdict(
            passed=passed, gate_name="promptguard", score=injection_score,
            reason="" if passed else f"Prompt injection detected (score={injection_score:.3f})",
            risk_category="" if passed else "injection",
        )
