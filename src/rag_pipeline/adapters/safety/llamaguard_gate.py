from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.core.models import SafetyVerdict
from rag_pipeline.utils.model_loading import resolve_model_path

logger = get_logger()


class LlamaGuardGate:

    def __init__(self, model_name: str = "", enabled: bool | None = None):
        settings = get_settings()
        self._model_name = model_name or settings.models.llama_guard
        self._enabled = enabled if enabled is not None else settings.safety.llama_guard_enabled
        self._max_length = settings.thresholds.max_tokenizer_length
        self._model = None
        self._tokenizer = None
        logger.info(f"LlamaGuardGate enabled={self._enabled} model={self._model_name}")

    def _lazy_load(self):
        if self._model is None and self._enabled:
            settings = get_settings()
            model_path = resolve_model_path(settings, self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            logger.info(f"LlamaGuard model loaded: {self._model_name}")

    def check(self, text: str) -> SafetyVerdict:
        if not self._enabled:
            return SafetyVerdict(passed=True, gate_name="llamaguard", score=1.0,
                                 reason="LlamaGuard disabled in current profile")
        self._lazy_load()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=self._max_length)
        outputs = self._model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        unsafe_score = probs[0][1].item()
        passed = unsafe_score < 0.5
        if not passed:
            logger.warning(f"Content safety violation: score={unsafe_score:.3f}")
        return SafetyVerdict(
            passed=passed, gate_name="llamaguard", score=unsafe_score,
            reason="" if passed else f"Content safety violation (score={unsafe_score:.3f})",
            risk_category="" if passed else "unsafe_content",
        )
