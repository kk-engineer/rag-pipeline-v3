from __future__ import annotations

from rag_pipeline.config.settings import get_logger, get_settings

logger = get_logger()


class PipelineBuilder:

    def __init__(self):
        self._settings = get_settings()
        self._tier = self._settings.pipeline.tier

    @property
    def pipeline_tier(self) -> str:
        return self._tier

    @pipeline_tier.setter
    def pipeline_tier(self, value: str) -> None:
        self._tier = value

    def build(self) -> list[dict]:
        tier = self._tier
        if tier == "vanilla":
            stages = [
                {"name": "intake", "config": {"safety": False}},
                {"name": "retrieval", "config": {"rerank": True}},
                {"name": "compression"},
                {"name": "synthesis"},
            ]
        elif tier == "reliable":
            stages = [
                {"name": "intake", "config": {"safety": True, "llamaguard": self._settings.safety.llama_guard_enabled}},
                {"name": "retrieval", "config": {"rerank": True, "circuit_breaker": True}},
                {"name": "compression"},
                {"name": "synthesis"},
                {"name": "verification", "config": {"self_correction": True, "max_attempts": self._settings.thresholds.correction_loop_max_attempts}},
            ]
        elif tier == "chaos":
            stages = [
                {"name": "intake", "config": {"safety": True, "llamaguard": True}},
                {"name": "retrieval", "config": {"rerank": True, "circuit_breaker": True}},
                {"name": "compression"},
                {"name": "synthesis"},
                {"name": "verification", "config": {"self_correction": True, "max_attempts": self._settings.thresholds.correction_loop_max_attempts}},
            ]
        else:
            raise ValueError(f"Unknown pipeline tier: {tier}")

        logger.info(f"Pipeline built: {tier} tier, {len(stages)} stages")
        return stages
