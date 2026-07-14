from __future__ import annotations

from typing import Any

from rag_pipeline.config.settings import get_logger
from rag_pipeline.core.models import SafetyVerdict

logger = get_logger()


class CompositeSafetyGate:

    def __init__(self, gates: list[Any] | None = None):
        self._gates = gates or []

    def add_gate(self, gate: Any) -> None:
        self._gates.append(gate)

    def check(self, text: str) -> SafetyVerdict:
        for gate in self._gates:
            verdict = gate.check(text)
            if not verdict.passed:
                logger.warning(f"Safety gate {gate.__class__.__name__} blocked: {verdict.reason}")
                return verdict
        return SafetyVerdict(passed=True, gate_name="composite", score=1.0, reason="All gates passed")
