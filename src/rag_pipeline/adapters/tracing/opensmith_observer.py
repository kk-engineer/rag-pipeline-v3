from __future__ import annotations

from typing import Any

from rag_pipeline.adapters.tracing.traced_decorator import TracerObserver
from rag_pipeline.config.settings import get_settings


class OpenSmithObserver(TracerObserver):
    def __init__(self) -> None:
        settings = get_settings()
        tracing = settings.tracing
        self._enabled = bool(tracing.opensmith_api_key)
        self._client = None
        if self._enabled:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=tracing.opensmith_api_key,
                    secret_key=tracing.opensmith_api_key,
                    host=tracing.opensmith_base_url,
                )
            except Exception:
                self._client = None

    def on_event(self, event: str, data: dict[str, Any]) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            if event == "stage.start":
                self._client.generation(
                    name=data.get("name", "unknown"),
                    input=data.get("args", ""),
                    metadata={"source": "opensmith"},
                )
            elif event == "stage.end":
                self._client.generation(
                    name=data.get("name", "unknown"),
                    output="ok",
                    metadata={"source": "opensmith"},
                )
        except Exception:
            pass
