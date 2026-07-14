from __future__ import annotations

from typing import Any

from rag_pipeline.adapters.tracing.traced_decorator import TracerObserver
from rag_pipeline.config.settings import get_settings


class LangfuseObserver(TracerObserver):
    def __init__(self) -> None:
        settings = get_settings()
        tracing = settings.tracing
        self._client = None
        if tracing.langfuse_public_key and tracing.langfuse_secret_key:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=tracing.langfuse_public_key,
                    secret_key=tracing.langfuse_secret_key,
                    host=tracing.langfuse_host,
                )
            except Exception:
                self._client = None

    def on_event(self, event: str, data: dict[str, Any]) -> None:
        if self._client is None:
            return
        try:
            if event == "stage.start":
                self._client.generation(
                    name=data.get("name", "unknown"),
                    input=data.get("args", ""),
                )
            elif event == "stage.end":
                self._client.generation(
                    name=data.get("name", "unknown"),
                    completion="completed",
                    metadata={"elapsed_seconds": data.get("elapsed_seconds")},
                )
        except Exception:
            pass
