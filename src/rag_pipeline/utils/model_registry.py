from __future__ import annotations

from typing import Any, Callable


class ModelRegistry:
    """Lazy-load and cache models, shared across all adapter instances.

    Models are loaded once on first access and cached for reuse. Keyed by
    model name so the same underlying model (e.g. cross-encoder used by
    both reranker and compressor) is only loaded once.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def get_or_load(self, key: str, factory: Callable[[], Any]) -> Any:
        if key not in self._models:
            self._models[key] = factory()
        return self._models[key]

    def get(self, key: str) -> Any | None:
        return self._models.get(key)

    def unload(self, key: str) -> None:
        self._models.pop(key, None)

    def clear(self) -> None:
        self._models.clear()


_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
