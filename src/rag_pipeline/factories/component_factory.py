from __future__ import annotations

from rag_pipeline.config.settings import Settings


class ComponentFactory:
    """Factory for instantiating pipeline components from config.

    This is the only place environment-branching logic is allowed to live.
    New adapters are registered here, not scattered through pipeline code.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def settings(self) -> Settings:
        return self._settings
