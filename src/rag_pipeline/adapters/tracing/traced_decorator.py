from __future__ import annotations

import functools
import time
from typing import Any, Callable


class TracerObserver:
    """Base observer for the Observer pattern.

    Pipeline stages publish a single telemetry event;
    all registered observers persist it independently.
    """

    def on_event(self, event: str, data: dict[str, Any]) -> None:
        ...


class TracedDecorator:
    """Decorator that wraps a stage function and emits telemetry events.

    Implements the Decorator pattern — stage logic and observability
    are separated, and adding tracing to a stage is a one-line addition.
    """

    def __init__(self) -> None:
        self._observers: list[TracerObserver] = []

    def add_observer(self, observer: TracerObserver) -> None:
        self._observers.append(observer)

    def _notify(self, event: str, data: dict[str, Any]) -> None:
        for observer in self._observers:
            observer.on_event(event, data)

    def trace(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._notify("stage.start", {"name": func.__name__, "args": str(args)[:200]})
            start = time.monotonic()
            try:
                result = func(*args, **kwargs)
                elapsed = time.monotonic() - start
                self._notify("stage.end", {
                    "name": func.__name__,
                    "elapsed_seconds": elapsed,
                })
                return result
            except Exception as e:
                elapsed = time.monotonic() - start
                self._notify("stage.error", {
                    "name": func.__name__,
                    "elapsed_seconds": elapsed,
                    "error": str(e),
                })
                raise
        return wrapper


# Global singleton
_tracer = TracedDecorator()


def get_tracer() -> TracedDecorator:
    return _tracer


def traced(func: Callable) -> Callable:
    """Decorator to add tracing to any pipeline stage."""
    return _tracer.trace(func)
