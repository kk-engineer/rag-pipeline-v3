from __future__ import annotations

import json

from rag_pipeline.config.settings import get_logger, get_settings

logger = get_logger()


class RedisMemory:

    def __init__(self, redis_url: str = "", ttl_seconds: int | None = None, max_history: int | None = None):
        settings = get_settings()
        self._redis_url = redis_url or settings.machine.redis_url
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.machine.redis_ttl
        self._max_history = max_history if max_history is not None else settings.machine.max_history
        self._client = None
        self._fallback_store: dict[str, list] = {}

        if self._redis_url:
            try:
                import redis as redis_lib
                self._client = redis_lib.from_url(self._redis_url)
                logger.info(f"RedisMemory connected: {self._redis_url}")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}. Using in-memory fallback.")
                self._client = None
        else:
            logger.info("RedisMemory in in-memory mode")

    def get_history(self, conversation_id: str) -> list:
        if self._client:
            try:
                data = self._client.get(f"conversation:{conversation_id}")
                if data:
                    return json.loads(data)
            except Exception:
                pass
        return self._fallback_store.get(conversation_id, [])

    def add_turn(self, conversation_id: str, user_msg: str, assistant_msg: str) -> None:
        history = self.get_history(conversation_id)
        history.append({"user": user_msg, "assistant": assistant_msg})
        if len(history) > self._max_history:
            history = history[-self._max_history:]

        if self._client:
            try:
                self._client.setex(f"conversation:{conversation_id}", self._ttl, json.dumps(history))
            except Exception:
                self._fallback_store[conversation_id] = history
        else:
            self._fallback_store[conversation_id] = history

        logger.debug(f"Turn added for {conversation_id}, total: {len(history)}")
