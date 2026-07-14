from __future__ import annotations

from pathlib import Path

import requests

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.prompts import SYNTHESIS_SYSTEM_PROMPT

logger = get_logger()


def _format_context(context: dict) -> str:
    chunks = context.get("chunks", [])
    parts = []
    for i, c in enumerate(chunks, 1):
        doc_name = c.filename or Path(c.doc_id).name if c.doc_id else "?"
        page = c.page_number
        parts.append(f"[{i}] (Doc: {doc_name}, Page: {page})\n{c.content}")
    return "\n\n".join(parts)


class NIMSynthesizer:

    def __init__(self):
        settings = get_settings()
        self._model_name = settings.synthesis.model
        self._base_url = settings.synthesis.base_url
        self._temperature = settings.synthesis.temperature
        self._seed = settings.synthesis.seed
        self._api_key = settings.nvidia_api_key
        self._max_tokens = settings.synthesis.max_tokens
        self._timeout = settings.synthesis.request_timeout
        logger.info(f"NIMSynthesizer model={self._model_name}")

    def synthesize(self, context: dict, query: str) -> dict:
        context_text = _format_context(context)
        prompt = SYNTHESIS_SYSTEM_PROMPT.format(context=context_text, query=query)

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self._model_name, "messages": [{"role": "user", "content": prompt}],
                   "temperature": self._temperature, "seed": self._seed, "max_tokens": self._max_tokens}

        try:
            response = requests.post(f"{self._base_url}/chat/completions", headers=headers, json=payload, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            logger.info(f"NIM synthesis successful: {len(answer)} chars, usage={usage}")
        except Exception as e:
            logger.error(f"NIM synthesis failed: {e}")
            answer = f"Synthesis error: {e}"
            usage = {}

        return {"answer": answer, "model": self._model_name,
                "raw_output": answer, "token_usage": usage}
