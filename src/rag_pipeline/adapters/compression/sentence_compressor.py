from __future__ import annotations

import re

import tiktoken
from sentence_transformers import CrossEncoder

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.core.models import RerankedResult
from rag_pipeline.utils.model_loading import resolve_model_path

logger = get_logger()


class SentenceCompressor:

    def __init__(self):
        settings = get_settings()
        self._model_name = settings.models.reranker
        self._max_tokens = settings.thresholds.max_compression_tokens
        model_path = resolve_model_path(settings, self._model_name)
        self._model = CrossEncoder(model_path)
        self._tokenizer = tiktoken.get_encoding(settings.models.tokenizer_encoding)
        logger.info(f"SentenceCompressor model={self._model_name} max_tokens={self._max_tokens}")

    def compress(self, query: str, reranked_result: RerankedResult) -> dict:
        all_sentences = []
        for chunk in reranked_result.chunks:
            for sentence in re.split(r'(?<=[.!?])\s+', chunk.content):
                sentence = sentence.strip()
                if sentence:
                    all_sentences.append((sentence, chunk))

        if not all_sentences:
            return {"chunks": [], "token_count": 0, "original_token_count": 0}

        scores = self._model.predict([(query, sent) for sent, _ in all_sentences])
        scored = sorted(zip(all_sentences, scores), key=lambda x: x[1], reverse=True)

        compressed = []
        total_tokens = 0
        original_tokens = sum(len(self._tokenizer.encode(s)) for s, _ in all_sentences)

        for (sent, chunk), score in scored:
            sent_tokens = len(self._tokenizer.encode(sent))
            if total_tokens + sent_tokens > self._max_tokens:
                break
            compressed.append(chunk)
            total_tokens += sent_tokens

        logger.info(f"Compressed: {len(all_sentences)} sentences -> {len(compressed)} chunks ({total_tokens}/{original_tokens} tokens)")
        return {"chunks": compressed, "token_count": total_tokens, "original_token_count": original_tokens}
