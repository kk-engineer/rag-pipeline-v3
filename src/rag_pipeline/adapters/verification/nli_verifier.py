from __future__ import annotations

import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.utils.model_loading import resolve_model_path

logger = get_logger()


class NLIVerifier:

    def __init__(self, model_name: str = ""):
        settings = get_settings()
        self._model_name = model_name or settings.models.verification
        self._max_length = settings.thresholds.max_tokenizer_length
        self._model = None
        self._tokenizer = None
        logger.info(f"NLIVerifier model={self._model_name}")

    def _lazy_load(self):
        if self._model is None:
            settings = get_settings()
            model_path = resolve_model_path(settings, self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            logger.info(f"NLI model loaded: {self._model_name}")

    def verify(self, answer: dict, context: dict) -> dict:
        self._lazy_load()
        sentences = self._split_sentences(answer.get("answer", ""))
        chunks = context.get("chunks", [])

        if not sentences or not chunks:
            return {
                "passed": False,
                "sentence_verdicts": [],
                "verified_chunk_ids": set(),
                "correction_attempts": 0,
                "max_attempts_reached": False,
                "final_answer": answer.get("answer", ""),
            }

        sentence_verdicts = []
        all_passed = True
        all_verified_ids: set[str] = set()

        for sent in sentences:
            supporting = []
            for c in chunks:
                inputs = self._tokenizer(sent, c.content, return_tensors="pt",
                                         truncation=True, max_length=self._max_length, padding=True)
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                entail_score = probs[0][0].item()
                contrad_score = probs[0][2].item()
                if entail_score > contrad_score:
                    supporting.append(c.chunk_id)

            passed = bool(supporting)
            sentence_verdicts.append({
                "sentence": sent,
                "passed": passed,
                "chunks": supporting,
            })
            if passed:
                all_verified_ids.update(supporting)
            else:
                logger.warning(f"NLI failed: {sent[:50]}")
                all_passed = False

        logger.info(f"NLI verification result: passed={all_passed} ({len(sentences)} sentences, {len(chunks)} chunks)")
        return {
            "passed": all_passed,
            "sentence_verdicts": sentence_verdicts,
            "verified_chunk_ids": all_verified_ids,
            "correction_attempts": 0,
            "max_attempts_reached": False,
            "final_answer": answer.get("answer", ""),
        }

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
