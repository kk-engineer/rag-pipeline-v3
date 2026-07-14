from __future__ import annotations

import re
import uuid
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.core.models import Chunk
from rag_pipeline.utils.model_loading import resolve_model_path

logger = get_logger()


class SemanticChunker:

    def __init__(self, model_name: str = "", threshold: float | None = None):
        settings = get_settings()
        self._model_name = model_name or settings.models.chunking_embedding
        self._threshold = threshold if threshold is not None else settings.thresholds.chunk_drift
        self._default_tenant_id = settings.machine.default_tenant_id
        model_path = resolve_model_path(settings, self._model_name)
        self._model = SentenceTransformer(model_path)
        logger.info(f"SemanticChunker initialised; model={self._model_name}, threshold={self._threshold}")

    def chunk(self, document: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        if metadata is None:
            metadata = {}

        sentences = self._split_sentences(document)
        logger.debug(f"Split into {len(sentences)} sentences")
        if not sentences:
            return []

        embeddings = self._model.encode(sentences, convert_to_numpy=True)
        similarities = self._cosine_similarity_matrix(embeddings)
        boundaries = self._find_drift_boundaries(similarities)
        chunks = self._build_chunks(sentences, boundaries, metadata)
        logger.info(f"Chunked {len(sentences)} sentences into {len(chunks)} chunks")
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalised = embeddings / norms
        return normalised @ normalised.T

    def _find_drift_boundaries(self, similarities: np.ndarray) -> list[int]:
        boundaries = [0]
        for i in range(len(similarities) - 1):
            if similarities[i, i + 1] < self._threshold:
                boundaries.append(i + 1)
        boundaries.append(len(similarities))
        return boundaries

    def _build_chunks(self, sentences: list[str], boundaries: list[int], metadata: dict[str, Any]) -> list[Chunk]:
        chunks = []
        for i in range(len(boundaries) - 1):
            content = " ".join(sentences[boundaries[i]:boundaries[i+1]])
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                doc_id=metadata.get("doc_id", ""),
                page_number=metadata.get("page_number", 0),
                chunk_index=i,
                filename=metadata.get("filename", ""),
                tenant_id=metadata.get("tenant_id", self._default_tenant_id),
                doc_version_uuid=metadata.get("doc_version_uuid", ""),
                metadata=metadata,
            )
            chunks.append(chunk)
        return chunks
