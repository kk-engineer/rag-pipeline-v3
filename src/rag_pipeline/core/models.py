from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    content: str
    doc_id: str
    page_number: int = 0
    chunk_index: int = 0
    filename: str = ""
    tenant_id: str = "default"
    doc_version_uuid: str = ""
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Query(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    conversation_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextualizedQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    original_query: str
    resolved_query: str
    coref_confidence: float = 1.0
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    intent: str = "unknown"
    confidence: float = 0.0
    is_in_scope: bool = True
    routed_to: str = "retrieval"


class SafetyVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool = True
    gate_name: str = ""
    score: float = 1.0
    reason: str = ""
    risk_category: str = ""


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: list[Chunk] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    method: str = "hybrid"


class RerankedResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: list[Chunk] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    method: str = "cross_encoder"


class CompressedContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: list[Chunk] = Field(default_factory=list)
    token_count: int = 0
    original_token_count: int = 0


class SynthesisOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    model: str = ""
    raw_output: str = ""
    token_usage: dict[str, int] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool = True
    sentence_verdicts: list[dict[str, Any]] = Field(default_factory=list)
    correction_attempts: int = 0
    max_attempts_reached: bool = False
    final_answer: str = ""
