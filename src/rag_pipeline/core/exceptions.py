from __future__ import annotations


class RAGPipelineError(Exception):
    """Base exception for all pipeline errors."""


class ConfigurationError(RAGPipelineError):
    """Raised when required configuration is missing or invalid."""


class IngestionError(RAGPipelineError):
    """Raised when document ingestion fails."""


class RetrievalError(RAGPipelineError):
    """Raised when retrieval fails."""


class SafetyGateError(RAGPipelineError):
    """Raised when a safety gate blocks processing."""


class CircuitBreakerError(RAGPipelineError):
    """Raised when the circuit breaker halts processing due to low confidence."""


class SynthesisError(RAGPipelineError):
    """Raised when synthesis fails."""


class VerificationError(RAGPipelineError):
    """Raised when verification fails."""


class MaxCorrectionAttemptsError(RAGPipelineError):
    """Raised when the self-correction loop exceeds the maximum number of attempts."""
