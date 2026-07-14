from __future__ import annotations

import os
from pathlib import Path

import tomli
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    log_level: str
    log_file: str
    log_json: bool


class LocalProfileSettings(BaseSettings):
    pass


class SpacesProfileSettings(BaseSettings):
    pass


class ProfileSettings(BaseSettings):
    active: str
    local: LocalProfileSettings
    spaces: SpacesProfileSettings


class PipelineSettings(BaseSettings):
    tier: str


class ThresholdSettings(BaseSettings):
    chunk_drift: float
    injection_confidence_floor: float
    intent_confidence_floor: float
    retrieval_circuit_breaker: float
    correction_loop_max_attempts: int
    max_compression_tokens: int
    default_retrieval_k: int
    max_tokenizer_length: int
    rrf_constant: int


class ModelSettings(BaseSettings):
    chunking_embedding: str
    reranker: str
    prompt_guard: str
    llama_guard: str
    verification: str
    intent_classifier: str
    coreference: str
    tokenizer_encoding: str


class SafetySettings(BaseSettings):
    prompt_guard_enabled: bool
    pii_enabled: bool
    llama_guard_enabled: bool
    pii_language: str


class SynthesisSettings(BaseSettings):
    base_url: str
    model: str
    temperature: float
    seed: int
    max_tokens: int
    request_timeout: int


class MachineSettings(BaseSettings):
    max_history: int
    redis_url: str
    redis_ttl: int
    default_tenant_id: str


class PathSettings(BaseSettings):
    chroma_db_path: str
    data_dir: str
    golden_dataset_dir: str
    eval_reports_dir: str
    ledger_db_path: str
    local_models_dir: str
    chroma_collection_name: str
    chroma_index_space: str


class TracingSettings(BaseSettings):
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str
    opensmith_project: str
    opensmith_api_key: str
    opensmith_base_url: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    nvidia_api_key: str

    app: AppSettings
    profile: ProfileSettings
    pipeline: PipelineSettings
    thresholds: ThresholdSettings
    models: ModelSettings
    paths: PathSettings
    safety: SafetySettings
    synthesis: SynthesisSettings
    machine: MachineSettings
    tracing: TracingSettings

    @classmethod
    def from_toml(cls, path: str | Path = "config/config.toml") -> Settings:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "rb") as f:
            data = tomli.load(f)

        app = AppSettings(**data.get("app", {}))
        profile_raw = data.get("profile", {})
        profile = ProfileSettings(
            active=profile_raw.get("active"),
            local=LocalProfileSettings(**profile_raw.get("local", {})),
            spaces=SpacesProfileSettings(**profile_raw.get("spaces", {})),
        )
        pipeline = PipelineSettings(**data.get("pipeline", {}))
        thresholds = ThresholdSettings(**data.get("thresholds", {}))
        models = ModelSettings(**data.get("models", {}))
        paths = PathSettings(**data.get("paths", {}))
        safety = SafetySettings(**data.get("safety", {}))
        synthesis = SynthesisSettings(**data.get("synthesis", {}))
        machine = MachineSettings(**data.get("memory", {}))
        tracing = TracingSettings(**data.get("tracing", {}))

        nvidia_api_key = data.get("nvidia_api_key")
        if not nvidia_api_key:
            nvidia_api_key = os.environ.get("nvidia_api_key") or ""

        return cls(
            nvidia_api_key=nvidia_api_key,
            app=app,
            profile=profile,
            pipeline=pipeline,
            thresholds=thresholds,
            models=models,
            paths=paths,
            safety=safety,
            synthesis=synthesis,
            machine=machine,
            tracing=tracing,
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_toml()
    return _settings


_logger_instance = None


def get_logger():
    global _logger_instance
    if _logger_instance is None:
        from rag_pipeline.utils.logging import Logger
        _logger_instance = Logger.get_instance(get_settings())
    return _logger_instance
