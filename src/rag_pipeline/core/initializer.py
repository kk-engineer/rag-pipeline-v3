from __future__ import annotations

from pathlib import Path

import requests

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.utils.model_loading import _check_complete

logger = get_logger()


def _check_model(settings, repo_id: str) -> bool:
    local_dir = Path(settings.paths.local_models_dir)

    # Check HF snapshot structure: models--org--name/snapshots/<hash>/
    cache_dir_name = "models--" + repo_id.replace("/", "--")
    snapshots_dir = local_dir / cache_dir_name / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots and _check_complete(snapshots[-1]):
            logger.info(f"  [local_models] {repo_id} ... found")
            return True

    # Check direct path: local_models/org/name/
    direct = local_dir / repo_id
    if direct.exists() and _check_complete(direct):
        logger.info(f"  [local_models] {repo_id} ... found")
        return True

    # Check direct path with snapshots subdir: local_models/org/name/snapshots/<hash>/
    snapshots_dir = direct / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots and _check_complete(snapshots[-1]):
            logger.info(f"  [local_models] {repo_id} ... found")
            return True

    logger.error(f"  [local_models] {repo_id} ... NOT FOUND")
    return False


def _all_models(settings) -> list[str]:
    m = settings.models
    return [
        m.chunking_embedding,
        m.reranker,
        m.prompt_guard,
        m.llama_guard,
        m.verification,
        m.intent_classifier,
    ]


def _check_all_models(settings) -> bool:
    all_ok = True
    for model_id in _all_models(settings):
        if not _check_model(settings, model_id):
            all_ok = False
    return all_ok


def _check_nim_reachable(settings) -> bool:
    key = settings.nvidia_api_key
    if not key:
        logger.error("  [llm] NVIDIA_API_KEY not set")
        return False
    base = settings.synthesis.base_url.rstrip("/")
    try:
        resp = requests.get(
            f"{base}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=settings.synthesis.request_timeout,
        )
        if resp.ok:
            logger.success("  [llm] NIM API reachable")
            return True
        logger.warning(f"  [llm] NIM API returned {resp.status_code}")
        return False
    except requests.RequestException as e:
        logger.error(f"  [llm] NIM API not reachable ({e})")
        return False


def _check_llm(settings) -> bool:
    return _check_nim_reachable(settings)


def initialize() -> bool:
    settings = get_settings()
    logger.notice(60 * "\u2500")
    logger.notice("RAG Pipeline Initialization")
    logger.notice(60 * "\u2500")

    all_ok = True

    logger.info("Checking models ...")
    if not _check_all_models(settings):
        all_ok = False

    logger.info("Checking LLM ...")
    if not _check_llm(settings):
        all_ok = False

    logger.notice(60 * "\u2500")
    if all_ok:
        logger.success("All checks passed. RAG pipeline is ready.")
    else:
        logger.error("Some checks failed. RAG pipeline is NOT ready.")
    logger.notice(60 * "\u2500")

    return all_ok
