from __future__ import annotations

from pathlib import Path

from rag_pipeline.config.settings import get_logger

logger = get_logger()

WEIGHT_FILES = {
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
}


def _has_model_weights(directory: Path) -> bool:
    return any((directory / f).exists() for f in WEIGHT_FILES)


def _check_complete(directory: Path) -> bool:
    return (directory / "config.json").exists() and _has_model_weights(directory)


def resolve_model_path(settings, model_name: str) -> str:
    local_dir = Path(settings.paths.local_models_dir)
    cache_dir_name = "models--" + model_name.replace("/", "--")

    # Check HuggingFace snapshot directory structure
    snapshots_dir = local_dir / cache_dir_name / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots and _check_complete(snapshots[-1]):
            return str(snapshots[-1])

    # Check direct model_name directory (e.g. local_models/org/name)
    direct = local_dir / model_name
    if direct.exists() and _check_complete(direct):
        return str(direct)

    # Check direct path with snapshots subdir: local_models/org/name/snapshots/<hash>/
    snapshots_dir = direct / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots and _check_complete(snapshots[-1]):
            return str(snapshots[-1])

    raise FileNotFoundError(
        f"Model '{model_name}' not found in {local_dir} "
        f"(checked {local_dir / cache_dir_name}, {direct})"
    )
