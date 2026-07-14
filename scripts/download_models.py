from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.core.initializer import _all_models, _check_model
from rag_pipeline.utils.model_loading import _check_complete

logger = get_logger()

NON_HF_KEYS = {"fastcoref", "cl100k_base"}


def main() -> int:
    settings = get_settings()
    models_dir = Path(settings.paths.local_models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_ids = _all_models(settings)
    unique_ids = sorted(set(all_ids))

    ok = 0
    failed = 0

    for model_id in unique_ids:
        if model_id in NON_HF_KEYS:
            print(f"  - {model_id} (non-HF, skipping)")
            ok += 1
            continue

        if _check_model(settings, model_id):
            print(f"  \u2713 {model_id}")
            ok += 1
            continue

        local_dir = models_dir / model_id
        print(f"  \u2193 Downloading {model_id} ...")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                token=os.environ.get("HF_TOKEN"),
            )
            if _check_complete(local_dir):
                print(f"    Done \u2192 {local_dir}")
                ok += 1
            else:
                print(f"    \u26a0 Download incomplete (missing weights in {local_dir})")
                failed += 1
        except Exception as e:
            print(f"    \u2717 Failed: {e}")
            failed += 1

    print()
    if failed == 0:
        print(f"All {ok} model(s) present.")
        return 0
    else:
        print(f"{ok} model(s) OK, {failed} failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
