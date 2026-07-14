from __future__ import annotations

"""Run Garak red-teaming against the local pipeline endpoint."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    from rag_pipeline.core.initializer import initialize
    if not initialize():
        print("FATAL: RAG pipeline initialization failed. Exiting.")
        sys.exit(1)

    report_dir = Path("data/golden_dataset/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "garak_report.json"

    print("Running Garak red-teaming...")
    print(f"Report will be saved to {report_path}")

    cmd = [
        sys.executable, "-m", "garak",
        "--model_type", "rest",
        "--model_name", "rag-pipeline",
        "--model_url", "http://localhost:8000/generate",
        "--probes", "promptinject,dan,jailbreak",
        "--report_prefix", str(report_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Garak report saved to {report_path}")
    except FileNotFoundError:
        print("Garak not installed. Install with: pip install garak")
    except subprocess.CalledProcessError as e:
        print(f"Garak failed with exit code {e.returncode}")


if __name__ == "__main__":
        main()