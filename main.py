from __future__ import annotations

from rag_pipeline.core.initializer import initialize


def main() -> None:
    ok = initialize()
    if not ok:
        import sys
        print("FATAL: RAG pipeline initialization failed. Exiting.")
        sys.exit(1)

    from interfaces_app.gradio_app.app import create_app
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
