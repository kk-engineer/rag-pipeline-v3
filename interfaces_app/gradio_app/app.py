from __future__ import annotations

import gradio as gr

from interfaces_app.gradio_app.tabs.debug_tab import create_debug_tab
from interfaces_app.gradio_app.tabs.ingest_tab import create_ingest_tab
from interfaces_app.gradio_app.tabs.query_tab import create_query_tab
from rag_pipeline.config.settings import get_logger

logger = get_logger()


def create_app(init_ok: bool = True) -> gr.Blocks:
    with gr.Blocks(title="RAG Pipeline") as app:
        gr.Markdown("# RAG Pipeline")
        gr.Markdown(
            "Production-grade retrieval-augmented generation with "
            "deterministic synthesis, multi-layer hallucination defense, "
            "and full observability."
        )

        if not init_ok:
            gr.Markdown(
                "## :warning: Initialization Failed\n"
                "The pipeline failed one or more startup checks. "
                "Please check the logs above for details."
            )

        create_ingest_tab()
        create_query_tab()
        create_debug_tab()

    return app


if __name__ == "__main__":
    from rag_pipeline.core.initializer import initialize

    init_ok = initialize()
    app = create_app(init_ok=init_ok)
    app.launch()
