from __future__ import annotations

import json

import gradio as gr

from rag_pipeline.core.pipeline.stages import run_pipeline
from interfaces_app.gradio_app.formatters import format_pipeline_result


def create_query_tab() -> gr.Tab:
    with gr.Tab("Query Pipeline") as tab:
        gr.Markdown("## Run Full Query Pipeline")
        gr.Markdown(
            "Enter a query to run through the full pipeline: contextualization, "
            "routing, safety gates, retrieval, reranking, synthesis, and verification."
        )

        query_input = gr.Textbox(label="Query", placeholder="Enter a query...")
        history_input = gr.Textbox(
            label="Conversation History (JSON)",
            placeholder='[{"user": "Hello", "assistant": "Hi, how can I help?"}]',
            value="[]",
        )
        query_button = gr.Button("Run Query", variant="primary")

        ctx_output = gr.Textbox(label="Contextualized Query")
        final_answer = gr.Markdown(label="Final Answer")
        with gr.Accordion("Debug Output", open=False):
            halted = gr.Textbox(label="Halted / Reason")
            retrieval_info = gr.Textbox(label="Retrieval")
            verification_info = gr.Textbox(label="Verification")
            raw = gr.JSON(label="Full Pipeline Output")

        def do_query(query_text: str, history_json: str, progress: gr.Progress = gr.Progress()):
            progress(0, desc="Parsing input")
            history = json.loads(history_json) if history_json else []

            progress(0.2, desc="Running pipeline")
            result = run_pipeline(query_text, history)

            progress(0.8, desc="Formatting output")
            fmt = format_pipeline_result(result)

            progress(1.0, desc="Done")
            return fmt["contextualized"], fmt["final_answer"], fmt["halted"], fmt["retrieval"], fmt["verification"], result

        query_button.click(
            fn=do_query,
            inputs=[query_input, history_input],
            outputs=[ctx_output, final_answer, halted, retrieval_info, verification_info, raw],
        )
        query_input.submit(
            fn=do_query,
            inputs=[query_input, history_input],
            outputs=[ctx_output, final_answer, halted, retrieval_info, verification_info, raw],
        )

    return tab
