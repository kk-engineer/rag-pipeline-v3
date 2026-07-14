from __future__ import annotations

import json

import gradio as gr

from rag_pipeline.core.models import Query
from rag_pipeline.core.pipeline.stages import intake_chain


def create_debug_tab() -> gr.Tab:
    with gr.Tab("Debug: Query Intake") as tab:
        gr.Markdown("## Query Intake Debug Panel")
        gr.Markdown(
            "Inspect each stage of the query intake pipeline: "
            "contextualization, routing, and safety gates."
        )

        query_input = gr.Textbox(label="Query", placeholder="Enter a query...")
        history_input = gr.Textbox(
            label="Conversation History (JSON)",
            placeholder='[{"user": "Hello", "assistant": "Hi, how can I help?"}]',
        )
        debug_button = gr.Button("Run Intake Chain", variant="primary")

        with gr.Row():
            contextualized_output = gr.Textbox(label="Contextualized Query")
            routing_output = gr.Textbox(label="Routing Decision")
            safety_output = gr.Textbox(label="Safety Verdict")

        halted_output = gr.Textbox(label="Halted / Reason")
        raw_output = gr.JSON(label="Full Debug Output")

        def run_intake(query_text, history_json):
            history = json.loads(history_json) if history_json else []
            q = Query(text=query_text)
            result = intake_chain(q, history)
            ctx = result.get("contextualized")
            routing = result.get("routing")
            safety = result.get("safety")
            return (
                ctx.resolved_query if ctx else "",
                f"{routing.intent} (conf={routing.confidence:.2f})" if routing else "",
                f"{'PASS' if safety.passed else 'FAIL'}: {safety.reason}" if safety else "",
                f"Halted: {result['halt_reason']}" if result["halted"] else "Proceeding",
                result,
            )

        debug_button.click(
            fn=run_intake,
            inputs=[query_input, history_input],
            outputs=[contextualized_output, routing_output, safety_output, halted_output, raw_output],
        )
        query_input.submit(
            fn=run_intake,
            inputs=[query_input, history_input],
            outputs=[contextualized_output, routing_output, safety_output, halted_output, raw_output],
        )

    return tab
