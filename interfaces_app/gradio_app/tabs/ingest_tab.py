from __future__ import annotations

from pathlib import Path

import gradio as gr

from rag_pipeline.core.pipeline.stages import ingest_pdf


def create_ingest_tab() -> gr.Tab:
    with gr.Tab("Ingest PDFs") as tab:
        gr.Markdown("## Ingest PDF Documents")
        gr.Markdown(
            "Upload PDF files to extract, anonymize (PII removal), "
            "semantically chunk, and index into the vector store."
        )

        file_input = gr.File(
            label="Upload PDF(s)",
            file_count="multiple",
            file_types=[".pdf"],
        )

        ingest_button = gr.Button("Ingest", variant="primary")
        output = gr.Dataframe(
            label="Ingestion Results",
            headers=["filename", "status", "chunk_count", "doc_version_uuid"],
        )

        def ingest_files(files):
            if not files:
                return [[]]
            results = []
            for f in files:
                try:
                    result = ingest_pdf(f.name)
                    results.append([
                        result.get("filename", ""),
                        result.get("status", ""),
                        result.get("chunk_count", 0),
                        result.get("doc_version_uuid", ""),
                    ])
                except Exception as e:
                    results.append([
                        Path(f.name).name,
                        f"error: {e}",
                        0,
                        "",
                    ])
            return results

        ingest_button.click(
            fn=ingest_files,
            inputs=[file_input],
            outputs=[output],
        )

    return tab
