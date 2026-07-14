from __future__ import annotations

from typing import Any


def format_pipeline_result(result: dict) -> dict[str, Any]:
    ctx = result.get("contextualized")
    ctx_text = ctx.resolved_query if ctx else result.get("query", "")
    answer_text = result.get("final_answer", "")
    halted_text = f"Halted: {result['halt_reason']}" if result.get("halted") else "Proceeding"

    reranked = result.get("reranked")
    retrieval_text = (
        f"{len(reranked.chunks)} chunks, "
        f"top score={reranked.scores[0]:.4f}" if reranked and reranked.scores else "N/A"
    )

    verification = result.get("verification")
    verification_text = (
        f"Passed: {verification['passed']}, "
        f"correction attempts: {verification.get('correction_attempts', 0)}"
        if verification else "N/A"
    )

    return {
        "contextualized": ctx_text,
        "final_answer": answer_text,
        "halted": halted_text,
        "retrieval": retrieval_text,
        "verification": verification_text,
    }
