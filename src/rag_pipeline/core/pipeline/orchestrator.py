from __future__ import annotations

import gc
from typing import Any

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.core.pipeline.builder import PipelineBuilder
from rag_pipeline.prompts import CIRCUIT_BREAKER_RESPONSE, SAFETY_REFUSAL_RESPONSE
from rag_pipeline.utils.citation_parser import build_source_list
from rag_pipeline.utils.logging import timed_step

logger = get_logger()


class PipelineOrchestrator:

    def __init__(self):
        self._settings = get_settings()
        self._builder = PipelineBuilder()
        self._stage_seq = self._builder.build()
        logger.info(f"Orchestrator initialised; tier={self._settings.pipeline.tier}, stages={len(self._stage_seq)}")

    @property
    def pipeline_tier(self) -> str:
        return self._settings.pipeline.tier

    @property
    def stage_count(self) -> int:
        return len(self._stage_seq)

    def run(self, query_text: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        from rag_pipeline.core.pipeline.stages import (
            compress_context,
            intake_chain,
            retrieve_and_rerank,
            synthesize_answer,
            verify_and_correct,
        )

        with timed_step("PipelineOrchestrator.run", logger):
            logger.info(f"Query: {query_text[:80]}")

            result = intake_chain(query=type("Q", (), {"text": query_text})(), history=history)
            result["query"] = query_text
            gc.collect()
            if result.get("halted"):
                result["final_answer"] = SAFETY_REFUSAL_RESPONSE
                logger.warning(f"Pipeline halted at intake: {result['halt_reason']}")
                return result

            retrieval_result = retrieve_and_rerank(result["contextualized"].resolved_query)
            result.update(retrieval_result)
            gc.collect()
            if result.get("halted"):
                result["final_answer"] = CIRCUIT_BREAKER_RESPONSE
                logger.warning(f"Pipeline halted at retrieval: {result['halt_reason']}")
                return result

            compressed = compress_context(result["contextualized"].resolved_query, result["reranked"])
            result["compressed"] = compressed
            gc.collect()

            answer = synthesize_answer(compressed, result["contextualized"].resolved_query)
            result["answer"] = answer

            stage_names = [s["name"] for s in self._stage_seq]
            if "verification" in stage_names:
                gc.collect()
                verification = verify_and_correct(answer, compressed, result["contextualized"].resolved_query)
                result["verification"] = verification
                result["final_answer"] = verification.get("final_answer", answer.get("answer", ""))

                if verification.get("passed") and verification.get("verified_chunk_ids"):
                    compressed_chunks = compressed.get("chunks", [])
                    source_text = build_source_list(compressed_chunks, verification["verified_chunk_ids"])
                    if source_text:
                        result["final_answer"] += "\n\nSources:\n\n" + source_text

            else:
                result["final_answer"] = answer.get("answer", "")

            logger.success("Pipeline run completed")
            return result
