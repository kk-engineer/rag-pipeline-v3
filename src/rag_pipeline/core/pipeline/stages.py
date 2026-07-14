from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from rag_pipeline.adapters.chunking.semantic_chunker import SemanticChunker
from rag_pipeline.adapters.compression.sentence_compressor import SentenceCompressor
from rag_pipeline.adapters.contextualization.fastcoref_resolver import FastCorefResolver
from rag_pipeline.adapters.pii.presidio_adapter import PresidioPIIAdapter
from rag_pipeline.adapters.reranking.cross_encoder_reranker import CrossEncoderReranker
from rag_pipeline.adapters.retrieval.bm25_retriever import BM25Retriever, get_bm25
from rag_pipeline.adapters.retrieval.chroma_retriever import ChromaRetriever
from rag_pipeline.adapters.retrieval.circuit_breaker import CircuitBreaker
from rag_pipeline.adapters.retrieval.hybrid_retriever import HybridRetriever
from rag_pipeline.adapters.retrieval.ledger import IngestionLedger
from rag_pipeline.adapters.routing.zero_shot_classifier import ZeroShotIntentClassifier
from rag_pipeline.adapters.safety.composite_gate import CompositeSafetyGate
from rag_pipeline.adapters.synthesis.nim_synthesizer import NIMSynthesizer
from rag_pipeline.adapters.verification.nli_verifier import NLIVerifier
from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.core.models import Query
from rag_pipeline.prompts import (
    CORRECTION_LOOP_PROMPT,
)
from rag_pipeline.utils.logging import timed_step

logger = get_logger()


# ---------------------------------------------------------------------------
# Stage 1 – Document Ingestion
# ---------------------------------------------------------------------------


def ingest_pdf(file_path: str | Path) -> dict:
    settings = get_settings()
    file_path = Path(file_path)
    filename = file_path.name

    with timed_step(f"Ingest PDF: {filename}", logger):
        with open(file_path, "rb") as f:
            content = f.read()

        logger.info(f"Read {len(content)} bytes from {filename}")
        ledger = IngestionLedger()

        file_hash = ledger.compute_hash(content)
        if ledger.is_ingested(filename, file_hash):
            logger.info(f"Skipping already-ingested file: {filename}")
            return {"status": "skipped", "filename": filename, "reason": "unchanged"}

        reader = PdfReader(file_path)
        page_texts = [(page_num, page.extract_text() or "") for page_num, page in enumerate(reader.pages)]
        logger.info(f"Extracted {len(page_texts)} pages from {filename}")

        pii = PresidioPIIAdapter() if settings.safety.pii_enabled else None
        chunker = SemanticChunker()
        all_chunks = []

        for page_num, text in page_texts:
            anonymized = pii.anonymize(text) if pii else text
            page_chunks = chunker.chunk(anonymized, {"doc_id": str(file_path), "page_number": page_num, "filename": file_path.name})
            all_chunks.extend(page_chunks)

        doc_version_uuid = str(uuid.uuid4())
        retriever = ChromaRetriever()
        retriever.index(all_chunks)
        retriever.close()
        ledger.record_ingestion(filename=filename, file_hash=file_hash, doc_version_uuid=doc_version_uuid, chunk_count=len(all_chunks))
        logger.info(f"Ingested {filename}: {len(all_chunks)} chunks")
        return {"status": "ingested", "filename": filename, "chunk_count": len(all_chunks), "doc_version_uuid": doc_version_uuid}


# ---------------------------------------------------------------------------
# Query Intake Chain (Chain of Responsibility)
# ---------------------------------------------------------------------------


def intake_chain(query: Query, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    settings = get_settings()
    with timed_step("Stage 1 – Query Intake Chain", logger):
        logger.info(f"Original query: {query.text[:100]}")
        result = {"query": query, "contextualized": None, "routing": None, "safety": None, "halted": False, "halt_reason": None}

        contextualized = FastCorefResolver().contextualize(query.text, history)
        result["contextualized"] = contextualized
        logger.info(f"Contextualized: {contextualized.resolved_query[:100]}")

        routing = ZeroShotIntentClassifier().classify(contextualized.resolved_query)
        result["routing"] = routing
        logger.info(f"Routing: intent={routing.intent}, confidence={routing.confidence:.3f}, in_scope={routing.is_in_scope}")

        if not routing.is_in_scope:
            result["halted"] = True
            result["halt_reason"] = f"Out of scope: {routing.intent}"
            logger.warning(f"Query out of scope: {routing.intent}")
            return result

        composite = CompositeSafetyGate()
        safety = composite.check(contextualized.resolved_query)
        result["safety"] = safety

        logger.info(f"Safety: passed={safety.passed}, reason={safety.reason}")

        if not safety.passed:
            result["halted"] = True
            result["halt_reason"] = f"Safety: {safety.reason}"
            return result

        logger.success("Intake chain passed")
        return result


# ---------------------------------------------------------------------------
# Stage 4 – Retrieval + Reranking + Circuit Breaker
# ---------------------------------------------------------------------------


def retrieve_and_rerank(query: str, k: int = 0) -> dict[str, Any]:
    settings = get_settings()
    with timed_step("Stage 2 – Retrieval & Reranking", logger):
        logger.info(f"Query: {query[:100]}")
        hybrid = HybridRetriever(ChromaRetriever(), get_bm25())
        retrieval_result = hybrid.retrieve(query, k)
        logger.info(f"Dense+sparse retrieval returned {len(retrieval_result.chunks)} chunks")
        reranked = CrossEncoderReranker().rerank(query, retrieval_result)
        breaker_result = CircuitBreaker(threshold=settings.thresholds.retrieval_circuit_breaker).check(reranked)
        best_score = reranked.scores[0] if reranked.scores else 0.0
        logger.info(f"Reranked {len(reranked.chunks)} chunks, best score={best_score:.4f}")
        logger.info(f"Circuit breaker: passed={breaker_result['passed']}, reason={breaker_result.get('reason', 'N/A')}")
        return {"retrieval": retrieval_result, "reranked": reranked, "circuit_breaker": breaker_result,
                "halted": not breaker_result["passed"], "halt_reason": breaker_result["reason"] if not breaker_result["passed"] else None}


# ---------------------------------------------------------------------------
# Stage 5 – Context Compression
# ---------------------------------------------------------------------------


def compress_context(query: str, reranked) -> dict:
    with timed_step("Stage 3 – Context Compression", logger):
        result = SentenceCompressor().compress(query, reranked)
        logger.info(f"Compressed from {result.get('original_token_count', 0)} to {result.get('token_count', 0)} tokens, {len(result.get('chunks', []))} chunks kept")
        return result


# ---------------------------------------------------------------------------
# Stage 6 – Answer Synthesis
# ---------------------------------------------------------------------------


def build_synthesizer():
    return NIMSynthesizer()


def synthesize_answer(context: dict, query: str) -> dict:
    with timed_step("Stage 4 – Synthesis", logger):
        syn = build_synthesizer()
        result = syn.synthesize(context, query)
        logger.info(f"Synthesis complete: {len(result.get('answer', ''))} chars")
        return result


# ---------------------------------------------------------------------------
# Stage 7 – Verification & Self-Correction
# ---------------------------------------------------------------------------


def verify_and_correct(answer: dict, context: dict, query: str) -> dict:
    settings = get_settings()
    verifier = NLIVerifier()
    current = answer
    max_attempts = settings.thresholds.correction_loop_max_attempts

    with timed_step("Stage 5 – Verification & Self-Correction", logger):
        for attempt in range(max_attempts):
            logger.info(f"Verification attempt {attempt + 1}/{max_attempts}")
            verification = verifier.verify(current, context)
            if verification["passed"]:
                verification["correction_attempts"] = attempt
                logger.success(f"Verification passed after {attempt} correction(s)")
                return verification

            logger.warning(f"Verification failed on attempt {attempt + 1}")
            if attempt + 1 >= max_attempts:
                verification["max_attempts_reached"] = True
                verification["correction_attempts"] = attempt + 1
                logger.warning("Max correction attempts reached")
                return verification

            unsupported = [v["sentence"] for v in verification["sentence_verdicts"] if not v["passed"]]
            logger.info(f"Unsupported statements: {len(unsupported)}")
            context_text = "\n\n".join(f"[{c.chunk_id}] {c.content}" for c in context.get("chunks", []))
            correction_prompt = CORRECTION_LOOP_PROMPT.format(
                previous_answer=current.get("answer", ""),
                unsupported_statements="\n".join(unsupported),
                context=context_text,
                query=query,
            )
            syn = build_synthesizer()
            current = syn.synthesize({"chunks": context.get("chunks", [])}, correction_prompt)

        return {"passed": False, "sentence_verdicts": [], "correction_attempts": max_attempts,
                "max_attempts_reached": True, "final_answer": current.get("answer", ""),
                "verified_chunk_ids": set()}


# ---------------------------------------------------------------------------
# Full Pipeline Run
# ---------------------------------------------------------------------------


def run_pipeline(query_text: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    from rag_pipeline.core.pipeline.orchestrator import PipelineOrchestrator
    return PipelineOrchestrator().run(query_text, history)
