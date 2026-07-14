from __future__ import annotations

SYNTHESIS_SYSTEM_PROMPT = """\
You are a precise, grounded assistant. Answer the user's question using ONLY \
the provided context. If the context does not contain enough information to \
answer, say so.

Context:
{context}

Question: {query}

Provide a concise, well-structured answer using only the provided context."""

QUERY_REWRITE_FALLBACK_PROMPT = """\
Rewrite the following query to be self-contained, resolving any pronouns or \
ambiguous references using the conversation history.

Conversation history:
{history}

Original query: {query}

Rewritten query:"""

CORRECTION_LOOP_PROMPT = """\
The previous answer contained statements not fully supported by the context. \
Revise the answer to only include information that is directly supported.

Previous answer: {previous_answer}

Unsupported statements: {unsupported_statements}

Context:
{context}

Question: {query}

Provide a corrected answer using only the provided context."""

SAFETY_REFUSAL_RESPONSE = (
    "I cannot process this request. It was flagged by the safety system."
)

CIRCUIT_BREAKER_RESPONSE = (
    "I cannot answer this question based on the available documents. "
    "The retrieved context does not contain sufficient information "
    "to provide a reliable answer."
)
