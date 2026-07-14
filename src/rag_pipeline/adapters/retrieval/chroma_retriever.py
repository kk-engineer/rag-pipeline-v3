from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from rag_pipeline.config.settings import get_logger, get_settings
from rag_pipeline.core.models import Chunk, RetrievalResult
from rag_pipeline.utils.model_loading import resolve_model_path

logger = get_logger()


class ChromaRetriever:

    def __init__(self, collection_name: str = "", persist_directory: str = "", model_name: str = ""):
        settings = get_settings()
        self._persist_dir = persist_directory or settings.paths.chroma_db_path
        self._model_name = model_name or settings.models.chunking_embedding
        self._collection_name = collection_name or settings.paths.chroma_collection_name
        self._default_tenant_id = settings.machine.default_tenant_id
        model_path = resolve_model_path(settings, self._model_name)
        self._model = SentenceTransformer(model_path)
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": settings.paths.chroma_index_space},
        )
        logger.info(f"ChromaRetriever initialised; path={self._persist_dir}, model={self._model_name}")

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [{"doc_id": c.doc_id, "page_number": c.page_number,
                       "chunk_index": c.chunk_index, "filename": c.filename,
                       "tenant_id": c.tenant_id, "doc_version_uuid": c.doc_version_uuid} for c in chunks]
        embeddings = self._model.encode(documents, convert_to_numpy=True).tolist()
        self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Indexed {len(chunks)} chunks")

    def retrieve(self, query: str, k: int = 0) -> RetrievalResult:
        settings = get_settings()
        top_k = k if k > 0 else settings.thresholds.default_retrieval_k
        q_emb = self._model.encode(query, convert_to_numpy=True).tolist()
        results = self._collection.query(query_embeddings=[q_emb], n_results=top_k,
                                         include=["documents", "metadatas", "distances"])
        chunks = []
        scores = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i]
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - dist
                chunk = Chunk(chunk_id=doc_id, content=content,
                              doc_id=meta.get("doc_id", ""),
                              page_number=meta.get("page_number", 0),
                              chunk_index=meta.get("chunk_index", 0),
                              filename=meta.get("filename", ""),
                              tenant_id=meta.get("tenant_id", self._default_tenant_id),
                              doc_version_uuid=meta.get("doc_version_uuid", ""))
                chunks.append(chunk)
                scores.append(score)
        logger.info(f"Retrieved {len(chunks)} chunks for query={query[:50]}")
        return RetrievalResult(chunks=chunks, scores=scores, method="dense")
