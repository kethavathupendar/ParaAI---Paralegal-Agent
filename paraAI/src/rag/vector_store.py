"""
Thin wrapper around a FAISS index so the rest of the app depends on a
simple `.upsert()` / `.query()` interface, not on FAISS directly. Swapping
to Qdrant/Pinecone/pgvector in production means editing only this file.

The index is persisted to disk (settings.vector_store_path) so restarts
don't lose case history, and every vector carries `case_id`/`doc_id`
metadata so retrieval can be scoped to one case and never leak across
clients — an important guarantee for a legal tool.
"""
from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import get_settings
from src.rag.multimodal_loader import Chunk


class VectorStore:
    _instance: Optional["VectorStore"] = None
    _lock = threading.Lock()

    def __init__(self):
        settings = get_settings()
        self.path = Path(settings.vector_store_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._embedder = None  # lazy-loaded, sentence-transformers is heavy to import
        self._index = None
        self._id_map: list[dict[str, Any]] = []  # parallel array: faiss row -> chunk metadata
        self._load_or_init()

    # --- singleton accessor so agents share one in-memory index per process ---
    @classmethod
    def get(cls) -> "VectorStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = VectorStore()
            return cls._instance

    def _embedder_model(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            settings = get_settings()
            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    def _load_or_init(self) -> None:
        import faiss

        index_file = self.path / "index.faiss"
        meta_file = self.path / "meta.pkl"
        if index_file.exists() and meta_file.exists():
            self._index = faiss.read_index(str(index_file))
            with open(meta_file, "rb") as f:
                self._id_map = pickle.load(f)
        else:
            dim = 768  # matches default embedding model; adjust if you swap models
            self._index = faiss.IndexFlatIP(dim)
            self._id_map = []

    def _persist(self) -> None:
        import faiss

        faiss.write_index(self._index, str(self.path / "index.faiss"))
        with open(self.path / "meta.pkl", "wb") as f:
            pickle.dump(self._id_map, f)

    def upsert(self, chunks: list[Chunk], case_id: str) -> int:
        """Embeds and appends chunks to the index. Returns number indexed."""
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        vectors = self._embedder_model().encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        vectors = np.asarray(vectors, dtype="float32")
        self._index.add(vectors)
        for c in chunks:
            self._id_map.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "case_id": case_id,
                    "modality": c.modality,
                    "text": c.text,
                    "metadata": c.metadata,
                }
            )
        self._persist()
        return len(chunks)

    def query(self, text: str, case_id: str, top_k: int = 6) -> list[dict[str, Any]]:
        if self._index.ntotal == 0:
            return []
        query_vec = self._embedder_model().encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )
        query_vec = np.asarray(query_vec, dtype="float32")
        # over-fetch then filter by case_id, since FAISS flat index has no
        # native metadata filter
        scores, idxs = self._index.search(query_vec, min(top_k * 5, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            record = self._id_map[idx]
            if record["case_id"] != case_id:
                continue
            results.append({**record, "score": float(score)})
            if len(results) >= top_k:
                break
        return results
