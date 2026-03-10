"""Embedding store backed by ChromaDB for semantic similarity search.

Pre-computes embeddings for all issues at startup using sentence-transformers.
Uses ChromaDB for persistent vector storage, automatic indexing, and
metadata-filtered similarity search. Supports multiple issue sources
(local, linear) in a single collection via source metadata.
"""

from __future__ import annotations

import logging

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """ChromaDB-backed embedding index for issue deduplication."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "data/chroma_db",
    ):
        self.model = SentenceTransformer(model_name)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="issues",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB ready at %s (%d vectors)",
            persist_dir,
            self._collection.count(),
        )

    def has_source(self, source: str) -> bool:
        """Check if any issues from the given source are already indexed."""
        result = self._collection.get(where={"source": source}, limit=1)
        return len(result["ids"]) > 0

    def index_issues(self, issues: list[dict], source: str) -> None:
        """Embed and upsert issues into the collection with source metadata.

        Each issue must have at least 'id', 'title', and 'description' keys.
        IDs are prefixed with '{source}:' to avoid collisions between sources.
        """
        if not issues:
            logger.warning("No issues to index for source '%s'", source)
            return

        texts = [f"{i['title']} {i.get('description', '')}" for i in issues]
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

        # ChromaDB expects lists, not numpy arrays
        self._collection.upsert(
            ids=[f"{source}:{i['id']}" for i in issues],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {
                    "source": source,
                    "title": i["title"],
                    "issue_id": i["id"],
                }
                for i in issues
            ],
        )
        logger.info(
            "Indexed %d issues from '%s' (%d dimensions)",
            len(issues),
            source,
            embeddings.shape[1],
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        source: str | None = None,
    ) -> list[dict]:
        """Find the most semantically similar issues to a query string.

        Returns a list of dicts with keys: id, title, score, source.
        Optionally filter by source (e.g. 'local' or 'linear').
        """
        if self._collection.count() == 0:
            logger.warning("No issues indexed in ChromaDB")
            return []

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )

        where = {"source": source} if source else None
        results = self._collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=where,
        )

        # ChromaDB returns distances; for cosine space, score = 1 - distance
        output = []
        for meta, dist, doc in zip(
            results["metadatas"][0],
            results["distances"][0],
            results["documents"][0],
        ):
            output.append(
                {
                    "id": meta["issue_id"],
                    "title": meta["title"],
                    "score": float(1.0 - dist),
                    "source": meta["source"],
                    "document": doc,
                }
            )
        return output

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode arbitrary texts for on-the-fly similarity comparison."""
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
