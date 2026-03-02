"""
FAISS vector store management.

Loads the pre-built FAISS index and the guidelines CSV, providing
a search interface used by the Retriever agent to find relevant
clinical guidelines for a given query embedding.
"""

import csv
import gzip
import logging
import shutil
from pathlib import Path

import faiss
import numpy as np

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages the FAISS index and its corresponding guideline metadata.

    The FAISS index maps vector positions to rows in guidelines.csv,
    so both must be loaded together.
    """

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._guidelines: list[dict] = []
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def load(
        self,
        index_path: str | None = None,
        guidelines_path: str | None = None,
    ) -> None:
        """
        Load the FAISS index and guidelines metadata from disk.

        Args:
            index_path: Path to the .index file. Defaults to settings.
            guidelines_path: Path to the guidelines CSV. Defaults to settings.
        """
        settings = get_settings()
        idx_path = Path(index_path or settings.faiss_index_path)
        csv_path = Path(guidelines_path or settings.guidelines_csv_path)

        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {idx_path}")
        if not csv_path.exists():
            # Auto-decompress .gz if the uncompressed file is missing
            gz_path = Path(str(csv_path) + ".gz")
            if gz_path.exists():
                logger.info("Decompressing %s → %s", gz_path, csv_path)
                with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            else:
                raise FileNotFoundError(f"Guidelines CSV not found: {csv_path}")

        logger.info("Loading FAISS index from %s", idx_path)
        self._index = faiss.read_index(str(idx_path))
        logger.info(
            "FAISS index loaded: %d vectors, dimension %d",
            self._index.ntotal,
            self._index.d,
        )

        logger.info("Loading guidelines metadata from %s", csv_path)
        self._guidelines = []
        csv.field_size_limit(10_000_000)  # Some guideline texts exceed default 131KB
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._guidelines.append(row)

        if self._index.ntotal != len(self._guidelines):
            logger.warning(
                "Index/CSV size mismatch: %d vectors vs %d guidelines. "
                "Results may be misaligned.",
                self._index.ntotal,
                len(self._guidelines),
            )

        self._loaded = True
        logger.info("Vector store ready (%d entries)", len(self._guidelines))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Search the FAISS index for the most similar guidelines.

        Args:
            query_embedding: A 1-D numpy array (float32) of the query vector.
            top_k: Number of results to return. Defaults to settings.retriever_top_k.

        Returns:
            List of dicts, each containing guideline metadata plus
            'score' (L2 distance — lower is more similar) and 'rank'.
        """
        if not self._loaded or self._index is None:
            raise RuntimeError("Vector store not loaded. Call load() first.")

        settings = get_settings()
        k = top_k or settings.retriever_top_k

        # FAISS expects a contiguous 2-D float32 array of shape (n_queries, dimension)
        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances, indices = self._index.search(query, k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                # FAISS returns -1 for unfilled slots
                continue
            if idx >= len(self._guidelines):
                logger.warning("Index %d out of range for guidelines list", idx)
                continue

            entry = dict(self._guidelines[idx])
            entry["score"] = float(dist)
            entry["rank"] = rank + 1
            results.append(entry)

        return results

    def unload(self) -> None:
        """Release memory used by the index and metadata."""
        self._index = None
        self._guidelines = []
        self._loaded = False
        logger.info("Vector store unloaded")


# Module-level singleton
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the singleton VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
