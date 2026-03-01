"""
Tests for the VectorStore service.

Uses a small in-memory FAISS index to test search behaviour
without needing the full guidelines dataset.
"""

import csv
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pytest

from src.services.vector_store import VectorStore


@pytest.fixture()
def small_index(tmp_path):
    """Create a tiny FAISS index with 3 vectors and matching CSV."""
    dim = 4
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    index_path = tmp_path / "test.index"
    faiss.write_index(index, str(index_path))

    csv_path = tmp_path / "test_guidelines.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "title", "clean_text", "url", "overview"])
        writer.writeheader()
        writer.writerow({"id": "aaa", "source": "nice", "title": "Guideline A", "clean_text": "Text A", "url": "", "overview": ""})
        writer.writerow({"id": "bbb", "source": "nice", "title": "Guideline B", "clean_text": "Text B", "url": "", "overview": ""})
        writer.writerow({"id": "ccc", "source": "nice", "title": "Guideline C", "clean_text": "Text C", "url": "", "overview": ""})

    return str(index_path), str(csv_path)


class TestVectorStore:
    def test_initial_state(self):
        vs = VectorStore()
        assert not vs.is_loaded
        assert vs.size == 0

    def test_load(self, small_index):
        index_path, csv_path = small_index
        vs = VectorStore()
        vs.load(index_path=index_path, guidelines_path=csv_path)
        assert vs.is_loaded
        assert vs.size == 3

    def test_search_returns_correct_nearest(self, small_index):
        index_path, csv_path = small_index
        vs = VectorStore()
        vs.load(index_path=index_path, guidelines_path=csv_path)

        # Query close to the first vector [1,0,0,0]
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = vs.search(query, top_k=2)

        assert len(results) == 2
        assert results[0]["title"] == "Guideline A"
        assert results[0]["rank"] == 1
        assert "score" in results[0]

    def test_search_top_k(self, small_index):
        index_path, csv_path = small_index
        vs = VectorStore()
        vs.load(index_path=index_path, guidelines_path=csv_path)

        query = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        results = vs.search(query, top_k=1)
        assert len(results) == 1

    def test_search_without_load_raises(self):
        vs = VectorStore()
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(RuntimeError, match="not loaded"):
            vs.search(query)

    def test_load_missing_index_raises(self, tmp_path):
        vs = VectorStore()
        with pytest.raises(FileNotFoundError):
            vs.load(
                index_path=str(tmp_path / "nonexistent.index"),
                guidelines_path=str(tmp_path / "nonexistent.csv"),
            )

    def test_unload(self, small_index):
        index_path, csv_path = small_index
        vs = VectorStore()
        vs.load(index_path=index_path, guidelines_path=csv_path)
        assert vs.is_loaded

        vs.unload()
        assert not vs.is_loaded
        assert vs.size == 0
