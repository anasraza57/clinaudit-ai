"""
Tests for the PubMedBERT embedding service.

Uses a tiny model for fast testing. The real PubMedBERT model (~440MB)
is too large for unit tests — integration tests with the real model
will be added later.
"""

import numpy as np
import pytest

from src.services.embedder import Embedder, get_embedder


class TestEmbedder:
    """Test the Embedder service."""

    def test_initial_state(self):
        embedder = Embedder()
        assert not embedder.is_loaded
        assert embedder.dimension == 0

    def test_encode_without_load_raises(self):
        embedder = Embedder()
        with pytest.raises(RuntimeError, match="not loaded"):
            embedder.encode("test")

    def test_encode_batch_without_load_raises(self):
        embedder = Embedder()
        with pytest.raises(RuntimeError, match="not loaded"):
            embedder.encode_batch(["test"])

    def test_unload(self):
        embedder = Embedder()
        # Even without loading, unload should not crash
        embedder.unload()
        assert not embedder.is_loaded

    def test_singleton(self):
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2


class TestEmbedderWithModel:
    """
    Test actual embedding using a tiny model.

    We use 'prajjwal1/bert-tiny' (~17MB) instead of PubMedBERT (~440MB)
    for fast unit tests. The encoding logic is the same — only the
    model weights and dimension differ.
    """

    @pytest.fixture(autouse=True)
    def _load_tiny_model(self):
        """Load a tiny BERT model for testing."""
        self.embedder = Embedder()
        self.embedder.load(model_name="prajjwal1/bert-tiny")
        # Override dimension to match tiny model (128-dim)
        self.embedder._dimension = 128
        yield
        self.embedder.unload()

    def test_load_sets_loaded(self):
        assert self.embedder.is_loaded

    def test_encode_returns_correct_shape(self):
        vec = self.embedder.encode("low back pain")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.shape == (128,)  # bert-tiny has 128 dimensions

    def test_encode_is_normalised(self):
        vec = self.embedder.encode("low back pain")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_encode_batch_returns_correct_shape(self):
        texts = ["low back pain", "osteoarthritis of knee", "carpal tunnel syndrome"]
        vecs = self.embedder.encode_batch(texts)
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (3, 128)

    def test_encode_batch_all_normalised(self):
        texts = ["query one", "query two"]
        vecs = self.embedder.encode_batch(texts)
        for i in range(len(texts)):
            norm = np.linalg.norm(vecs[i])
            assert abs(norm - 1.0) < 1e-5

    def test_encode_batch_empty_list(self):
        vecs = self.embedder.encode_batch([])
        assert vecs.shape[0] == 0

    def test_different_texts_produce_different_embeddings(self):
        v1 = self.embedder.encode("low back pain")
        v2 = self.embedder.encode("knee replacement surgery")
        # They shouldn't be identical
        assert not np.allclose(v1, v2)

    def test_same_text_produces_same_embedding(self):
        v1 = self.embedder.encode("low back pain")
        v2 = self.embedder.encode("low back pain")
        assert np.allclose(v1, v2)
