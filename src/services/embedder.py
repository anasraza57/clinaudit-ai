"""
PubMedBERT embedding service.

Loads the PubMedBERT Matryoshka model and provides a method to encode
text into 768-dimensional vectors suitable for FAISS similarity search.

The same model and encoding approach was used to build the pre-built
FAISS index (by Cyprian), so our query embeddings will be comparable
to the guideline embeddings in the index.

Encoding steps (matching Cyprian's approach):
1. Tokenise text (padding, truncation at 512 tokens)
2. Forward pass through PubMedBERT (no gradients)
3. Mean pooling over last_hidden_state
4. L2 normalisation (so FAISS inner product = cosine similarity)
"""

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Encodes text into PubMedBERT embeddings.

    Usage:
        embedder = get_embedder()
        embedder.load()
        vector = embedder.encode("low back pain guidelines")
        vectors = embedder.encode_batch(["query 1", "query 2"])
        embedder.unload()
    """

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._model_name: str = ""
        self._dimension: int = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dimension(self) -> int:
        return self._dimension

    def load(self, model_name: str | None = None) -> None:
        """
        Load the PubMedBERT tokeniser and model into memory.

        Args:
            model_name: HuggingFace model ID. Defaults to settings.
        """
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model_name
        self._dimension = settings.embedding_dimension

        logger.info("Loading embedding model: %s", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()  # Set to inference mode
        self._loaded = True
        logger.info(
            "Embedding model loaded (dimension=%d)",
            self._dimension,
        )

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a normalised embedding vector.

        Args:
            text: The text to encode.

        Returns:
            A 1-D float32 numpy array of shape (dimension,), L2-normalised.
        """
        if not self._loaded:
            raise RuntimeError("Embedder not loaded. Call load() first.")

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean pooling: average all token embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().astype("float32")

        # L2 normalise so FAISS inner product = cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode multiple texts into normalised embedding vectors.

        Args:
            texts: List of text strings to encode.

        Returns:
            A 2-D float32 numpy array of shape (len(texts), dimension),
            each row L2-normalised.
        """
        if not self._loaded:
            raise RuntimeError("Embedder not loaded. Call load() first.")

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean pooling for each text in the batch
        # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_dim)
        # attention_mask expands to match hidden_dim for proper masking
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        sum_embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1)

        embeddings = (sum_embeddings / count).numpy().astype("float32")

        # L2 normalise each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        embeddings = embeddings / norms

        return embeddings

    def unload(self) -> None:
        """Release model and tokeniser from memory."""
        self._tokenizer = None
        self._model = None
        self._loaded = False
        logger.info("Embedding model unloaded")


# Module-level singleton
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the singleton Embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
