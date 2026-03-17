"""
Build the FAISS guideline index from guidelines.csv using PubMedBERT.

Reads each guideline's clean_text, encodes it with the same PubMedBERT
model used by the Guideline Evidence Finder, and saves a FAISS IndexFlatL2.

Usage:
    python3 scripts/build_index.py

The script:
1. Loads guidelines from CSV (or .csv.gz)
2. Loads the PubMedBERT embedding model
3. Encodes all guideline texts in batches
4. Builds a FAISS IndexFlatL2 index
5. Saves to data/guidelines.index
"""

import csv
import gzip
import sys
import time
from pathlib import Path

import faiss
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.services.embedder import get_embedder


def load_guidelines(csv_path: Path) -> list[str]:
    """Load clean_text from guidelines CSV (supports .gz)."""
    csv.field_size_limit(10_000_000)

    texts = []
    opener = gzip.open if csv_path.suffix == ".gz" else open
    kwargs = {"mode": "rt", "encoding": "utf-8"} if csv_path.suffix == ".gz" else {"newline": "", "encoding": "utf-8"}

    with opener(csv_path, **kwargs) as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("clean_text", "").strip()
            if text:
                texts.append(text)
            else:
                print(f"  Warning: empty clean_text for guideline {row.get('id', '?')}")
                texts.append(row.get("title", "untitled"))

    return texts


def main():
    settings = get_settings()

    # --- Locate guidelines CSV ---
    csv_path = Path(settings.guidelines_csv_path)
    gz_path = csv_path.with_suffix(".csv.gz")

    if csv_path.exists():
        source = csv_path
    elif gz_path.exists():
        source = gz_path
    else:
        print(f"Error: Neither {csv_path} nor {gz_path} found.")
        sys.exit(1)

    print(f"=== Building FAISS index ===")
    print(f"  Source: {source}")

    # --- Load guideline texts ---
    texts = load_guidelines(source)
    print(f"  Loaded {len(texts)} guidelines")

    # --- Load PubMedBERT embedder ---
    print(f"\n  Loading embedding model: {settings.embedding_model_name}")
    embedder = get_embedder()
    t0 = time.time()
    embedder.load()
    print(f"  Model loaded in {time.time() - t0:.1f}s (dimension={embedder.dimension})")

    # --- Encode all guidelines in batches ---
    batch_size = 32
    all_embeddings = []
    total = len(texts)

    print(f"\n  Encoding {total} guidelines (batch_size={batch_size})...")
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        embeddings = embedder.encode_batch(batch)
        all_embeddings.append(embeddings)

        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        print(f"    [{done}/{total}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining", end="\r")

    print(f"\n  Encoding complete in {time.time() - t0:.1f}s")

    # --- Build FAISS index ---
    matrix = np.vstack(all_embeddings).astype(np.float32)
    assert matrix.shape == (total, embedder.dimension), f"Shape mismatch: {matrix.shape}"

    index = faiss.IndexFlatL2(embedder.dimension)
    index.add(matrix)

    print(f"\n  FAISS index built: {index.ntotal} vectors, dimension {index.d}")

    # --- Save ---
    output_path = Path(settings.faiss_index_path)
    faiss.write_index(index, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.1f} MB)")

    # --- Cleanup ---
    embedder.unload()
    print("\nDone.")


if __name__ == "__main__":
    main()
