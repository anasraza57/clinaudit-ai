# Guideline Evidence Finder Explained

## What Does the Guideline Evidence Finder Do?

The Guideline Evidence Finder is **Stage 3** of the 4-agent pipeline. It takes the search queries from the Audit Query Generator, converts them into numerical vectors using PubMedBERT, and searches the FAISS guideline index to find the most relevant NICE guideline passages for each diagnosis.

Think of it as a research assistant who takes a set of search terms, goes into a massive digital library, and returns the most relevant documents for each topic.

## The Two Components

The Guideline Evidence Finder is actually built from two services working together:

### 1. The Embedder (`src/services/embedder.py`)

The Embedder converts text into numbers. Specifically, it takes a query string like "NICE guidelines for low back pain management" and converts it into a **768-dimensional vector** — a list of 768 numbers that represent the meaning of that text in a mathematical space.

It uses **PubMedBERT Matryoshka** (`NeuML/pubmedbert-base-embeddings-matryoshka`), a version of BERT that was specifically trained on biomedical literature. This means it understands medical terminology — it knows that "osteoarthritis" and "degenerative joint disease" are related, even though the words look completely different.

**How encoding works, step by step:**

```
Input: "NICE guidelines for low back pain management"
  ↓
1. Tokenise — split into subword tokens: ["NICE", "guide", "##lines", "for", "low", "back", "pain", ...]
  ↓
2. Forward pass through PubMedBERT — each token gets a 768-dim contextual embedding
  ↓
3. Mean pooling — average all token embeddings into one 768-dim vector
  ↓
4. L2 normalise — scale the vector to unit length (so cosine similarity = dot product)
  ↓
Output: [0.023, -0.157, 0.089, ..., 0.041]  (768 numbers)
```

**Why mean pooling?** BERT outputs one embedding per token. We need one embedding per sentence. Mean pooling (averaging all token embeddings) is a simple, effective way to combine them. It's what Cyprian used when building the FAISS index, so we must use the same approach for our queries to be comparable.

**Why L2 normalisation?** After normalising, the dot product between two vectors equals their cosine similarity. FAISS can compute dot products very efficiently, so normalising lets us use cosine similarity (which measures meaning similarity) through fast inner product search.

### 2. The Vector Store (`src/services/vector_store.py`)

This was built in Phase 1. It loads the pre-built FAISS index (1,656 guideline vectors) and provides a search interface. When given a query vector, it returns the top-K most similar guideline entries with their metadata (title, text, URL, similarity score).

### How They Work Together

```
Audit Query Generator produces:  "NICE guidelines for low back pain management"
                          ↓
Embedder encodes it:   [0.023, -0.157, 0.089, ..., 0.041]  (768 floats)
                          ↓
VectorStore searches:  Compares against 1,656 guideline vectors
                          ↓
Returns top-5:         [
                         {title: "Low back pain and sciatica in over 16s", score: 0.12, ...},
                         {title: "Chronic pain management guidelines", score: 0.31, ...},
                         ...
                       ]
```

## What the Guideline Evidence Finder Adds

The Embedder and VectorStore are low-level services. The Guideline Evidence Finder (`src/agents/retriever.py`) adds the **intelligence layer** on top:

### 1. Batch Encoding

Each diagnosis has 1-3 queries. Instead of encoding them one at a time (separate forward passes through PubMedBERT), the Guideline Evidence Finder uses `encode_batch()` to encode all queries for a diagnosis in a **single forward pass**. This is faster and uses less memory — the model weights are loaded into the CPU cache once instead of per-query.

### 2. Multi-Query Aggregation

After batch encoding, the Guideline Evidence Finder searches FAISS for each query embedding separately, then **merges the results**. This means if Query 1 and Query 3 both find the same guideline, we keep it once (with the better score).

### 3. Deduplication

Without deduplication, we might send the Compliance Auditor Agent 3 copies of the same guideline (found by 3 different queries). The Guideline Evidence Finder keeps only unique guidelines, tracked by their `id` field.

### 4. Best-Score Selection

When the same guideline is found by multiple queries, we keep the one with the **best similarity score** (lowest L2 distance). This ensures the Compliance Auditor Agent gets the most confident matches.

### 5. Top-K Limiting

After merging and deduplicating across all queries for a diagnosis, we keep only the top-K results (default: 5, configurable via `RETRIEVER_TOP_K`). This prevents information overload for the Compliance Auditor Agent.

### 6. Structured Output

The Guideline Evidence Finder produces a `RetrievalResult` with `DiagnosisGuidelines` for each diagnosis, containing `GuidelineMatch` objects with title, text, URL, score, rank, and which query found it.

## Data Flow

```
QueryResult (from Audit Query Generator)
│
├── Diagnosis: "Low back pain" (3 queries)
│   ├── Batch encode all 3 queries → 3 embeddings (single forward pass)
│   ├── Embedding 1 → FAISS search → 5 results
│   ├── Embedding 2 → FAISS search → 5 results
│   └── Embedding 3 → FAISS search → 5 results
│   └── Merge + dedup + top-5 → 5 unique guidelines
│
├── Diagnosis: "Osteoarthritis of knee" (3 queries)
│   ├── Batch encode all 3 queries → 3 embeddings (single forward pass)
│   ├── ...
│   └── Merge + dedup + top-5 → 5 unique guidelines
│
└── RetrievalResult
    ├── pat_id: "pat-001"
    ├── total_diagnoses: 2
    ├── total_guidelines: 10
    └── diagnosis_guidelines: [
          {diagnosis: "Low back pain", guidelines: [5 matches with titles, texts, scores]},
          {diagnosis: "Osteoarthritis", guidelines: [5 matches with titles, texts, scores]},
        ]
```

## What Happens Next?

The `RetrievalResult` is passed to the **Compliance Auditor Agent** (Stage 4), which:
1. Takes each diagnosis + its matched guideline texts
2. Also receives the patient's treatments, referrals, and investigations
3. Asks the LLM to evaluate whether the documented care matches the guidelines
4. Produces a per-diagnosis adherence score (+1 adherent / -1 non-adherent) with explanations

## Cyprian's Previous Approach

### What He Did

Cyprian's retriever was a function inside a Jupyter notebook:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer_emb = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings-matryoshka")
model_emb = AutoModel.from_pretrained("NeuML/pubmedbert-base-embeddings-matryoshka")

def embed_text(text):
    inputs = tokenizer_emb(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_emb(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().astype('float32')
    faiss.normalize_L2(emb.reshape(1, -1))
    return emb

def retriever_node(state):
    retrieved = []
    concepts = state['diagnoses'] + state['treatments'] + state['procedures']
    for concept in concepts:
        query = f"guidelines for {concept.lower().replace(' ', '_')}"
        query_emb = embed_text(query)
        D, I = index.search(query_emb.reshape(1, -1), 5)
        top_guidelines = [(float(D[0][j]), all_chunks[I[0][j]]) for j in range(len(I[0]))]
        retrieved.append(top_guidelines)
    state['retrieved_guidelines'] = retrieved
    return state
```

### Problems with Cyprian's Approach

1. **Naive query construction** — he just prepends "guidelines for" to the raw concept and replaces spaces with underscores. "guidelines for low_back_pain" is a much worse query than "NICE guidelines for assessment and management of low back pain and sciatica".

2. **No deduplication** — if multiple concepts return the same guideline, it appears multiple times. The Scorer wastes time evaluating the same guideline repeatedly.

3. **No separation of concerns** — the embedding model, FAISS search, and retrieval logic are all mixed together in one function inside a notebook. Nothing is reusable or testable.

4. **Global state** — the model, tokenizer, and FAISS index are all global variables in the notebook. No loading/unloading, no error handling.

5. **Searches for everything** — he searches for diagnoses, treatments, AND procedures. But the Compliance Auditor Agent only evaluates per-diagnosis, so searching for "steroid injection guidelines" doesn't help — we need guidelines for the *condition*, not the treatment.

6. **Hardcoded top-K** — always returns exactly 5 results with no configuration.

7. **`faiss.normalize_L2` crash potential** — calls `faiss.normalize_L2` on tensors that may not be writable numpy arrays (we hit this exact bug and fixed it by using numpy normalization instead).

### What We Improved

| Aspect | Cyprian | Ours |
|---|---|---|
| **Query quality** | "guidelines for low_back_pain" | Expert-crafted templates + LLM queries from Audit Query Generator |
| **Architecture** | One function in a notebook | Separate Embedder service + VectorStore + RetrieverAgent |
| **Deduplication** | None — same guideline appears multiple times | Deduped by guideline ID, best score kept |
| **What gets searched** | Diagnoses + treatments + procedures (noisy) | Only diagnoses (what the Compliance Auditor Agent actually evaluates) |
| **Model management** | Global variables, never unloaded | Singleton with load/unload, proper memory management |
| **Error handling** | None | RuntimeError if not loaded, logging throughout |
| **Configuration** | Hardcoded k=5 | Configurable via `RETRIEVER_TOP_K` setting |
| **Relevance filtering** | None — all top-K returned blindly | Two-layer filter: title exclusion + topic matching + L2 threshold |
| **Testing** | None | 34 unit tests (13 embedder + 21 retriever) |
| **Portability** | Google Colab notebook | Standard Python modules, works anywhere |

## Memory Safety: PyTorch Tensors and FAISS

We encountered two related segfault issues during development, both caused by the interaction between PyTorch tensors, numpy arrays, and FAISS's C++ code.

### Bug 1: faiss.normalize_L2 segfault

`faiss.normalize_L2()` crashed when called on numpy arrays from PyTorch tensor operations — the arrays were not writable/contiguous in memory.

**Fix:** Replaced with pure numpy normalization:

```python
# Before (crashes):
faiss.normalize_L2(embedding.reshape(1, -1))

# After (works):
norm = np.linalg.norm(embedding)
if norm > 0:
    embedding = embedding / norm
```

### Bug 2: Tensor memory lifecycle crashes

Calling `.numpy()` on a PyTorch tensor without `.detach()` leaves the numpy array sharing memory with the tensor's computation graph. When the tensor is garbage-collected, the numpy array points to freed memory — causing segfaults in FAISS or later code.

**Fix:** Three changes in the embedder:

```python
# 1. Detach before numpy conversion
embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# 2. Explicitly free tensors after use
del outputs, inputs

# 3. Return contiguous arrays for FAISS compatibility
return np.ascontiguousarray(embedding)
```

The vector store also enforces contiguous arrays before FAISS search:
```python
query = np.ascontiguousarray(query_embedding, dtype=np.float32)
```

## HuggingFace Tokenizers Parallelism Segfault

During development, the server kept crashing with a segmentation fault during PubMedBERT encoding. Adding `faulthandler.enable()` revealed the crash happened inside `concurrent/futures/thread.py` — a threading conflict.

**Root cause:** HuggingFace tokenizers use **Rust-based parallelism** (rayon) internally for tokenization. These Rust threads conflict with uvicorn's async event loop and Python thread pools on macOS, causing segfaults.

**Fix (in `src/main.py`, before any imports):**
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable Rust parallelism
os.environ["OMP_NUM_THREADS"] = "1"              # Limit PyTorch/OpenMP threads
```

This forces single-threaded tokenization, which is still fast enough for our query sizes (1-3 short sentences per diagnosis). The fix may not be needed on Linux/Docker where threading behaviour differs.

## Post-Retrieval Relevance Filtering (Phase 10a)

### The Problem

After running 50-patient audits with both OpenAI and Ollama, a comparison analysis revealed that 80% of inter-model disagreements were caused by the FAISS retriever returning **irrelevant guidelines**. For example:

- "Carpal tunnel syndrome" → retrieved "Recent-onset chest pain of suspected cardiac origin" guidelines
- "Foot pain" → retrieved "Diabetic foot problems" guidelines
- "Primary prevention of cardiovascular disease" → retrieved MSK-focused guidelines

OpenAI's scorer masked these by giving credit for any referral regardless of the guideline context. Ollama's scorer correctly flagged them as "NOT RELEVANT" (score 0, confidence 1.0). The root cause: the 277K-guideline FAISS corpus includes oncology, cardiology, endocrinology, etc., and PubMedBERT embeddings aren't specific enough to distinguish MSK from non-MSK at the margins.

### The Solution

A two-layer post-retrieval filter in `_filter_irrelevant()`, applied **after** FAISS search + dedup but **before** top-K truncation:

**Layer A — Title-based filtering (zero cost):**
1. **Excluded specialty terms** — Guidelines whose titles contain terms like "cancer", "cardiac", "diabetic", "pregnancy" are excluded for MSK diagnoses. Defined in `_EXCLUDE_TITLE_TERMS`.
2. **Topic overlap check** — The diagnosis and guideline title are each mapped to topic tags (e.g., "spine", "knee", "shoulder") using `_TOPIC_KEYWORDS`. If both have topic tags but none overlap, the guideline is filtered.

**Layer B — Distance threshold:**
- Guidelines with L2 distance > `RETRIEVER_MIN_SIMILARITY` (default 1.2) are removed. High distance means weak semantic match.

**Fallback safety:**
- If ALL guidelines are filtered, the best match is kept as a fallback, so the scorer never receives zero context.

### Data Flow (Updated)

```
Audit Query Generator produces:  "NICE guidelines for carpal tunnel syndrome management"
                          ↓
Embedder encodes it:   [0.023, -0.157, 0.089, ..., 0.041]  (768 floats)
                          ↓
VectorStore searches:  Returns top-K from 277K guidelines
                          ↓
Merge + dedup:         5 unique guidelines
                          ↓
Filter irrelevant:     Remove "chest pain", "cancer", etc.     ← NEW
                          ↓
Top-K:                 Return filtered results (up to 5)
```

### Topic Keyword Groups

13 body-region/condition groups are defined in `_TOPIC_KEYWORDS`:

| Group | Example keywords |
|---|---|
| `spine` | back, lumbar, spinal, sciatica, disc, thoracic, cervical |
| `knee` | knee, patella, meniscal, cruciate, acl |
| `hip` | hip, femoral, acetabular, groin |
| `shoulder` | shoulder, rotator, supraspinatus, impingement |
| `hand_wrist` | hand, wrist, carpal, tunnel, finger, thumb |
| `foot_ankle` | foot, ankle, plantar, heel, toe, metatarsal |
| `osteoarthritis` | osteoarthritis, arthritis, arthrosis, joint replacement |
| `fracture` | fracture, broken, fractures |
| ... | (and 5 more groups) |

### Why Not Re-Index?

We considered filtering the guideline corpus to only MSK guidelines, but this would lose cross-specialty references (e.g., cardiovascular risk assessment in rheumatoid arthritis guidelines). Post-retrieval filtering is more surgical — it catches obvious mismatches while preserving genuine cross-references.

## Configuration

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `NeuML/pubmedbert-base-embeddings-matryoshka` | HuggingFace model for encoding queries |
| `EMBEDDING_DIMENSION` | 768 | Vector dimension (must match the FAISS index) |
| `RETRIEVER_TOP_K` | 5 | Number of guideline matches to return per diagnosis |
| `RETRIEVER_MIN_SIMILARITY` | 1.2 | Max L2 distance; results above this are filtered out |

## Important: Why the Same Model Matters

The FAISS index was built by encoding all 1,656 guidelines using PubMedBERT with mean pooling and L2 normalization. For our query embeddings to be **comparable** to the guideline embeddings, we must use:

- The **same model** (`NeuML/pubmedbert-base-embeddings-matryoshka`)
- The **same encoding** (mean pooling of `last_hidden_state`)
- The **same normalization** (L2 normalization to unit vectors)

If we used a different model or different pooling, the vectors would live in a different mathematical space, and similarity scores would be meaningless. This is like trying to compare distances measured in miles with distances measured in kilometres — the numbers don't match unless you're using the same units.
