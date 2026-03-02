# 11. Embeddings, Transformers, and Why Index Building Is Slow

## The Question

When we run `scripts/build_index.py` to encode 1,656 guidelines with PubMedBERT, it takes 5–15 minutes on a laptop. Why is it so slow? Is it an internet problem or a machine problem?

**Answer: It's pure computation.** The model is already on your hard drive — no internet is needed. The bottleneck is running 1,656 texts through a 110-million-parameter neural network on your CPU.

This document explains exactly what's happening inside the machine during that process.

---

## What Is PubMedBERT?

PubMedBERT is a **transformer model** — the same family of neural networks behind ChatGPT, BERT, and all modern language AI. Specifically, our model (`NeuML/pubmedbert-base-embeddings-matryoshka`) is:

| Property | Value |
|----------|-------|
| Architecture | BERT-base (12 layers, 12 attention heads) |
| Parameters | ~110 million |
| Hidden dimension | 768 |
| Max input length | 512 tokens (~350 words) |
| File size on disk | ~440 MB |
| Trained on | PubMed biomedical literature |

It's called "PubMedBERT" because it was trained specifically on medical/biomedical text. This is why it understands that "low back pain" and "lumbar spine disorder" are related concepts — it learned this from millions of medical papers.

---

## What Does "Encoding a Text" Actually Mean?

When we call `embedder.encode("NICE guidelines for low back pain management")`, here's what happens step by step:

### Step 1: Tokenisation (~0.1ms)

The text is split into **tokens** — subword pieces the model understands:

```
"NICE guidelines for low back pain management"
→ ["NICE", "guide", "##lines", "for", "low", "back", "pain", "management"]
→ [7592, 5765, 3210, 1012, 2659, 2067, 3555, 2968]   ← integer IDs
```

Each token maps to a row in a vocabulary table (30,522 words). This step is fast — it's just dictionary lookups, no neural network involved.

**What are those `##` prefixes?** BERT uses "WordPiece" tokenisation. If a word isn't in the vocabulary as a whole, it gets split. "guidelines" → "guide" + "##lines". The `##` means "continuation of previous word".

### Step 2: The Forward Pass (~50–200ms per text on CPU)

This is where the time goes. The token IDs enter the transformer and flow through **12 layers**, each performing:

```
Input tokens (8 tokens x 768 dimensions = 6,144 numbers)
    │
    ├── Layer 1: Self-Attention → Feed-Forward → Output
    ├── Layer 2: Self-Attention → Feed-Forward → Output
    ├── Layer 3: Self-Attention → Feed-Forward → Output
    │   ... (12 layers total)
    └── Layer 12: Self-Attention → Feed-Forward → Output
    │
Output: 8 tokens x 768 dimensions (each token now "understands" context)
```

#### What happens inside each layer?

**Self-Attention (the expensive part):**

Each token "looks at" every other token to understand context. For 8 tokens, that's 8 x 8 = 64 attention calculations. But each calculation involves 768-dimensional vectors, and there are 12 "attention heads" running in parallel.

The actual math:
```
Q = tokens x W_query    (8x768) x (768x768) = 8x768   ← matrix multiply
K = tokens x W_key      (8x768) x (768x768) = 8x768   ← matrix multiply
V = tokens x W_value    (8x768) x (768x768) = 8x768   ← matrix multiply
Attention = softmax(Q x K^T / sqrt(64)) x V             ← matrix multiply + softmax
```

That's **3 matrix multiplications** of size (8x768) x (768x768), plus the attention score computation. And this happens **12 times** (once per layer).

**Feed-Forward Network:**

After attention, each token passes through a 2-layer neural network:
```
hidden = ReLU(token x W1 + b1)    (768 → 3072)   ← matrix multiply
output = hidden x W2 + b2          (3072 → 768)   ← matrix multiply
```

That's 2 more matrix multiplications per layer, with a larger intermediate size (3072).

#### Total computation per text:

Per layer: ~5 large matrix multiplications
Total: 12 layers x 5 = **60 large matrix multiplications**

Each multiplication involves tens of thousands to millions of floating-point operations. This is why it takes 50–200ms per text on CPU.

### Step 3: Mean Pooling (~0.01ms)

After the 12 layers, we have one 768-dimensional vector per token. We need a single vector for the whole text:

```
Token 1 vector: [0.12, -0.45, 0.78, ..., 0.33]   (768 numbers)
Token 2 vector: [0.08, -0.31, 0.65, ..., 0.29]
Token 3 vector: [0.15, -0.52, 0.71, ..., 0.41]
...
Token 8 vector: [0.11, -0.38, 0.69, ..., 0.35]

Mean pooling → average each column:
Result:         [0.115, -0.415, 0.71, ..., 0.345]  (768 numbers)
```

This averaging produces a single vector that represents the "meaning" of the entire text. It's trivially fast — just averaging numbers.

### Step 4: L2 Normalisation (~0.01ms)

We scale the vector so its length (magnitude) equals exactly 1.0:

```
vector = vector / ||vector||

where ||vector|| = sqrt(v[0]^2 + v[1]^2 + ... + v[767]^2)
```

**Why?** When all vectors have length 1, the L2 distance between them is directly related to cosine similarity:

```
L2_distance(a, b)^2 = 2 - 2 * cosine_similarity(a, b)
```

So our FAISS index (which uses L2 distance) effectively finds the most semantically similar guidelines by cosine similarity. This is a standard trick in information retrieval.

---

## Why CPU Is Slow and GPU Is Fast

The bottleneck is **matrix multiplication** — Step 2 above. Here's the fundamental difference:

### CPU: Sequential Processing

A modern CPU has 4–8 cores. Each core can do 1 multiply-add per clock cycle (simplified). For a matrix multiplication of (768 x 768):

```
Operations needed: 768 x 768 x 768 = ~453 million multiply-adds
CPU speed: ~4 GHz x 8 cores = ~32 billion operations/second
Time: 453M / 32B ≈ 14ms per matrix multiply
Total for 60 multiplies: ~840ms per text
```

That's ~1 second per guideline. For 1,656 guidelines: **~25 minutes** (theoretical worst case; batching helps reduce this).

### GPU: Massive Parallelism

A GPU has **thousands** of cores (e.g., NVIDIA T4 has 2,560 CUDA cores). Each core is simpler than a CPU core, but matrix multiplication is "embarrassingly parallel" — every element can be computed independently:

```
GPU speed: ~2,560 cores x 1.5 GHz = ~3.8 trillion operations/second
Time per matrix multiply: 453M / 3.8T ≈ 0.12ms
Total for 60 multiplies: ~7ms per text
```

That's **~100x faster**. For 1,656 guidelines: **~12 seconds** on GPU vs ~15 minutes on CPU.

This is why:
- Cyprian built his index on Google Colab (free GPU) in under a minute
- Our laptop takes 5–15 minutes (CPU only)
- Machine learning companies spend millions on GPU clusters

### Why We Don't Need a GPU for This Project

Index building is a **one-time cost**. You run the script once, it saves `guidelines.index`, and you never run it again (unless guidelines change).

At runtime, the Retriever Agent only encodes **search queries** (short texts, batch-encoded per diagnosis — typically 3 at a time), not all 1,656 guidelines. Encoding a batch of 3 short queries takes ~100ms on CPU — fast enough for an API response. Batch encoding is more efficient than individual calls because the model weights are loaded into the CPU cache once per batch.

---

## What the Numbers Mean: 110 Million Parameters

"Parameters" are the learned weights — the numbers the model uses to transform inputs into outputs. Here's where they live:

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Token embeddings (vocabulary) | 30,522 x 768 = 23.4M | 21% |
| 12 Attention layers (Q, K, V, output) | 12 x 4 x (768 x 768) = 28.3M | 26% |
| 12 Feed-forward layers | 12 x 2 x (768 x 3072) = 56.6M | 52% |
| Layer norms, biases, etc. | ~1.7M | 1% |
| **Total** | **~110M** | **100%** |

Each parameter is a 32-bit float (4 bytes), so:
```
110M parameters x 4 bytes = 440 MB on disk
```

That's why the model file is ~440 MB. When loaded into RAM, it takes the same ~440 MB, plus temporary memory for activations during the forward pass.

---

## Batching: Why We Process 16 Texts at a Time

Instead of encoding one text at a time, we batch 16 together. Why?

**Without batching (1 text at a time):**
```
Load W_query matrix from RAM → CPU cache
Multiply: (1 x 768) x (768 x 768) = (1 x 768)
Repeat for next text: reload W_query from RAM again
```

**With batching (16 texts at a time):**
```
Load W_query matrix from RAM → CPU cache (once)
Multiply: (16 x 768) x (768 x 768) = (16 x 768)
All 16 texts processed with one matrix load
```

The weight matrix `W_query` (768 x 768 = 2.4 MB) only needs to be loaded from RAM into the CPU cache **once** for the whole batch instead of 16 times. Since RAM-to-cache transfer is the slowest part, batching can be 3–5x faster than processing one at a time.

We use batch_size=32 in our script. Larger batches are faster but use more memory (each text's activations take ~50 KB, so 32 texts need ~1.6 MB of activation memory — negligible).

---

## The Full Pipeline for Index Building

Here's the complete picture of what `scripts/build_index.py` does:

```
guidelines.csv (1,656 rows, "clean_text" column)
        │
        ▼
   Load PubMedBERT model into RAM (~440 MB)
        │
        ▼
   For each batch of 32 guidelines:
        │
        ├── Tokenise all 32 texts (fast, ~1ms)
        ├── Forward pass through 12 transformer layers (slow, ~3-5 seconds)
        ├── Mean pool to get 32 x 768 matrix (fast, ~0.01ms)
        └── L2 normalise each row (fast, ~0.01ms)
        │
        ▼
   Stack all batches → 1656 x 768 matrix (float32, ~5 MB)
        │
        ▼
   Create FAISS IndexFlatL2(768)
   Add all 1656 vectors to the index
        │
        ▼
   Save to data/guidelines.index (~5 MB file)
```

**Time breakdown:**
- Loading model: ~10 seconds (reading 440 MB from disk)
- Encoding 1,656 texts: ~5–15 minutes (the computation bottleneck)
- Building FAISS index: <1 second (just copying vectors into the index structure)
- Saving to disk: <1 second (writing ~5 MB)

**99% of the time is spent in Step 2 of the encoding — the transformer forward pass.**

---

## Why FAISS Search Is Fast But Encoding Is Slow

An important distinction:

| Operation | Speed | Why |
|-----------|-------|-----|
| **Encoding** (text → vector) | ~100ms/text on CPU | Full neural network forward pass (60 matrix multiplications) |
| **FAISS search** (vector → nearest neighbours) | ~0.1ms/query | Simple L2 distance calculation (1656 x 768 subtractions + additions) |

FAISS search is ~1000x faster because it's just arithmetic on pre-computed vectors — no neural network involved. This is the whole point of building an index: you pay the encoding cost **once** (during index building), then searches are nearly free.

---

## Key Takeaways

1. **Index building is slow because it's pure computation** — running 1,656 texts through a 110M-parameter neural network on CPU.

2. **It's a one-time cost** — run the script once, get the index file, never run it again unless guidelines change.

3. **GPU would be ~100x faster** but isn't worth setting up for a one-time operation.

4. **At runtime, only queries are encoded** (batch-encoded per diagnosis, typically 3 at a time), which takes ~100ms per batch — perfectly fine for API responses.

5. **The 440 MB model size comes directly from the 110M parameters** — each stored as a 4-byte float.

6. **Batching helps** because it reuses the weight matrices loaded into CPU cache across multiple texts.
