# Query Agent Explained

## What Does the Query Agent Do?

The Query Agent is **Stage 2** of the 4-agent pipeline. Its job is to take the diagnoses identified by the Extractor and generate targeted search queries that will be used to find relevant NICE guidelines.

Think of it as a librarian who reads a doctor's notes and writes search cards: "I need to find the guidelines about low back pain management", "I need the referral criteria for knee osteoarthritis", etc.

## Why Do We Need It?

The Extractor gives us diagnoses like "Low back pain" or "Osteoarthritis of knee". But we can't just throw those raw terms into our guideline search — we need **optimised search queries** that will find the right guideline passages.

Here's why:

### The FAISS Search Problem

Our guideline database uses **semantic similarity search**. When we search for guidelines, we:
1. Take a query text
2. Convert it to a 768-dimensional vector using PubMedBERT
3. Find the guideline vectors closest to our query vector (cosine similarity)

The problem: the search quality depends entirely on what we search for. Consider:

| Search Query | What FAISS Finds |
|---|---|
| "Low back pain" | Matches anything mentioning back pain — too broad |
| "NICE guidelines for assessment and management of low back pain and sciatica" | Matches the actual NICE guideline title — precise |
| "low back pain referral criteria and imaging recommendations" | Matches specific guideline sections about when to refer or image |

The more our query text matches how NICE guidelines are actually written, the better the search results. That's what the Query Agent optimises for.

### Real Example

For a patient diagnosed with "Low back pain", the Query Agent generates:

```
Query 1: "NICE guidelines for assessment and management of low back pain and sciatica"
Query 2: "non-specific low back pain pharmacological and non-pharmacological treatment"
Query 3: "low back pain referral criteria and imaging recommendations"
```

Each query targets a different aspect of care:
- **Query 1** finds the main guideline document
- **Query 2** finds treatment-specific sections
- **Query 3** finds referral and imaging criteria

Together, they give the Retriever Agent a comprehensive set of guideline passages to work with.

## How It Works

### The Three-Tier Approach

Just like the Extractor uses rules first and LLM second, the Query Agent has three tiers:

```
Diagnosis → [1. Template Match?] → YES → Use pre-written queries
                    ↓ NO
             [2. LLM Available?] → YES → Generate queries via LLM
                    ↓ NO
             [3. Default Queries] → Generic template queries
```

### Tier 1: Template Queries (Most Common Diagnoses)

For the most common MSK conditions, we have **hand-crafted query templates** — pre-written search queries that we know work well with our FAISS index.

```python
"low back pain": [
    "NICE guidelines for assessment and management of low back pain and sciatica",
    "non-specific low back pain pharmacological and non-pharmacological treatment",
    "low back pain referral criteria and imaging recommendations",
]
```

We have templates for ~15 common MSK conditions: low back pain, osteoarthritis, carpal tunnel syndrome, plantar fasciitis, gout, fibromyalgia, osteoporosis, fractures, shoulder pain, hip pain, sciatica, and more.

**Why templates?** Because for common conditions, we can write better queries than an LLM would generate — we know exactly how NICE guidelines are titled and structured. Templates are also free, instant, and deterministic.

**Matching logic:** The agent first tries an exact match (case-insensitive), then a substring match. So "Osteoarthritis of knee" matches the "osteoarthritis" template even though it's not an exact match.

### Tier 2: LLM Generation (Unusual Diagnoses)

For diagnoses that don't match any template (e.g., "Acquired hallux valgus" or "Dupuytren's contracture"), the agent asks the LLM to generate queries.

The prompt instructs the LLM to:
- Generate exactly 3 search queries (configurable via `max_queries_per_diagnosis`)
- Cover different aspects of care (assessment, treatment, referral)
- Use clinical terminology matching how NICE guidelines are written
- Include "NICE" or "guidelines" in at least one query

The prompt also includes **episode context** — what treatments and referrals the patient received — so the LLM can generate more targeted queries.

### Tier 3: Default Queries (Fallback)

If there's no template AND no LLM available (or the LLM fails), the agent generates generic queries using a simple template:

```
1. "NICE clinical guidelines for {diagnosis} management"
2. "{diagnosis} treatment options and recommendations"
3. "{diagnosis} referral criteria and investigations"
```

These aren't as optimised as the other tiers, but they're reasonable and always available.

## Data Flow

```
ExtractionResult (from Extractor)
│
├── Episode 1 (2024-01-15)
│   ├── Diagnosis: "Low back pain"        → Template queries (3)
│   └── Diagnosis: "Sciatica"             → Template queries (3)
│
├── Episode 2 (2024-06-01)
│   └── Diagnosis: "Hallux valgus"        → LLM queries (3)
│
└── QueryResult
    ├── pat_id: "pat-001"
    ├── total_diagnoses: 3
    ├── total_queries: 9
    └── diagnosis_queries: [
          {diagnosis: "Low back pain", queries: [...], source: "template"},
          {diagnosis: "Sciatica", queries: [...], source: "template"},
          {diagnosis: "Hallux valgus", queries: [...], source: "llm"},
        ]
```

## What Happens Next?

The `QueryResult` is passed to the **Retriever Agent** (Stage 3), which:
1. Takes each query text
2. Encodes it using PubMedBERT into a 768-dimensional vector
3. Searches the FAISS index for the top-5 most similar guideline passages
4. Returns the matched guideline texts for the Scorer Agent to evaluate

## Hiruni's Previous Approach (In Detail)

### How Her Query Agent Worked

Hiruni's approach was simple: **send everything to a local LLM**. Every single diagnosis — no matter how common or straightforward — went through the same path.

Here's her actual code (simplified for readability):

```python
from llama_cpp import Llama

class QueryAgent:
    def __init__(self, model_path: str):
        # Load a 4GB Mistral-7B model into RAM
        self.llm = Llama(
            model_path=model_path,  # "C:\Users\hirun\agentic-msk\models\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
            n_ctx=4096,
            n_threads=8
        )

    def format_extractor_output(self, extracted_data):
        # Turn extracted concepts into a readable string
        return "\n".join(
            f"[{item['type']}] {item['term']} ({item['concept_id']}) on {item['date']}"
            for item in extracted_data
        )

    def generate(self, extracted_data):
        extractor_text = self.format_extractor_output(extracted_data)
        prompt = f"""
You are a medical query generation assistant.
Input: List of extracted medical concepts from a clinical note.
Output: Generate 1–3 concise search queries per concept to retrieve clinical guidelines.
Keep queries short and include relevant terms like "clinical guideline",
"management recommendations", or "treatment protocol".
Only output queries, one per line.

Input:
{extractor_text}
"""
        output = self.llm(prompt, max_tokens=200)
        text = output["choices"][0]["text"].strip()
        return [line.strip("- ").strip() for line in text.split("\n") if line.strip()]
```

And here's how it was wired into the pipeline using LangGraph:

```python
MODEL_PATH = r"C:\Users\hirun\agentic-msk\models\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
query_agent_instance = QueryAgent(model_path=MODEL_PATH)

def query_node(state):
    state["queries"] = query_agent_instance.generate(state["extracted_data"])
    return state
```

### What This Means in Practice

Let's say a patient has "Low back pain". Here's what Hiruni's code does:

1. Format it as: `[disorder] Low back pain (279039007) on 2024-01-15`
2. Send the entire prompt to Mistral-7B running locally
3. Wait for Mistral-7B to generate text (~2-5 seconds per diagnosis on CPU)
4. Parse the response by splitting on newlines
5. Hope the output is well-formatted

The LLM might return something like:
```
- low back pain clinical guideline
- management recommendations for back pain
- treatment protocol for lumbar pain
```

### The Problems

**1. Underpowered model for the task.** Mistral-7B is a general-purpose 7-billion-parameter model. It's decent at following instructions, but it doesn't know:
- How NICE guidelines are titled (e.g., "NG59: Low back pain and sciatica in over 16s")
- What clinical language PubMedBERT responds best to
- That our guideline database is specifically NICE UK guidelines, not general medical literature

So it produces **generic medical queries** instead of **targeted guideline queries**. The difference matters — "low back pain clinical guideline" will match many irrelevant things in our FAISS index, while "NICE guidelines for assessment and management of low back pain and sciatica" will match the actual NICE guideline document.

**2. Every diagnosis goes through the LLM.** Even "Low back pain" — arguably the most common MSK diagnosis in our dataset — requires a full LLM inference. This is wasteful because:
- We know exactly what NICE guidelines exist for low back pain
- We can write better search queries ourselves than Mistral-7B would generate
- We're paying the inference cost (time + compute) for zero benefit

**3. Local model dependency.** The code requires:
- A 4GB model file downloaded to a specific Windows path
- Enough RAM to load the model (~8GB minimum)
- The `llama_cpp` library compiled with appropriate CPU/GPU support
- Nobody else can run this without replicating Hiruni's exact setup

**4. Hardcoded path.** `C:\Users\hirun\agentic-msk\models\mistral-7b-instruct-v0.1.Q4_K_M.gguf` — this is a Windows-specific absolute path to Hiruni's personal machine. It's not even configurable.

**5. No error handling.** If the model fails to load, the format is wrong, or inference produces garbage — the entire pipeline crashes. There's no try/except, no fallback, no graceful degradation.

**6. No context awareness.** The prompt sends the diagnosis but doesn't include what treatments or referrals the patient received. This context could help generate more targeted queries (e.g., if the patient was referred to orthopaedics, we should search for referral criteria guidelines).

## Why We Need Three Tiers

### The Core Insight

Not all diagnoses are equally difficult to search for. Consider these three cases:

| Diagnosis | How Often It Appears | How Easy to Query |
|---|---|---|
| "Low back pain" | Very common (~15% of patients) | Trivial — we know the exact NICE guideline |
| "Acquired hallux valgus" | Rare | Moderate — an LLM can reason about it |
| "Bandy legged" | Very rare | Hard — even an LLM might struggle |

It makes no sense to use the same approach for all three. That's like using a GPS to navigate to your own kitchen — the right tool depends on the difficulty of the problem.

### Tier 1: Templates — Why Hand-Crafted Queries Are Best for Common Conditions

For "Low back pain", we don't need an LLM to tell us what to search for. We **know** that:

- NICE has a guideline called "Low back pain and sciatica in over 16s: assessment and management" (NG59)
- The guideline covers assessment, pharmacological treatment, non-pharmacological treatment, and referral criteria
- PubMedBERT responds best to clinical language that matches guideline titles

So we write queries by hand:

```python
"low back pain": [
    "NICE guidelines for assessment and management of low back pain and sciatica",
    "non-specific low back pain pharmacological and non-pharmacological treatment",
    "low back pain referral criteria and imaging recommendations",
]
```

These queries are **better than anything an LLM would generate** because:
- They use the exact language of NICE guidelines
- They're tuned for PubMedBERT semantic similarity
- Each covers a different aspect of care (overview, treatment, referral)
- They're deterministic — same diagnosis always produces same queries
- They cost nothing and take zero time

We have templates for ~15 common MSK conditions. Together, these cover the majority of diagnoses in our dataset because MSK conditions follow a power law distribution — a few common conditions (back pain, osteoarthritis, shoulder pain) account for most cases.

**The analogy:** If you're a librarian and someone asks for "the main Shakespeare plays", you don't need to look them up — you already know where Hamlet, Macbeth, and Romeo and Juliet are shelved. You only need the catalogue system for obscure requests.

### Tier 2: LLM — Why We Still Need AI for Rare Diagnoses

Templates can't cover everything. Our dataset has hundreds of unique diagnoses, and some are unusual:

- "Acquired hallux valgus" (bunion deformity)
- "Dupuytren's contracture" (finger condition)
- "Chronic regional pain syndrome"
- "Meralgia paraesthetica" (thigh nerve compression)

Writing templates for every possible diagnosis would be impractical and fragile. Instead, we ask the LLM to reason about these cases. The LLM can:

- Understand that "hallux valgus" relates to foot deformity guidelines
- Know that "Dupuytren's" relates to hand surgery referral criteria
- Generate queries using appropriate clinical terminology

Our LLM prompt is also smarter than Hiruni's — it includes **episode context**:

```
Diagnosis: Acquired hallux valgus
Context: Referrals: Orthopaedic referral; Treatments: Prescription of drug
```

This helps the LLM generate queries about both conservative management AND surgical referral criteria, because it knows the patient was actually referred.

We also use a much stronger model than Hiruni did — GPT-4o-mini (via our provider abstraction) instead of Mistral-7B. And since templates handle the common cases, the LLM is only called for rare diagnoses — keeping costs minimal.

### Tier 3: Defaults — Why We Need a Safety Net

What if:
- The diagnosis doesn't match any template AND
- The LLM is unavailable (no API key, rate limit, network error)

Without Tier 3, the pipeline would crash — exactly what happened with Hiruni's code. Our default queries ensure the pipeline **always produces output**, even in degraded conditions:

```
1. "NICE clinical guidelines for {diagnosis} management"
2. "{diagnosis} treatment options and recommendations"
3. "{diagnosis} referral criteria and investigations"
```

These aren't as optimised as templates or LLM queries, but they're functional. The Retriever will still find *something* relevant, and the Scorer can still evaluate *something* — even if the retrieval quality is slightly lower.

**The engineering principle:** A system that produces a slightly imperfect result is infinitely more useful than a system that crashes.

### How the Three Tiers Work Together

```
Patient diagnosed with "Low back pain"
  → Tier 1 matches "low back pain" template
  → Returns 3 expert-crafted queries
  → LLM never called (saves time and money)

Patient diagnosed with "Dupuytren's contracture"
  → Tier 1: no template match
  → Tier 2: LLM generates 3 queries with episode context
  → Returns LLM queries

Patient diagnosed with "Unusual rare condition" (and LLM is down)
  → Tier 1: no template match
  → Tier 2: LLM call fails (API error)
  → Tier 3: returns 3 generic default queries
  → Pipeline continues (degraded but functional)
```

### Honest Comparison

| | Hiruni's (Pure LLM) | Ours (3-Tier) |
|---|---|---|
| **Query quality for common diagnoses** | Mediocre — Mistral-7B doesn't know NICE guidelines | Excellent — hand-crafted by us to match guideline language |
| **Query quality for rare diagnoses** | Mediocre — Mistral-7B is weak at medical reasoning | Good — GPT-4o-mini with episode context |
| **Speed** | Slow — every diagnosis needs LLM inference (~2-5s each) | Mostly instant — templates for common cases, LLM only for rare |
| **Cost** | Free (local model) but high compute cost | Minimal — LLM only for rare diagnoses |
| **Reliability** | Fragile — no fallback, model load failure = crash | Robust — three fallback layers, pipeline never crashes |
| **Dependencies** | 4GB model file + llama_cpp + enough RAM | Zero (templates in code; LLM and defaults are optional layers) |
| **Portability** | Hardcoded to Hiruni's Windows machine | Works anywhere with `pip install` |
| **Determinism** | Non-deterministic — LLM may give different queries each time | Deterministic for common cases (templates), LLM only for rare |
| **Tunability** | Can only change the prompt | Can tune individual templates per condition + change LLM prompt |

### Is Our Approach Better?

**Yes, and here's the key insight:** Hiruni treated query generation as a uniform problem — every diagnosis gets the same treatment (send to LLM). But the real problem has different difficulty levels. Our three-tier approach recognises this and uses the right tool for each level:

- **Easy problems** (common MSK diagnoses) → simple, fast, perfect solution (templates)
- **Medium problems** (unusual diagnoses) → powerful tool (LLM with context)
- **Edge cases** (LLM unavailable) → safety net (defaults)

This is a general software engineering principle called **graceful degradation** — the system delivers the best possible result at each level and never completely fails.

## Configuration

The Query Agent reads one setting from the environment:

| Setting | Default | Description |
|---|---|---|
| `MAX_QUERIES_PER_DIAGNOSIS` | 3 | Maximum search queries generated per diagnosis |

This is defined in `src/config/settings.py` and can be overridden via the `.env` file.
