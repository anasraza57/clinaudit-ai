# Consultation Insight Agent Explained

## What Does the Consultation Insight Agent Do?

The Consultation Insight Agent is **Stage 1** of the 4-agent pipeline. Its job is to take a patient's raw clinical records and organise them into something the rest of the pipeline can work with.

Think of it as a filing clerk who receives a pile of medical notes and sorts them into labelled folders: "these are diagnoses", "these are treatments", "these are referrals", etc.

## The Problem It Solves

A patient's clinical entries come as flat rows with SNOMED CT codes:

```
ConceptID    Term                    ConceptDisplay
279039007    Low back pain           Low back pain
183545006    Orthopaedic referral    Referral to orthopaedic service
308752009    B12 injection           Intramuscular injection of vitamin B12
```

To audit whether the GP followed guidelines, we need to know **which of these are diagnoses** (the conditions being treated) and **what actions were taken** (treatments, referrals, investigations). The raw data doesn't tell us this directly.

## How It Works

### Step 1: Categorisation (SNOMED Categoriser)

Each clinical concept is classified into one of these categories:

| Category | What it means | Examples |
|----------|--------------|----------|
| **diagnosis** | A condition, symptom, or injury | Low back pain, Osteoarthritis of knee |
| **treatment** | A medication, injection, or therapy | Steroid injection, Ibuprofen |
| **referral** | A referral to another service | Referral to physiotherapist |
| **investigation** | A test or scan | X-ray of knee, Blood test |
| **procedure** | A surgical or clinical procedure | Knee replacement, Arthroscopy |
| **administrative** | Admin actions, reviews, certificates | Med3 certificate, Medication review |

> **Note:** There are exactly 6 categories — no "other" bucket. Every concept must map to one of these six. If the LLM or rules can't find a clear match, the fallback is "administrative" rather than an ambiguous "other" category. This was a deliberate design decision to ensure every concept gets a meaningful classification.

#### Two-Tier Classification

1. **Rule-based (84% of concepts)**: Keyword pattern matching on the concept display text. Fast, free, and deterministic. For example, if the text contains "pain" or "fracture", it's a diagnosis. If it contains "referral", it's a referral.

2. **LLM fallback (16% of concepts)**: For concepts that don't match any rule, we ask the LLM to classify them. The LLM prompt explicitly lists only the 6 valid categories and forbids returning anything else. This handles edge cases like "Bandy legged" or "Acquired hallux valgus".

Since there are only ~1,261 unique concepts in the entire dataset, we classify each one **once** and cache the result. Categories are also **persisted to the database** (the `category` column in `clinical_entries`), so after the first run, subsequent runs load categories from DB with zero LLM calls.

#### Batched LLM Categorisation

The original implementation made **one LLM call per unmatched concept** — 322 individual API calls. This caused the server to crash from memory exhaustion (each call accumulates HTTP response objects, connection state, and log data).

We fixed this by **batching 50 concepts per prompt** using JSON response format:

```python
# Old: 322 individual calls
for display in unmatched:
    response = await ai_provider.chat_simple(prompt_for_one_concept)

# New: 7 batched calls
for batch in chunks(unmatched, 50):
    response = await ai_provider.chat_simple(prompt_for_50_concepts)
    # Returns: {"Knee pain": "diagnosis", "Ibuprofen": "treatment", ...}
```

This reduces 322 API calls to just 7, with safety features:
- **JSON response format** — each concept is explicitly mapped to its category (no line-alignment risk)
- **Case-insensitive matching** — parsed results matched back to original concept names robustly
- **80% match threshold** — if a batch matches fewer than 80% of concepts, misses are retried individually
- **Fallback to single calls** — if a batch fails to parse entirely, each concept is retried one at a time
- **gc.collect()** between batches — forces garbage collection to prevent memory buildup

#### Category Persistence

The `clinical_entries` table has a `category` column that starts as `NULL`. On the first pipeline run:

1. Query DB for concepts where `category IS NOT NULL` — these are already classified, skip them
2. Query DB for concepts where `category IS NULL` — these need classification
3. Run rules + batched LLM on the uncategorised concepts
4. **Write categories back to DB** — UPDATE all matching rows with the new category
5. Load everything into the in-memory cache for the Consultation Insight Agent to use

On subsequent runs (or after a server crash), step 1 returns everything and steps 2-4 are skipped entirely. This means:
- **First run:** ~7 LLM batch calls (one-time cost)
- **Every run after:** 0 LLM calls, instant load from DB
- **After a crash:** no work is lost — categories are persisted

### Step 2: Grouping by Episode

A patient may have visited the GP for different MSK problems at different times. Each visit date (called the "index date") represents a separate **episode**. The Consultation Insight Agent groups entries by index date.

### Step 3: Structured Output

The Consultation Insight Agent produces an `ExtractionResult` containing:

```
Patient: pat-001
├── Episode 1 (index_date: 2024-01-15)
│   ├── Diagnoses: [Low back pain]
│   ├── Treatments: [Ibuprofen]
│   ├── Referrals: [Referral to physiotherapist]
│   ├── Investigations: [X-ray of lumbar spine]
│   └── Administrative: [Telephone consultation]
├── Episode 2 (index_date: 2024-06-01)
│   ├── Diagnoses: [Osteoarthritis of knee]
│   ├── Treatments: [Steroid injection into knee]
│   └── Referrals: [Referral to orthopaedics]
```

## What Happens Next?

The diagnoses from the Consultation Insight Agent's output are passed to the **Audit Query Generator** (Stage 2), which generates search queries to find relevant NICE guidelines for each diagnosis. The treatments and referrals are later compared against those guidelines by the **Compliance Auditor Agent** (Stage 4).

## Deep Dive: How the Rule-Based Patterns Work

### What Are Regex Patterns?

Regular expressions (regex) are text-matching rules. We use them to scan the concept display text and decide what category it belongs to. For example:

```python
_p(r"\bpain\b", "diagnosis")       # Matches "pain" as a whole word
_p(r"\breferral\b", "referral")    # Matches "referral" as a whole word
_p(r"\binjection\b", "treatment")  # Matches "injection" as a whole word
```

The `\b` means **word boundary** — so `\bpain\b` matches "Low back pain" but would NOT match "painting" or "painful". The `r` prefix means it's a raw string (Python won't interpret backslashes specially).

### Why Patterns Are Checked in Order

Patterns are evaluated **top to bottom — first match wins**. This ordering matters because some concepts match multiple categories:

- "Review of medication" contains both "review" (administrative) AND "medication" (treatment)
- "Referral for X-ray" contains both "referral" AND "X-ray" (investigation)

We want "Review of medication" to be **administrative** (it's a review, not prescribing medication), so administrative rules come before treatment rules. Similarly, "Referral for X-ray" is a **referral** action, not an investigation itself.

This was an actual bug we caught and fixed during development.

### Three Rounds of Pattern Building

We didn't write all the patterns at once. Instead, we built them iteratively by testing against the real database:

**Round 1 — Core keywords:**
```
pain, fracture, referral, injection, x-ray, consultation, certificate...
```
Result: 70% coverage (885 / 1,261 concepts matched)

**Round 2 — Medical suffixes:**
```
-itis (inflammation: tendinitis, fasciitis, epicondylitis)
-pathy (disease: neuropathy, arthropathy)
-osis (condition: stenosis, scoliosis, osteoporosis)
```
Plus specific terms: abrasion, abscess, cyst, wound, burn, etc.
Result: 81% coverage (1,022 / 1,261)

**Round 3 — Surgical suffixes and remaining gaps:**
```
-ectomy (surgical removal: bursectomy, appendectomy)
-otomy (surgical incision: osteotomy)
-plasty (surgical repair: arthroplasty)
```
Plus: vaccine, cast, sling, care plan, follow-up, blood pressure, etc.
Result: 84% coverage (1,069 / 1,261)

### Why Not 100% Rules?

The remaining 192 concepts are genuine edge cases that don't follow obvious keyword patterns — things like "Bandy legged", "Acquired hallux valgus", or "Application of adhesive skin closure". For these, we fall back to the LLM which can understand medical meaning beyond simple keywords.

Since we only have 1,261 unique concepts and classify each one **once** (then persist to DB), the LLM cost is negligible — roughly 7 batched API calls total, ever. After the first run, categories load from the database with zero LLM calls.

### Pattern Categories and Examples

| Category | Pattern Examples | What They Match |
|----------|-----------------|-----------------|
| **diagnosis** | `\bpain\b`, `\bfracture\b`, `\b\w+itis\b` | "Low back pain", "Finger fracture", "Tendinitis" |
| **referral** | `\breferral\b`, `\brefer(?:red)?\s+to\b` | "Referral to orthopaedics", "Referred to physio" |
| **treatment** | `\binjection\b`, `\bprescription\b`, `\bvaccin` | "Steroid injection", "Prescription of drug" |
| **investigation** | `\bx[- ]?ray\b`, `\bmri\b`, `\bblood\s+test\b` | "X-ray of knee", "MRI scan", "Blood test" |
| **procedure** | `\b\w+ectomy\b`, `\b\w+plasty\b`, `\barthroscop` | "Bursectomy", "Arthroplasty", "Arthroscopy" |
| **administrative** | `\breview\b`, `\bconsultation\b`, `\bcertificate\b` | "Medication review", "Telephone consultation" |

## Why We Changed the Approach from Hiruni's

### What Hiruni Did (FHIR Server)

Hiruni's extractor sent each SNOMED concept ID to a **FHIR server** called HADES running locally on her machine. The server returned the **Fully Specified Name (FSN)** which contains an official semantic tag in parentheses:

```
Input:  SNOMED ID 279039007
FHIR returns: "Low back pain (disorder)"
                                ^^^^^^^^^^
                         Hiruni parsed this tag → category = "diagnosis"
```

Other examples: "(procedure)", "(finding)", "(substance)", "(regime/therapy)".

This is the **officially correct** way to categorise SNOMED concepts — the semantic tags come from SNOMED's own terminology hierarchy.

### Why We Couldn't Use It

1. **HADES isn't included** — it's a separate Java server that needs to be installed, configured, and loaded with the full SNOMED CT database. Nobody documented how to set it up, and it wasn't provided in the project files.
2. **External dependency** — the entire pipeline breaks if the FHIR server is down, unreachable, or misconfigured. Hiruni had zero error handling for this.
3. **Slow** — each of the 21,530 entries would need an individual HTTP request to the server.
4. **Not portable** — anyone cloning our repo would need to set up this server before anything works.

### Honest Comparison

| | Hiruni's (FHIR Server) | Ours (Rules + LLM) |
|---|---|---|
| **Accuracy** | Very high — SNOMED's official semantic tags | High — 84% rules (reliable), 16% LLM (mostly correct) |
| **Dependencies** | Requires HADES FHIR server running | Zero external services needed |
| **Speed** | Slow — HTTP request per concept | Instant for rules, seconds for LLM batch |
| **Cost** | Free (local server) but high setup cost | Free for rules, minimal LLM cost (~192 calls total) |
| **Reliability** | Fragile — server crash = pipeline crash | Robust — rules never fail, LLM has error handling |
| **Setup** | Complex — install Java, HADES, load SNOMED data | Zero — patterns are in the code |
| **Portability** | Poor — works on Hiruni's machine only | Perfect — works anywhere with `pip install` |

### Is Our Approach Better?

**For our situation, yes.** Here's why:

The FHIR approach uses SNOMED's official tags, which sounds ideal. But our keyword rules are effectively doing the same classification — medical terms ending in "-itis" ARE inflammatory disorders, terms containing "referral" ARE referrals, terms containing "pain" ARE diagnoses. We're matching the same linguistic signals that SNOMED's own hierarchy encodes.

The 16% edge cases where keywords aren't enough? We send those to the LLM, which can reason about medical meaning — arguably better than a simple semantic tag for borderline cases like "Bandy legged" or "Application of adhesive skin closure".

**Trade-off:** We sacrifice the theoretical correctness of SNOMED's official hierarchy for practical advantages: zero setup, instant speed, no external dependencies, and works on any machine. Given that this is a research project (not a clinical production system), this trade-off makes sense.
