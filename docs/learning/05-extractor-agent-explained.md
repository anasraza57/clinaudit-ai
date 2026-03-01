# Extractor Agent Explained

## What Does the Extractor Do?

The Extractor is **Stage 1** of the 4-agent pipeline. Its job is to take a patient's raw clinical records and organise them into something the rest of the pipeline can work with.

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
| **other** | Anything else | — |

#### Two-Tier Classification

1. **Rule-based (84% of concepts)**: Keyword pattern matching on the concept display text. Fast, free, and deterministic. For example, if the text contains "pain" or "fracture", it's a diagnosis. If it contains "referral", it's a referral.

2. **LLM fallback (16% of concepts)**: For concepts that don't match any rule, we ask the LLM to classify them. This handles edge cases like "Bandy legged" or "Acquired hallux valgus".

Since there are only ~1,261 unique concepts in the entire dataset, we classify each one **once** and cache the result. This means even the LLM calls are minimal.

### Step 2: Grouping by Episode

A patient may have visited the GP for different MSK problems at different times. Each visit date (called the "index date") represents a separate **episode**. The Extractor groups entries by index date.

### Step 3: Structured Output

The Extractor produces an `ExtractionResult` containing:

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

The diagnoses from the Extractor's output are passed to the **Query Agent** (Stage 2), which generates search queries to find relevant NICE guidelines for each diagnosis. The treatments and referrals are later compared against those guidelines by the **Scorer Agent** (Stage 4).

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

Since we only have 1,261 unique concepts and classify each one **once** (then cache it), the LLM cost is negligible — roughly 192 API calls total, ever.

### Pattern Categories and Examples

| Category | Pattern Examples | What They Match |
|----------|-----------------|-----------------|
| **diagnosis** | `\bpain\b`, `\bfracture\b`, `\b\w+itis\b` | "Low back pain", "Finger fracture", "Tendinitis" |
| **referral** | `\breferral\b`, `\brefer(?:red)?\s+to\b` | "Referral to orthopaedics", "Referred to physio" |
| **treatment** | `\binjection\b`, `\bprescription\b`, `\bvaccin` | "Steroid injection", "Prescription of drug" |
| **investigation** | `\bx[- ]?ray\b`, `\bmri\b`, `\bblood\s+test\b` | "X-ray of knee", "MRI scan", "Blood test" |
| **procedure** | `\b\w+ectomy\b`, `\b\w+plasty\b`, `\barthroscop` | "Bursectomy", "Arthroplasty", "Arthroscopy" |
| **administrative** | `\breview\b`, `\bconsultation\b`, `\bcertificate\b` | "Medication review", "Telephone consultation" |

## How This Differs from Hiruni's Approach

Hiruni's extractor required a running FHIR server (HADES) to look up SNOMED codes and parse their Fully Specified Names for semantic tags like "(disorder)" and "(procedure)". Our approach:
- Uses keyword pattern matching instead of an external FHIR server
- Falls back to the LLM for edge cases instead of requiring specific SNOMED infrastructure
- Achieves 84% coverage with rules alone, making it fast and cheap
- Doesn't require any external service to be running
