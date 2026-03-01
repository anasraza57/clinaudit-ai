# Project Overview — What Are We Building and Why

## The Problem

Imagine you're the head of quality at an NHS region. You oversee thousands of GP consultations for musculoskeletal (MSK) conditions — back pain, knee arthritis, shoulder injuries, fractures. For each condition, NICE has published detailed guidelines on what treatments to try first, when to refer to a specialist, what imaging to order, and what NOT to do (like prescribing opioids for chronic pain as a first step).

You want to know: **Are our GPs actually following these guidelines?**

Currently, your only option is a **manual clinical audit** — expert clinicians sit down, pull up patient records one by one, compare what the GP did against what the guidelines say, and write up their findings. In the CrossCover trial at Keele University, auditing just 120 patients out of 10,000+ required significant clinician time and expertise.

This doesn't scale. You can never audit every patient. You can never do it in real-time. And different auditors may assess the same case differently.

## The Solution

We're building an **AI system that automates this audit**. It takes structured patient records (already coded in SNOMED — the international medical terminology standard) and automatically:

1. **Understands** what happened in the consultation (diagnoses, treatments, referrals, procedures)
2. **Finds** the relevant NICE guidelines for each condition
3. **Evaluates** whether the documented care follows those guidelines
4. **Produces** a score and explanation for each patient

This turns a weeks-long manual process into something that can run across thousands of patients in hours.

## The Data

Our patient data comes from the **CrossCover clinical trial** — a real UK clinical trial with 10,000+ anonymised MSK patients from primary care. Each patient's record contains time-stamped clinical events coded in SNOMED CT (e.g., "279039007 = Low back pain", "308447003 = Referral to physiotherapist").

Our guideline database contains **1,656 NICE guideline documents** covering various conditions, treatments, and recommendations.

## The Architecture — Four Agents Working Together

The system is built as a pipeline of four specialised AI agents:

### Agent 1: Extractor
**Job:** Read the patient's clinical events and categorise each one.
**Input:** Raw rows from the patient database (SNOMED codes + terms)
**Output:** A structured list: "This is a diagnosis, this is a treatment, this is a referral..."
**Example:** Sees code 239872002 ("Osteoarthritis of hip") → categorises as DIAGNOSIS. Sees 428906001 ("Injection of steroid into hip joint") → categorises as TREATMENT.

### Agent 2: Query Generator
**Job:** Take the extracted clinical concepts and create search queries to find relevant guidelines.
**Input:** The structured list from the Extractor
**Output:** Search queries like "NICE guidelines for osteoarthritis of hip management" and "steroid injection for hip osteoarthritis evidence"
**Why needed:** You can't just search for "osteoarthritis" — you need targeted queries that will find the specific guideline sections about treatment, referral criteria, imaging, etc.

### Agent 3: Retriever
**Job:** Search the guideline database to find the most relevant guideline passages.
**Input:** Search queries from the Query Generator
**Output:** The top 5 most relevant guideline text passages for each query
**How:** Converts queries into numerical vectors using PubMedBERT (a medical-specialist AI), then uses FAISS to find the guideline vectors most similar in meaning.

### Agent 4: Scorer
**Job:** Compare what the GP actually did against what the guidelines recommend.
**Input:** The patient's diagnoses/treatments + the retrieved guideline passages
**Output:** A score per diagnosis (+1 if guidelines were followed, -1 if not) with an explanation
**How:** An LLM reads the patient data alongside the guideline text and makes a judgment.

## The End Result

For each patient, the system produces:
- **Per-diagnosis scores** — was each condition managed according to guidelines?
- **Explanations** — what was done well, what was missed
- **An overall adherence score** — a single number summarising guideline compliance
- **Aggregate analytics** — across the whole dataset, what are the common guideline violations?

## How We'll Know It Works

We validate against the **120 gold-standard cases** — patients that were manually audited by expert clinicians. If our AI's scores closely match the human auditors' assessments, the system is working correctly.
