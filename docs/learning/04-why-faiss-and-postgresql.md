# Why Do We Use Both FAISS and PostgreSQL?

## The Question

If we already store patient records and guidelines in PostgreSQL, why do we also need FAISS? Where exactly does FAISS fit in, and why are guidelines stored in both places?

## The Short Answer

PostgreSQL and FAISS solve **different problems**. PostgreSQL handles structured data — storing, querying, and linking records. FAISS handles **semantic search** — finding guidelines by meaning, not by exact keywords.

## PostgreSQL vs FAISS — What Each Does

### PostgreSQL (Relational Database)
- Stores structured data in tables with rows and columns
- Excels at **exact lookups**: "Get patient X", "Find all entries with SNOMED code 239873007"
- Maintains **relationships**: patients → clinical entries → audit results → guidelines
- Supports filtering, sorting, aggregation: "Show all patients with adherence score below 0.5"

### FAISS (Vector Similarity Search)
- Stores **embeddings** — numerical representations of text (arrays of 768 numbers)
- Excels at **meaning-based search**: "Find guidelines most relevant to knee osteoarthritis management"
- Returns results ranked by **semantic similarity**, not keyword matching
- Cannot store relationships, filter by date, or do anything a database does

## Why SQL Can't Replace FAISS

Imagine the Query Agent generates: *"NICE guidelines for knee osteoarthritis treatment"*

With SQL, you'd have to do something like:
```sql
SELECT * FROM guidelines WHERE title LIKE '%knee%' OR clean_text LIKE '%osteoarthritis%';
```

This fails because:
- A guideline titled "Osteoarthritis: care and management" doesn't contain the word "knee" but is highly relevant
- A guideline mentioning "knee replacement surgery" would match but might not be about conservative treatment
- SQL matches **words**, not **meaning**

With FAISS, the query is converted to a vector (using PubMedBERT), and FAISS finds the guidelines whose vectors are closest in meaning — even if they use completely different words.

## Why FAISS Can't Replace PostgreSQL

FAISS is purely a search index. It can't:
- Store patient records or audit results
- Maintain relationships between tables (e.g., linking an audit result to the specific guideline it references)
- Let you browse, filter, or aggregate data through an API
- Track audit job progress or store error messages

## Why Guidelines Are in Both Places

Guidelines exist in **both** systems because each uses them differently:

| Storage | Purpose |
|---------|---------|
| **FAISS** | Finding the right guidelines by semantic search |
| **PostgreSQL** | Storing guideline details, linking them to audit results, serving them via the API |

## How They Work Together in the Pipeline

```
Step 1 — Query Agent
   Input:  Patient diagnosed with "Osteoarthritis of knee"
   Output: Search query → "NICE guidelines for knee osteoarthritis treatment"

Step 2 — FAISS Search
   The query is converted to a 768-dimensional vector using PubMedBERT.
   FAISS compares this vector against all 1,656 guideline vectors.
   Returns the indices of the top 5 most similar guidelines: [42, 107, 890, 1201, 455]

Step 3 — PostgreSQL Retrieval
   SELECT * FROM guidelines WHERE id IN (42, 107, 890, 1201, 455)
   Returns the full text, title, and URL for each matched guideline.

Step 4 — Scorer Agent
   Receives the patient's treatments + the 5 guideline passages.
   Uses an LLM to evaluate: "Did the GP follow these guidelines?"
   Produces a score (0.0 to 1.0) with explanations.

Step 5 — PostgreSQL Storage
   INSERT INTO audit_results (patient_id=..., overall_score=0.8, details_json=...)
   The result is permanently stored, linked to both the patient and the guidelines used.
```

## Analogy

Think of it like a library:

- **FAISS** is the librarian who understands your question and knows which shelf to look at — "You're asking about knee problems? Try these 5 books."
- **PostgreSQL** is the catalogue system that stores every book's details, tracks who borrowed what, and lets you browse by author, date, or category.

You need both. The librarian finds the right books; the catalogue keeps everything organised.
