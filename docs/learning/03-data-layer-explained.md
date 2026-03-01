# Data Layer Explained

## What is a "Data Layer"?

The data layer is the part of the application responsible for storing, retrieving, and managing data. Think of it as the app's organised filing cabinet. Without it, data would be lost every time the app restarts.

## Our Data Layer Components

### 1. PostgreSQL (the Database)

PostgreSQL is our relational database — it stores structured data in tables with rows and columns, like a spreadsheet but much more powerful. We chose it over SQLite because:
- It handles multiple users at once
- It supports advanced queries
- It's what production apps use

### 2. SQLAlchemy (the ORM)

**ORM** = Object-Relational Mapper. Instead of writing raw SQL like:
```sql
SELECT * FROM patients WHERE pat_id = 'abc-123';
```

We write Python:
```python
patient = await session.get(Patient, pat_id="abc-123")
```

SQLAlchemy translates our Python objects into database operations. Each Python class (called a "model") maps to a database table.

### 3. Alembic (Database Migrations)

When we change our models (add a column, rename a table), the database needs to be updated too. Alembic handles this through **migrations** — versioned scripts that transform the database schema step by step.

Think of it like version control (Git) but for your database structure.

### 4. FAISS (Vector Search)

FAISS stores guideline **embeddings** (numeric representations of text) and lets us find the most similar guidelines to a given query. It's an in-memory index loaded from a file — separate from PostgreSQL because vector similarity search requires specialised data structures.

## Our Database Tables

```
patients
├── id (auto-generated)
├── pat_id (the anonymised UUID from CrossCover)
├── created_at, updated_at

clinical_entries
├── id
├── patient_id → patients.id (foreign key)
├── index_date (first MSK visit date)
├── cons_date (this event's date)
├── concept_id (SNOMED code)
├── term (human-readable name)
├── concept_display (formal SNOMED name)
├── notes
├── category (diagnosis/treatment/referral/etc.)

guidelines
├── id
├── source_id (hash from original dataset)
├── source (e.g., "nice")
├── title, clean_text, url, overview

audit_jobs
├── id
├── status (pending/running/completed/failed)
├── total_patients, processed_patients, failed_patients
├── started_at, completed_at

audit_results
├── id
├── patient_id → patients.id
├── job_id → audit_jobs.id
├── overall_score (0.0 to 1.0)
├── diagnoses_found, guidelines_followed, guidelines_not_followed
├── details_json (full breakdown)
```

## Data Flow

1. **CSV files** (raw data from CrossCover trial and NICE guidelines)
2. **Import service** reads CSVs and inserts rows into PostgreSQL
3. **FAISS index** is loaded from a pre-built `.index` file
4. During an audit, the pipeline queries PostgreSQL for patient data and FAISS for relevant guidelines

## Key Concepts

- **Foreign Key**: A column that references another table's ID, creating a relationship. `clinical_entries.patient_id` points to `patients.id`.
- **Index**: A database optimisation that speeds up lookups on specific columns (like an index in a book).
- **Migration**: A script that changes the database structure. Migrations are applied in order and can be rolled back.
- **Idempotent**: Our import service can be run multiple times safely — it skips records that already exist.
