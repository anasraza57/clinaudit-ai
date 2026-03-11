# Compliance Auditor Agent Explained

## What Does the Compliance Auditor Agent Do?

The Compliance Auditor Agent (formerly `ComplianceAuditorAgent`, renamed Phase 8a) is **Stage 4** (final) of the 4-agent pipeline. It takes two inputs:

1. **ExtractionResult** (from the Consultation Insight Agent) — what the GP actually did: diagnoses, treatments, referrals, investigations, procedures
2. **RetrievalResult** (from the Guideline Evidence Finder) — what NICE guidelines recommend for each diagnosis

It combines them and asks an LLM to evaluate whether the documented clinical care follows the guidelines, producing a per-diagnosis score on a **5-level scale** (-2 to +2) with confidence scores, NICE guideline citations, and explanations.

Think of it as a virtual clinical auditor who reads the patient's file, reads the relevant guidelines, and gives a graded verdict on each diagnosis.

## Cyprian's Original Implementation

### His Code (from `scorer_deployed.ipynb`)

```python
def scorer_node(state: State) -> State:
    if state['diagnoses'] is None or state['retrieved_guidelines'] is None:
        return state  # Wait for both inputs
    scores = []
    for diag in state['diagnoses']:
        treat = ', '.join(state['treatments'].get(diag, []))
        relevant_guidelines = [g for sublist in state['retrieved_guidelines']
                               for sim, g in sublist if diag.lower() in g.lower()]
        guidelines_text = '\n\n'.join(relevant_guidelines[:1])[:500]  # Limit to avoid token overflow
        prompt = f"""Given the diagnosis: {diag}
Treatments in the note: {treat}
Relevant guidelines: {guidelines_text}

Evaluate if the treatments follow the guidelines properly.
- If treatments follow the guidelines, output score: +1 and a brief explanation.
- If treatments do not follow the guidelines, output score: -1 and a brief explanation.
- If the diagnosis is mentioned but no treatment is provided, output score: -1 and a brief explanation.

Output format: Score: [ +1 or -1 ]\nExplanation: [reasoning behind score]"""
        response = llm.invoke(prompt)
        content = response.content.lower()
        score_match = re.search(r"Score:\s*(\+1|-1)", content)
        score = 1 if score_match and score_match.group(1) == "+1" else -1
        scores.append(score)
    state['scores'] = scores
    followed = sum(1 for s in scores if s == 1)
    total = len(scores)
    state['final_score'] = followed / total if total > 0 else 0
    return state
```

### What He Also Built Around It

Cyprian didn't just have the scorer function — he wrapped it in an elaborate server architecture:

```python
# Global mutable state dictionary
scorer_state = {
    "medical_note": None,
    "diagnoses": None,
    "treatments": None,
    "procedures": None,
    "retrieved_guidelines": None,
    "scores": None,
    "final_score": None
}

# Flask JSON-RPC server on port 5001
scorer_app = Flask(__name__)

@scorer_app.post("/rpc")
def scorer_rpc():
    # Handle JSON-RPC methods: submit_extracted, submit_guidelines, get_score
    ...

def compute_if_ready():
    # Check if both inputs arrived, then run LangGraph
    if scorer_state['diagnoses'] is not None and scorer_state['retrieved_guidelines'] is not None:
        final_state = graph.invoke(initial_state)
        scorer_state['scores'] = final_state['scores']
        scorer_state['final_score'] = final_state['final_score']
```

He ran two Flask servers in threads (Retriever on port 5000, Scorer on port 5001) inside a Colab notebook, communicating via JSON-RPC. The scorer waited for two separate inputs (extracted data from the Extractor and guidelines from the Retriever) before computing.

### Problems with Cyprian's Approach

Let's go through each problem one by one:

#### 1. Only Uses Treatments, Ignores Referrals and Investigations

```python
treat = ', '.join(state['treatments'].get(diag, []))
```

He only passes **treatments** to the prompt. But NICE guidelines often recommend **referrals** (e.g., "refer to physiotherapy") and **investigations** (e.g., "order blood tests for inflammatory markers"). A patient who was correctly referred to physiotherapy for low back pain would get scored as non-adherent because the scorer only sees treatments.

#### 2. Guideline Matching Is Naive and Lossy

```python
relevant_guidelines = [g for sublist in state['retrieved_guidelines']
                       for sim, g in sublist if diag.lower() in g.lower()]
guidelines_text = '\n\n'.join(relevant_guidelines[:1])[:500]
```

Two problems here:

**First**, he filters guidelines by checking if the diagnosis term appears literally in the guideline text. If the diagnosis is "Low back pain" but the guideline talks about "lumbar spine" or "non-specific back pain", it won't match. This throws away potentially relevant guidelines.

**Second**, he takes only 1 guideline (`[:1]`) and truncates to 500 characters (`[:500]`). A typical NICE guideline section is 1,000-3,000 characters. 500 characters often cuts off mid-sentence, losing critical recommendations. Our Retriever already selected the top-K most relevant guidelines — we should use them, not throw them away.

#### 3. Case-Sensitive Regex Bug

```python
content = response.content.lower()  # Converts to lowercase
score_match = re.search(r"Score:\s*(\+1|-1)", content)  # Searches for capital "Score:"
```

He converts the response to lowercase but then searches for "Score:" with a capital S. This will **never match** because `content` is all lowercase. The result? Every score defaults to -1 (non-adherent), regardless of what the LLM said. This is a silent, critical bug.

#### 4. Uses GPT-3.5-Turbo (Weakest Model)

```python
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
```

GPT-3.5-turbo is the cheapest and weakest OpenAI model for complex reasoning. Medical adherence evaluation requires understanding nuanced clinical guidelines and applying them to specific patient scenarios. This is a task that benefits significantly from a more capable model.

#### 5. No Error Handling

```python
response = llm.invoke(prompt)  # What if this fails?
content = response.content.lower()  # What if content is None?
```

If the API call fails (network error, rate limit, timeout), the entire pipeline crashes. No retry, no fallback, no error logging.

#### 6. Global Mutable State

```python
scorer_state = {
    "diagnoses": None,
    "treatments": None,
    "retrieved_guidelines": None,
    ...
}
```

A single global dictionary holds all state. If two patients are processed concurrently, they overwrite each other's data. This is a classic race condition.

#### 7. Over-Engineered Architecture

Running Flask JSON-RPC servers in threads inside a Colab notebook to do something that's fundamentally a function call. The Retriever sends guidelines to the Scorer via HTTP POST to localhost — this could just be `scorer.score(guidelines)`.

#### 8. LangGraph for a Two-Step Pipeline

The "workflow" is:
```
retriever → scorer → END
```

That's two functions in sequence. LangGraph's graph abstractions add complexity without any benefit for a linear chain this simple.

## Our Implementation

### Architecture Overview

```
ExtractionResult (from Consultation Insight Agent)
        ↓
RetrievalResult (from Guideline Evidence Finder)
        ↓
   ComplianceAuditorAgent.score()
        ↓
   For each diagnosis:
     1. Look up the patient's episode (treatments, referrals, investigations)
     2. Look up the retrieved guidelines
     3. Format the scoring prompt
     4. Call the LLM (temperature=0 for deterministic scoring)
     5. Parse the structured response (score, judgement, confidence, citation)
        ↓
   ScoringResult (per-diagnosis scores + aggregate)
```

### The Scoring Rubric

The system uses a **5-level grading scale** per diagnosis (upgraded from binary +1/-1 in Phase 8b). Each score also includes a **confidence** value (0.0-1.0) and a **cited NICE guideline** quote.

#### What Each Score Means

| Score | Judgement | Meaning | When It's Given |
|-------|-----------|---------|-----------------|
| **+2 (COMPLIANT)** | Fully follows guidelines | Multiple recommended actions documented | Appropriate treatment AND referral AND investigation present |
| **+1 (PARTIALLY COMPLIANT)** | Minor gaps | Some guideline-aligned actions, minor omissions | Referral made but first-line treatment absent from coded record |
| **0 (NOT RELEVANT)** | Cannot assess | Guideline not applicable or data too sparse | No meaningful basis for comparison |
| **-1 (NON-COMPLIANT)** | Major deviation | No documented management at all | No treatments, referrals, or investigations; or clear deviation without safety risk |
| **-2 (RISKY NON-COMPLIANT)** | Safety risk | Actions contradict guidelines with potential harm | Contraindicated treatment prescribed, or safety-critical red flag missed |

#### The "Benefit of the Doubt" Principle

The scorer is designed to be **generous rather than punitive**, for good reason:

- **Coded records are incomplete.** Our data comes from SNOMED-coded clinical entries. Many GP actions don't generate SNOMED codes: verbal advice ("take paracetamol, rest, apply ice"), over-the-counter recommendations, lifestyle guidance, and clinical reasoning are all absent. Prescriptions may be in a separate prescribing system. The absence of a coded treatment does NOT mean no treatment was given.
- **GPs exercise clinical judgment.** A GP may have valid reasons for deviating from guidelines that aren't captured in the record (e.g., patient has a contraindication, already tried the recommended treatment, prefers a different approach).
- **Referrals count as management.** A referral to physiotherapy, a specialist, or "further care" IS a form of management — it means the GP has taken action and directed the patient to appropriate next steps. A referral alone warrants at least +1.

#### Aggregate Score

The patient-level aggregate score is a **normalised mean** of all scored diagnoses:

```
aggregate_score = mean((score + 2) / 4)   for each non-error diagnosis
```

This maps the [-2, +2] scale to [0.0, 1.0]:
- -2 → 0.0, -1 → 0.25, 0 → 0.5, +1 → 0.75, +2 → 1.0
- Diagnoses that errored during scoring are **excluded** from the aggregate
- A patient with 3 diagnoses scoring +2, +1, -1 gets: mean((4/4 + 3/4 + 1/4)) = mean(1.0 + 0.75 + 0.25) = 0.667

#### Real Examples from Our Dataset

| Patient | Diagnosis | Actions | Score | Judgement | Why |
|---------|-----------|---------|-------|-----------|-----|
| 01aa45a7 | Hip pain | Referral to physiotherapist | **+1** | PARTIAL | Physio referral is appropriate but no other coded actions |
| 01fde560 | Elbow joint pain | None | **-1** | NON-COMPLIANT | No treatments, no referrals, no investigations at all |
| 001e1fe6 | Finger pain | None | **-1** | NON-COMPLIANT | Diagnosis only, no clinical actions documented |
| 00abc394 | Neck pain | Referral + NSAIDs + exercises | **+2** | COMPLIANT | Multiple guideline-recommended actions present |
| 0381f1e4 | Shoulder pain | Opioids prescribed (contraindicated) | **-2** | RISKY | Guidelines advise against opioids for this condition |

### The Scoring Prompt

The prompt is the most important part of the Compliance Auditor. It's what the LLM sees when evaluating adherence. Here's its structure (updated in Phase 8b for 5-level scoring):

```
You are a clinical audit expert evaluating whether a GP's management of a
musculoskeletal condition adheres to NICE clinical guidelines.

## Patient Information
**Diagnosis:** Low back pain
**Index Date:** 2024-01-15

**Documented Actions (from coded SNOMED records):**
- Treatments: Ibuprofen 400mg tablets
- Referrals: Physiotherapy referral
- Investigations: None documented
- Procedures: None documented

## Relevant NICE Guidelines
### Low back pain and sciatica in over 16s
Offer exercise therapy as first-line treatment. Consider NSAIDs for short-term
pain relief...

## Task
Evaluate whether the documented clinical actions follow the NICE guidelines
for this diagnosis.

## Important Rules — READ CAREFULLY
**About the data:** These are SNOMED-coded clinical records, NOT free-text notes.

**Scoring guidance — use this 5-level scale:**
- Score +2 (COMPLIANT): Actions clearly and fully align with NICE guidelines
- Score +1 (PARTIALLY COMPLIANT): Some aligned actions but with minor gaps
- Score  0 (NOT RELEVANT): Guideline not applicable or data too sparse
- Score -1 (NON-COMPLIANT): No documented management or clear deviation
- Score -2 (RISKY NON-COMPLIANT): Actions contradict guidelines with patient harm risk

**General principles:**
- Give the benefit of the doubt
- A physio/specialist referral alone warrants at least +1
- Base evaluation ONLY on provided guidelines, not general medical knowledge
- You MUST cite the specific guideline text that informed your judgement

## Output Format
Score: -2, -1, 0, +1, or +2
Judgement: COMPLIANT, PARTIALLY COMPLIANT, NOT RELEVANT, NON-COMPLIANT, or RISKY NON-COMPLIANT
Confidence: a number between 0.0 and 1.0
Cited Guideline: a direct quote from the NICE guideline text, or "None"
Explanation: 2-3 sentence explanation
Guidelines Followed: comma-separated list or "None"
Guidelines Not Followed: comma-separated list or "None"

Example:
Score: +1
Judgement: PARTIALLY COMPLIANT
Confidence: 0.75
Cited Guideline: "Consider referral to physiotherapy. Do not offer opioids."
Explanation: The GP referred the patient to physiotherapy which is recommended.
Guidelines Followed: Physiotherapy referral
Guidelines Not Followed: Exercise therapy advice, NSAID prescription
```

Key improvements over Cyprian's prompt:
- **5-level grading scale** — captures nuance that binary +1/-1 could not (partial compliance, safety-critical non-compliance)
- **Confidence scores** — LLM reports how certain it is (0.0-1.0), useful for flagging borderline cases
- **NICE guideline citations** — direct quotes from the source text, improving transparency and auditability
- **Includes referrals, investigations, and procedures** — not just treatments
- **Includes full guideline text** — up to 2,000 characters (not 500)
- **Multiple guidelines** — includes all top-K retrieved guidelines, sorted by relevance
- **Structured 7-field output format** — score, judgement, confidence, cited guideline, explanation, followed, not followed
- **"Benefit of the doubt" rule** — prevents over-penalisation for reasonable clinical judgment
- **"ONLY on provided guidelines" rule** — prevents the LLM from using its own medical knowledge
- **Sparse data context** — explains that these are SNOMED-coded records where many GP actions aren't captured
- **Example output** — includes a concrete example to prevent the LLM from misinterpreting the format

### Response Parsing

The LLM returns a 7-field structured response that we parse with regex. Updated in Phase 8b for the 5-level scale:

```python
_SCORE_PATTERN = re.compile(r"Score:\s*\[?([+-]?[012])\]?", re.IGNORECASE)
_JUDGEMENT_PATTERN = re.compile(r"Judgement:\s*(.+)", re.IGNORECASE)
_CONFIDENCE_PATTERN = re.compile(r"Confidence:\s*([\d.]+)", re.IGNORECASE)
_CITED_GUIDELINE_PATTERN = re.compile(
    r"Cited Guideline:\s*(.+?)(?=\nExplanation:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_EXPLANATION_PATTERN = re.compile(
    r"Explanation:\s*(.+?)(?=\nGuidelines Followed:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# ... plus _FOLLOWED_PATTERN and _NOT_FOLLOWED_PATTERN
```

Note the `re.IGNORECASE` flag — this fixes Cyprian's case-sensitivity bug. Whether the LLM outputs "Score:", "score:", or "SCORE:", we'll catch it.

The score pattern `[+-]?[012]` now accepts values from -2 to +2. The `\[?` and `\]?` handle LLMs that output brackets around the score.

The parser handles all 7 fields:
- **Score**: -2, -1, 0, +1, +2 (with optional brackets/plus sign)
- **Judgement**: matched to `JUDGEMENT_LABELS` dict, falls back to label lookup by score
- **Confidence**: float 0.0-1.0, clamped to range, defaults to 0.0
- **Cited Guideline**: direct NICE quote, strips surrounding quotes
- **Explanation**: multiline text
- **Guidelines Followed/Not Followed**: comma-separated lists
- Default to -1 (NON-COMPLIANT) if score parsing fails entirely (conservative default)

### Aggregate Scoring

Updated in Phase 8b — now uses a **normalised mean** instead of Cyprian's simple proportion:

```python
@property
def aggregate_score(self) -> float:
    scored = [ds for ds in self.diagnosis_scores if ds.error is None]
    if not scored:
        return 0.0
    normalized = [(ds.score + 2) / 4 for ds in scored]
    return sum(normalized) / len(normalized)
```

This maps each score from [-2, +2] to [0.0, 1.0] then averages. Errors are excluded — if a diagnosis fails to score (API error), it doesn't count against the patient.

Backward-compatible properties `adherent_count` and `non_adherent_count` are preserved for legacy code: adherent = compliant + partial, non-adherent = non-compliant + risky.

### Guideline Formatting

Guidelines are sorted by rank (best match first) and truncated intelligently:

```python
def _format_guidelines(self, dg: DiagnosisGuidelines) -> str:
    if not dg.guidelines:
        return "No relevant guidelines found."

    parts = []
    total_chars = 0

    for match in sorted(dg.guidelines, key=lambda g: g.rank):
        header = f"### {match.title}\n"
        text = match.clean_text

        addition = header + text + "\n\n"
        if total_chars + len(addition) > self._max_guideline_chars:
            remaining = self._max_guideline_chars - total_chars
            if remaining > len(header) + 50:  # Only add if meaningful
                parts.append(header + text[:remaining - len(header) - 5] + "...")
            break

        parts.append(addition)
        total_chars += len(addition)

    return "\n".join(parts) if parts else "No relevant guidelines found."
```

Instead of a hard cut at 500 characters, we:
1. Add guidelines one by one in order of relevance
2. Stop when we'd exceed the limit (default 2,000 chars, configurable)
3. If we can fit a partial last guideline meaningfully (>50 chars of content), we include it with an ellipsis
4. Include the guideline title as a markdown header for structure

### Error Handling

Every LLM call is wrapped in a try/except:

```python
try:
    response = await self._ai_provider.chat_simple(prompt, temperature=0.0)
    parsed = parse_scoring_response(response)
    return DiagnosisScore(...)
except Exception as e:
    logger.error("Scoring failed for %r: %s", diagnosis_term, e)
    return DiagnosisScore(
        ...,
        score=-1,
        explanation="Scoring failed due to an error.",
        error=str(e),
    )
```

On error:
- The error is logged with context (diagnosis, patient)
- A DiagnosisScore is still returned (with `error` field set)
- The pipeline continues processing other diagnoses
- The error count is tracked in the ScoringResult
- Errors are excluded from the aggregate score

### Data Flow Through the Compliance Auditor

```
ExtractionResult          RetrievalResult
├── pat_id: "pat-001"     ├── pat_id: "pat-001"
├── episodes:             └── diagnosis_guidelines:
│   └── Episode 1:            └── "Low back pain":
│       ├── date: 2024-01-15      ├── guideline 1 (rank 1)
│       ├── diagnosis: LBP        ├── guideline 2 (rank 2)
│       ├── treatment: Ibuprofen  └── guideline 3 (rank 3)
│       └── referral: Physio
│
└─── Auditor combines them ──→ For "Low back pain":
                                 Prompt includes:
                                   - Diagnosis: Low back pain
                                   - Treatments: Ibuprofen
                                   - Referrals: Physio
                                   - Guidelines: [3 passages]
                                      ↓
                                   LLM evaluates
                                      ↓
                                   Score: +1
                                   Judgement: PARTIALLY COMPLIANT
                                   Confidence: 0.75
                                   Cited Guideline: "Consider referral to..."
                                   Explanation: "GP referred correctly..."
                                   Followed: ["Physio referral"]
                                   Not Followed: ["Exercise therapy"]
```

### The Dual-Input Challenge

The Compliance Auditor is the only agent that receives input from **two** previous agents:
- **Consultation Insight Agent** provides what the GP did (treatments, referrals, etc.)
- **Guideline Evidence Finder** provides what the GP should have done (guidelines)

Cyprian handled this by running two Flask servers and a global state dict that waited for both inputs. We handle it much more simply — the `score()` method takes both as parameters:

```python
async def score(
    self,
    extraction: ExtractionResult,
    retrieval: RetrievalResult,
) -> ScoringResult:
```

The pipeline orchestrator calls this with both results. No servers, no global state, no waiting.

## Side-by-Side Comparison

| Aspect | Cyprian's Scorer | Our Compliance Auditor |
|--------|-----------------|----------------------|
| **Scoring scale** | Binary +1/-1 | 5-level: -2 to +2 with judgement labels |
| **Confidence** | None | 0.0-1.0 per diagnosis |
| **Guideline citations** | None | Direct NICE quotes per diagnosis |
| **LLM model** | GPT-3.5-turbo (weakest) | GPT-4o-mini / Ollama (swappable via provider abstraction) |
| **Clinical context** | Treatments only | Treatments + referrals + investigations + procedures |
| **Guideline text** | 1 guideline, 500 chars | Top-K guidelines, 2,000 chars (configurable) |
| **Guideline selection** | Substring match (`diag in text`) | Pre-matched by Guideline Evidence Finder (semantic search) |
| **Score parsing** | Case-sensitive bug (always -1) | Case-insensitive regex, 7 fields, handles brackets/edge cases |
| **Output structure** | Score + explanation only | Score + judgement + confidence + citation + explanation + followed + not followed |
| **Error handling** | None (crashes on API error) | Per-diagnosis error capture, continues processing |
| **Aggregate formula** | `followed / total` | `mean((score + 2) / 4)` normalised [0,1], excludes errors |
| **Architecture** | Flask + JSON-RPC + global state | Direct function call, no servers |
| **Testing** | 5 manual test cases in notebook | 48+ automated unit tests |
| **Async** | No | Yes (async/await) |
| **Temperature** | 0 (same) | 0 (same — deterministic scoring) |
| **Logging** | None | Structured logging with patient context |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `scorer_max_guideline_chars` | 2000 | Max characters of guideline text in prompt (vs Cyprian's 500) |
| `ai_provider` | `openai` | Which LLM provider to use — `openai`, `ollama`, or `local` |
| `openai_model` | `gpt-4o-mini` | Default model (vs Cyprian's gpt-3.5-turbo) |
| `ollama_model` | `mistral-small` | Local Ollama model (Phase 8c) — 22B, good structured output |

## Test Coverage

48+ tests covering:

- **AuditJudgement enum** (2 tests): enum values, label coverage
- **Response parsing** (11 tests): all 5 score levels, confidence parsing, cited guideline extraction, case insensitive, multiline, whitespace, defaults, missing fields
- **Data classes** (10 tests): DiagnosisScore with judgement/confidence fields, ScoringResult per-level counters, backward-compat adherent/non_adherent properties, new aggregate formula, summary structure
- **ComplianceAuditorAgent** (14 tests): single diagnosis, multi-diagnosis, LLM call verification, prompt content, temperature setting, empty inputs, risky scores, partial scores, error handling, missing episodes, guideline title storage, field preservation
- **Guideline formatting** (4 tests): with guidelines, empty, max chars truncation, rank ordering

## What Happens Next?

The ScoringResult is the **final output** of the 4-agent pipeline. Next steps:
1. **Evaluation framework** — per-agent metrics (Extraction P/R/F1, Query relevance, Retriever Recall@k/nDCG/MRR)
2. **System-level metrics** — Accuracy, Precision, Recall, F1, AUROC for 5-class scoring
3. **Gold-standard validation** — compare AI scores against 120 manually-audited cases (when available)
4. **LLM-as-Judge** — evaluation without clinician labels
