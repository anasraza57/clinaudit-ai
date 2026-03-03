# Scorer Agent Explained

## What Does the Scorer Agent Do?

The Scorer Agent is **Stage 4** (final) of the 4-agent pipeline. It takes two inputs:

1. **ExtractionResult** (from the Extractor Agent) — what the GP actually did: diagnoses, treatments, referrals, investigations, procedures
2. **RetrievalResult** (from the Retriever Agent) — what NICE guidelines recommend for each diagnosis

It combines them and asks an LLM to evaluate whether the documented clinical care follows the guidelines, producing a per-diagnosis adherence score (+1 adherent / -1 non-adherent) with explanations.

Think of it as a virtual clinical auditor who reads the patient's file, reads the relevant guidelines, and gives a verdict on each diagnosis.

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
ExtractionResult (from Extractor)
        ↓
RetrievalResult (from Retriever)
        ↓
   ScorerAgent.score()
        ↓
   For each diagnosis:
     1. Look up the patient's episode (treatments, referrals, investigations)
     2. Look up the retrieved guidelines
     3. Format the scoring prompt
     4. Call the LLM (temperature=0 for deterministic scoring)
     5. Parse the structured response
        ↓
   ScoringResult (per-diagnosis scores + aggregate)
```

### The Scoring Rubric

The system uses **binary scoring** per diagnosis: **+1 (adherent)** or **-1 (non-adherent)**. This matches the approach used in the original GuidelineGuard paper (Shahriyear, 2024).

#### What Each Score Means

| Score | Meaning | When It's Given |
|-------|---------|-----------------|
| **+1 (Adherent)** | The GP's documented management broadly follows NICE guidelines | At least one appropriate clinical action was taken |
| **-1 (Non-adherent)** | The GP's documented management does not follow NICE guidelines | No relevant actions taken, or actions contradict guidelines |

#### When a Diagnosis Scores +1 (Adherent)

A diagnosis receives +1 if **any** of the following are true:

1. **A relevant referral was made** — e.g., referral to physiotherapy, orthopaedics, rheumatology, or "further care". For most MSK conditions, NICE guidelines recommend physiotherapy as first-line management. A physio referral alone is sufficient for +1.
2. **Appropriate treatments were prescribed** — e.g., NSAIDs for low back pain, analgesics for osteoarthritis, corticosteroid injection for tennis elbow.
3. **Relevant investigations were ordered** — e.g., X-ray for suspected fracture, blood tests for inflammatory markers in suspected gout.
4. **The documented actions show reasonable clinical engagement** — the GP took some action that aligns with what the guidelines recommend.

#### When a Diagnosis Scores -1 (Non-adherent)

A diagnosis receives -1 if **either** of the following are true:

1. **No actions at all** — the diagnosis is documented but there are NO treatments, NO referrals, AND NO investigations. The GP identified the condition but the coded record shows no management whatsoever.
2. **Actions contradict guidelines** — e.g., prescribing opioids as first-line treatment for chronic low back pain (NICE specifically recommends against this).

#### The "Benefit of the Doubt" Principle

The scorer is designed to be **generous rather than punitive**, for good reason:

- **Coded records are incomplete.** Our data comes from SNOMED-coded clinical entries. Many GP actions don't generate SNOMED codes: verbal advice ("take paracetamol, rest, apply ice"), over-the-counter recommendations, lifestyle guidance, and clinical reasoning are all absent. Prescriptions may be in a separate prescribing system. The absence of a coded treatment does NOT mean no treatment was given.
- **GPs exercise clinical judgment.** A GP may have valid reasons for deviating from guidelines that aren't captured in the record (e.g., patient has a contraindication, already tried the recommended treatment, prefers a different approach).
- **Referrals count as management.** A referral to physiotherapy, a specialist, or "further care" IS a form of management — it means the GP has taken action and directed the patient to appropriate next steps.

#### Aggregate Score

The patient-level aggregate score is the **proportion of adherent diagnoses**:

```
aggregate_score = adherent_count / (adherent_count + non_adherent_count)
```

- Range: 0.0 (all diagnoses non-adherent) to 1.0 (all adherent)
- Diagnoses that errored during scoring are **excluded** from the aggregate — they don't count for or against the patient
- A patient with 3 diagnoses scoring +1, +1, -1 gets an aggregate of 0.667

#### Real Examples from Our Dataset

| Patient | Diagnosis | Actions | Score | Why |
|---------|-----------|---------|-------|-----|
| 01aa45a7 | Hip pain | Referral to physiotherapist | **+1** | Physio referral is first-line NICE management for hip pain |
| 01fde560 | Elbow joint pain | None | **-1** | No treatments, no referrals, no investigations at all |
| 001e1fe6 | Finger pain | None | **-1** | Diagnosis only, no clinical actions documented |
| 00abc394 | Neck pain | Referral for further care | **+1** | Referral demonstrates clinical engagement |
| 0381f1e4 | Shoulder pain | Referral to physiotherapist, hormone treatments (unrelated) | **+1** | Physio referral is relevant and appropriate |

### The Scoring Prompt

The prompt is the most important part of the Scorer. It's what the LLM sees when evaluating adherence. Here's its structure:

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
**About the data:** These are SNOMED-coded clinical records, NOT free-text
notes. Many GP actions are NOT captured in coded data — verbal advice,
over-the-counter recommendations, prescriptions from separate systems, and
clinical reasoning are typically absent.

**Scoring guidance:**
- Score +1 (ADHERENT) if ANY of the following are true:
  - A relevant referral was made (referrals ARE a form of management)
  - Appropriate treatments were prescribed
  - Relevant investigations were ordered
  - The documented actions show reasonable clinical engagement
- Score -1 (NON-ADHERENT) only if:
  - NO treatments, NO referrals, AND NO investigations at all
  - Actions clearly CONTRADICT the guidelines
- Give the benefit of the doubt
- A physiotherapy or specialist referral alone is sufficient for +1
- Base evaluation ONLY on provided guidelines, not general medical knowledge

## Output Format
Score: +1 or -1
Explanation: 2-3 sentence explanation
Guidelines Followed: comma-separated list or "None"
Guidelines Not Followed: comma-separated list or "None"
```

Key improvements over Cyprian's prompt:
- **Includes referrals, investigations, and procedures** — not just treatments
- **Includes full guideline text** — up to 2,000 characters (not 500)
- **Multiple guidelines** — includes all top-K retrieved guidelines, sorted by relevance
- **Structured output format** — explicitly asks for guidelines followed/not followed
- **"Benefit of the doubt" rule** — prevents over-penalisation for reasonable clinical judgment
- **"ONLY on provided guidelines" rule** — prevents the LLM from using its own medical knowledge
- **Sparse data context** — explains that these are SNOMED-coded records where many GP actions aren't captured, so absence of coded treatments ≠ no treatment
- **Referral-as-management** — explicitly states that referrals to physiotherapy/specialists are sufficient for +1 (first-line NICE management for most MSK conditions)
- **Example output** — includes a concrete example to prevent the LLM from misinterpreting format placeholders as literal brackets

### Response Parsing

The LLM returns a structured response that we parse with regex:

```python
_SCORE_PATTERN = re.compile(r"Score:\s*\[?([+-]?1)\]?", re.IGNORECASE)
_EXPLANATION_PATTERN = re.compile(
    r"Explanation:\s*(.+?)(?=\nGuidelines Followed:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FOLLOWED_PATTERN = re.compile(
    r"Guidelines Followed:\s*(.+?)(?=\nGuidelines Not Followed:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_NOT_FOLLOWED_PATTERN = re.compile(
    r"Guidelines Not Followed:\s*(.+?)$",
    re.IGNORECASE | re.DOTALL,
)
```

Note the `re.IGNORECASE` flag — this fixes Cyprian's case-sensitivity bug. Whether the LLM outputs "Score:", "score:", or "SCORE:", we'll catch it.

The `\[?` and `\]?` in the score pattern handle LLMs that output `Score: [+1]` with brackets (copying the format from prompt templates literally). Without this, the regex fails to match and ALL scores default to -1 — a critical parsing bug we discovered when all 49 initial patients scored 0.0 adherence.

The parser also handles:
- Missing plus sign (`1` treated same as `+1`)
- Square brackets around score (`[+1]` and `[-1]`)
- Multiline explanations
- "None" for empty lists
- Extra whitespace
- Default to -1 if parsing fails entirely (conservative default)

### Aggregate Scoring

Same formula as Cyprian's (proportion of adherent diagnoses):

```python
@property
def aggregate_score(self) -> float:
    scored = self.adherent_count + self.non_adherent_count
    if scored == 0:
        return 0.0
    return self.adherent_count / scored
```

One key difference: **errors are excluded from the aggregate**. If a diagnosis fails to score (API error), it doesn't count against the patient. Only successfully scored diagnoses affect the aggregate.

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

### Data Flow Through the Scorer

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
└─── Scorer combines them ──→ For "Low back pain":
                                Prompt includes:
                                  - Diagnosis: Low back pain
                                  - Treatments: Ibuprofen
                                  - Referrals: Physio
                                  - Guidelines: [3 passages]
                                     ↓
                                  LLM evaluates
                                     ↓
                                  Score: +1
                                  Explanation: "Treatments align..."
                                  Followed: ["NSAIDs", "physio referral"]
                                  Not Followed: ["None"]
```

### The Dual-Input Challenge

The Scorer is the only agent that receives input from **two** previous agents:
- **Extractor** provides what the GP did (treatments, referrals, etc.)
- **Retriever** provides what the GP should have done (guidelines)

Cyprian handled this by running two Flask servers and a global state dict that waited for both inputs. We handle it much more simply — the `score()` method takes both as parameters:

```python
async def score(
    self,
    extraction: ExtractionResult,
    retrieval: RetrievalResult,
) -> ScoringResult:
```

The pipeline orchestrator (Phase 6) will call this with both results. No servers, no global state, no waiting.

## Side-by-Side Comparison

| Aspect | Cyprian's Scorer | Our Scorer |
|--------|-----------------|------------|
| **LLM model** | GPT-3.5-turbo (weakest) | GPT-4o-mini via abstraction (swappable) |
| **Clinical context** | Treatments only | Treatments + referrals + investigations + procedures |
| **Guideline text** | 1 guideline, 500 chars | Top-K guidelines, 2,000 chars (configurable) |
| **Guideline selection** | Substring match (`diag in text`) | Pre-matched by Retriever (semantic search) |
| **Score parsing** | Case-sensitive bug (always -1) | Case-insensitive regex, handles brackets/edge cases |
| **Output structure** | Score + explanation only | Score + explanation + followed + not followed |
| **Error handling** | None (crashes on API error) | Per-diagnosis error capture, continues processing |
| **Aggregate formula** | `followed / total` (same) | `adherent / scored` (excludes errors) |
| **Architecture** | Flask + JSON-RPC + global state | Direct function call, no servers |
| **Testing** | 5 manual test cases in notebook | 32 automated unit tests |
| **Async** | No | Yes (async/await) |
| **Temperature** | 0 (same) | 0 (same — deterministic scoring) |
| **Logging** | None | Structured logging with patient context |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `scorer_max_guideline_chars` | 2000 | Max characters of guideline text in prompt (vs Cyprian's 500) |
| `ai_provider` | `openai` | Which LLM provider to use (via abstraction layer) |
| `openai_model` | `gpt-4o-mini` | Default model (vs Cyprian's gpt-3.5-turbo) |

## Test Coverage

34 tests covering:

- **Response parsing** (9 tests): adherent/non-adherent, case insensitive, multiline, whitespace, defaults, missing fields, square bracket format
- **Data classes** (8 tests): DiagnosisScore creation, ScoringResult aggregate calculation (all adherent, all non-adherent, mixed, with errors, empty), summary structure
- **ScorerAgent** (12 tests): single diagnosis, multi-diagnosis, LLM call verification, prompt content (diagnosis terms, treatments, guidelines), temperature setting, empty inputs, non-adherent results, error handling, missing episodes, guideline title storage, field preservation
- **Guideline formatting** (4 tests): with guidelines, empty, max chars truncation, rank ordering

## What Happens Next?

The ScoringResult is the **final output** of the 4-agent pipeline. In Phase 6 (Pipeline Integration), we'll:
1. Chain all 4 agents together (Extractor → Query → Retriever → Scorer)
2. Build API endpoints to trigger audits for single patients or batches
3. Store results in the database
4. In Phase 7, compare our scores against the 120 gold-standard human audits
