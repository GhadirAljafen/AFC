> **Version 2 — 2026-09-26.** Supersedes
> `docs/archive/ACADEMIC_DOCUMENTATION_v1_2025.md`, retained unchanged as a
> historical record.
>
> **What changed and why.** Version 1 documented an evaluation section (§6) whose
> figures were *predictions*, not measurements — §6.3 was titled "Expected
> Outcomes". Those predictions have since been tested against the AVeriTeC
> benchmark and were **contradicted**. Section 6 is rewritten below from measured
> results, and §8.1 now lists the limitations that measurement actually revealed.
> Sections describing the confidence-threshold rules have been corrected: those
> rules were unreachable and have been removed.
>
> **Revision 2026-10-10.** §6.5 reports the live open-web evaluation at **89 claims**
> (76 scorable), superseding the 65-, 41- and 15-claim versions. Live full-text evidence
> beats the majority-class baseline and **reaches gold-evidence quality**; the closed
> corpus is now established as understating live performance over four samples.
>
> This section's conclusions reversed twice as the sample grew (n=13 → 35 → 55 → 76)
> without the system changing. §6.5 records the reversals and their cause: small samples
> combined with an uncached temperature-0.2 verdict call. Measurement now uses 3-run
> majority-vote labels throughout, and §6.6 quantifies the residual variance per
> condition. Earlier figures are left in place rather than deleted.
>
> Supporting evidence: `results/claim_detection/RESULTS.md`,
> `results/evidence_retrieval/RESULTS.md`, `results/verdict/FINDINGS.md`.

# An Agentic Automated Fact-Checking System: Architecture, Development, and Implementation

## Abstract

This document presents a comprehensive overview of an agentic automated fact-checking system designed to verify factual claims through an iterative, multi-agent architecture. The system employs Large Language Models (LLMs) for reasoning, web search for evidence retrieval, and an agentic loop for evidence evaluation and refinement. This work documents the system's evolution from initial conception through multiple iterations, culminating in a production-ready implementation that addresses the challenge of misinformation through automated verification.

**Keywords:** Fact-checking, Automated Verification, Agentic Systems, Large Language Models, Evidence Retrieval, Misinformation Detection

---

## 1. Introduction

### 1.1 Background and Motivation

The proliferation of misinformation and disinformation in digital media has created an urgent need for automated fact-checking systems. Traditional manual fact-checking, while accurate, is time-consuming and cannot scale to the volume of claims circulating in modern information ecosystems. This project addresses this challenge by developing an automated system that can:

1. Process individual claims or extract claims from longer texts
2. Retrieve relevant evidence from web sources
3. Evaluate evidence quality and sufficiency
4. Generate verdicts with confidence scores and explanations
5. Operate autonomously through an iterative agentic loop

The system is inspired by modern agentic AI architectures, particularly the principles outlined in Anthropic's "Building Effective Agents" framework, which emphasizes iterative refinement, tool use, and structured reasoning.

### 1.2 Objectives

The primary objectives of this system are:

- **Automation**: Reduce human effort in fact-checking through automated evidence retrieval and evaluation
- **Scalability**: Process multiple claims efficiently, including batch processing from long texts
- **Transparency**: Provide explanations and evidence sources for all verdicts
- **Accuracy**: Balance between decisive verdicts (SUPPORTS/REFUTES) and appropriate uncertainty (NOT_ENOUGH_INFO)
- **Extensibility**: Modular architecture allowing easy integration of new evidence sources and reasoning methods

---

## 2. System Architecture

### 2.1 High-Level Architecture

The system follows a modular, agent-based architecture where specialized agents handle distinct aspects of the fact-checking process. The architecture consists of:

1. **Data Models Layer**: Pydantic models for type safety and validation
2. **LLM Abstraction Layer**: Provider-agnostic interface for language model interactions
3. **Agent Layer**: Specialized agents for each fact-checking stage
4. **Pipeline Orchestration**: Main pipeline coordinating agent interactions
5. **Interface Layer**: CLI and programmatic APIs

### 2.2 Core Components

#### 2.2.1 Data Models

The system uses Pydantic models for type safety and validation:

- **`Claim`**: Represents a user-submitted claim with raw text, normalized text, and metadata
- **`DetectedClaim`**: Extends Claim with position information (character indices, sentence index) and importance scores for claims extracted from longer texts
- **`EvidenceSnippet`**: Represents retrieved evidence with source, text, relevance scores, and metadata
- **`FactCheckVerdict`**: Contains the verdict label (SUPPORTS/REFUTES/NOT_ENOUGH_INFO), confidence score, reasoning, and referenced evidence IDs
- **`FactCheckResult`**: Complete result containing claim, verdict, explanation, and all evidence

#### 2.2.2 LLM Client Abstraction

The system implements a provider-agnostic LLM interface (`LLMClient`) that abstracts over different language model providers. Current implementations include:

- **`OpenAILLMClient`**: Integration with OpenAI's API (GPT-4, GPT-4o-mini, etc.)
- **`DummyLLMClient`**: Testing and development fallback

The abstraction allows seamless switching between providers and supports both completion and chat-style interactions.

#### 2.2.3 Agent Architecture

The system employs six specialized agents:

1. **ClaimUnderstandingAgent**: Normalizes claims and detects check-worthy claims from long texts
2. **RetrievalAgent**: Generates search queries and retrieves evidence from web sources
3. **EvidenceSelectionAgent**: Selects and ranks evidence, prioritizing contradictory evidence
4. **EvidenceEvaluationAgent**: Assesses whether collected evidence is sufficient for fact-checking
5. **ReasoningAndVerdictAgent**: Analyzes evidence and generates verdicts with confidence scores
6. **ExplanationAgent**: Generates natural-language explanations of verdicts

### 2.3 Pipeline Flow

The fact-checking pipeline implements an iterative retrieval-evaluation loop:

```
Input Claim
    ↓
[1] Claim Understanding (Normalization)
    ↓
[2] Retrieval-Evaluation Loop (Iterative)
    ├─→ Retrieve Evidence
    ├─→ Select Top Evidence
    ├─→ Evaluate Sufficiency
    └─→ [If insufficient] → Retrieve More Evidence
    ↓
[3] Reasoning & Verdict Generation
    ↓
[4] Explanation Generation
    ↓
Output: FactCheckResult
```

The iterative loop allows the system to refine its evidence collection until sufficient information is gathered or a maximum iteration limit is reached.

---

## 3. Development History

### 3.1 Phase 1: Initial Structure (Foundation)

The project began with establishing a clean, modular structure:

**Objectives:**
- Set up project infrastructure (Poetry, testing, linting)
- Define core data models
- Create placeholder agents with minimal implementations
- Ensure basic importability and testability

**Key Decisions:**
- Use Pydantic for data validation (type safety, serialization)
- Adopt async/await throughout for I/O operations
- Implement abstract base classes for extensibility
- Use Poetry for dependency management

**Deliverables:**
- Project structure with `src/` layout
- Core models: `Claim`, `EvidenceSnippet`, `FactCheckVerdict`, `FactCheckResult`
- Agent skeletons with method signatures
- Basic CLI entry point
- Smoke tests ensuring system importability

### 3.2 Phase 2: LLM Integration

**Objectives:**
- Integrate real LLM providers (OpenAI)
- Implement provider-agnostic abstraction
- Add fallback mechanisms for development/testing

**Implementation:**
- Created `LLMClient` abstract base class
- Implemented `OpenAILLMClient` with async API calls
- Added `DummyLLMClient` for testing
- Created factory function `get_default_llm_client()` for automatic provider selection

**Challenges:**
- Handling API key configuration
- Managing async operations across the pipeline
- Error handling and fallback strategies

### 3.3 Phase 3: Evidence Retrieval

**Objectives:**
- Integrate web search for evidence retrieval
- Implement query generation using LLMs
- Handle search result parsing and deduplication

**Implementation:**
- Integrated Google Custom Search JSON API
- Implemented LLM-based query generation
- Added query expansion to include refutation-focused queries
- Implemented URL-based deduplication

**Key Features:**
- Multi-query generation (2-4 queries per claim)
- Automatic refutation query inclusion
- Result parsing and EvidenceSnippet creation
- Error handling for missing API keys

### 3.4 Phase 4: Agent Implementation

**Objectives:**
- Implement full agent logic using LLMs
- Create iterative retrieval-evaluation loop
- Implement evidence sufficiency evaluation

**Implementation:**
- **EvidenceEvaluationAgent**: LLM-based sufficiency assessment
- **ReasoningAndVerdictAgent**: Evidence analysis and verdict generation
- **ExplanationAgent**: Natural-language explanation generation
- **Pipeline**: Iterative loop with early termination on sufficiency

**Challenges:**
- Balancing between thoroughness and efficiency
- Handling LLM response parsing (JSON extraction)
- Managing iteration limits to prevent infinite loops

### 3.5 Phase 5: Claim Understanding Enhancement

**Objectives:**
- Extend claim understanding beyond basic normalization
- Add claim detection from long texts
- Implement LLM-based claim normalization

**Implementation:**
- Added `DetectedClaim` model for extracted claims
- Implemented `detect_claims()` method using LLM
- Added `normalize_claim_llm()` for advanced normalization
- Maintained backward compatibility with basic normalization

**Features:**
- Importance scoring for detected claims
- Position tracking (character indices, sentence indices)
- Batch claim detection from articles/posts

### 3.6 Phase 6: System Refinement

**Objectives:**
- Address bias toward NOT_ENOUGH_INFO verdicts
- Improve refutation detection
- Enhance evidence quality

**Problem Identified:**
Initial implementation showed bias toward NOT_ENOUGH_INFO (~70%) with few REFUTES verdicts (~5%), even when evidence contradicted claims.

**Root Causes:**
1. LLM prompt didn't explicitly guide toward refutation detection
2. Search queries didn't explicitly seek refuting evidence
3. Evidence selection didn't prioritize contradictory snippets
4. LLM conservatism defaulting to uncertainty

**Solutions Implemented:**

**Priority 1.1: Enhanced Verdict Prompt**
- Added explicit instructions to look for refuting evidence
- Emphasized REFUTES verdicts when evidence contradicts
- Clearer decision criteria for each verdict type

**Priority 1.2: Refutation Search Queries**
- Modified query generation to explicitly request refutation queries
- Added fallback to include refutation queries even when LLM unavailable
- Examples: "claim X false", "claim X debunked", "claim X incorrect"

**Priority 2.1: Evidence Selection Prioritization**
- Rewrote selection algorithm to prioritize contradictory evidence
- Added keyword detection (false, incorrect, debunked, misleading, etc.)
- Scoring boost (+0.3 base + 0.1 per keyword) for contradictory evidence

**Priority 2.2: Confidence Thresholds**
- Added threshold logic: SUPPORTS requires ≥0.5 confidence
- REFUTES allows lower confidence (≥0.4) with uncertainty notes
- Prevents over-confident SUPPORTS on weak evidence

**Expected Impact:**
- REFUTES verdicts: 5% → 30-40%
- NOT_ENOUGH_INFO: 70% → 30-40%
- More balanced verdict distribution

---

## 4. Current Implementation

### 4.1 System Components

#### 4.1.1 Claim Understanding Agent

The `ClaimUnderstandingAgent` provides three main capabilities:

1. **Basic Normalization** (`normalize_claim`): Synchronous, rule-based text cleanup
   - Whitespace normalization
   - Case normalization
   - Duplicate space removal

2. **Advanced Normalization** (`normalize_claim_llm`): LLM-based claim refinement
   - Removes filler words and qualifiers
   - Standardizes entities (dates, numbers, locations)
   - Ensures fact-checkable form

3. **Claim Detection** (`detect_claims`): Extracts check-worthy claims from long texts
   - Uses LLM to identify factual claims
   - Assigns importance scores (0.0-1.0)
   - Tracks position information
   - Filters by minimum importance threshold

#### 4.1.2 Retrieval Agent

The `RetrievalAgent` implements evidence retrieval through:

1. **Query Generation** (`_generate_queries`):
   - LLM-based query generation (3-5 queries per claim)
   - Explicit refutation query inclusion
   - Fallback to basic queries if LLM unavailable

2. **Web Search** (`_google_search`):
   - Google Custom Search JSON API integration
   - Configurable results per query (default: 5)
   - Error handling for API failures

3. **Evidence Creation**:
   - Converts search results to `EvidenceSnippet` objects
   - URL-based deduplication
   - Metadata extraction (title, URL)

#### 4.1.3 Evidence Selection Agent

The `EvidenceSelectionAgent` implements intelligent evidence ranking:

- **Scoring Algorithm**:
  - Base score from retrieval (if available) or 0.5 default
  - Boost for contradictory evidence: +0.3 base + 0.1 per keyword
  - Keywords: false, incorrect, debunked, misleading, untrue, not true, disproven, refuted, contradicts, disputes, wrong, inaccurate, myth, hoax, fake

- **Selection**: Returns top-k evidence sorted by score

#### 4.1.4 Evidence Evaluation Agent

The `EvidenceEvaluationAgent` assesses evidence sufficiency:

- **LLM-Based Evaluation**:
  - Analyzes collected evidence against claim
  - Determines if evidence is sufficient for fact-checking
  - Provides confidence score and notes

- **Fallback Heuristic**:
  - If LLM unavailable: sufficient if ≥3 evidence snippets
  - Confidence based on evidence count

#### 4.1.5 Reasoning and Verdict Agent

The `ReasoningAndVerdictAgent` generates final verdicts:

- **Enhanced Prompt**:
  - Explicit instructions to look for refuting evidence
  - Clear decision criteria for each verdict type
  - Emphasis on decisiveness when evidence contradicts

- **Confidence Thresholds**:
  - SUPPORTS: Requires ≥0.5 confidence
  - REFUTES: Allows ≥0.4 confidence (with uncertainty notes)
  - NOT_ENOUGH_INFO: No threshold

- **Output**: JSON-structured verdict with label, confidence, reasoning, and evidence IDs

#### 4.1.6 Explanation Agent

The `ExplanationAgent` generates natural-language explanations:

- Uses LLM to synthesize verdict, reasoning, and evidence
- Produces clear, neutral explanations
- Includes evidence summaries
- Handles all three verdict types appropriately

### 4.2 Pipeline Implementation

The `FactCheckingPipeline` orchestrates the agentic loop:

```python
async def process(claim: Claim) -> FactCheckResult:
    # Stage 1: Normalization
    normalized_claim = self.claim_understanding.normalize_claim(claim)
    
    # Stage 2: Iterative retrieval-evaluation loop
    all_evidence = []
    for iteration in range(max_iterations):
        candidates = await self.retrieval.retrieve_evidence(normalized_claim)
        selected = self.evidence_selection.select(normalized_claim, candidates, k)
        all_evidence.extend(new_evidence)
        evaluation = await self.evidence_evaluation.evaluate(normalized_claim, all_evidence)
        if evaluation.sufficient:
            break
    
    # Stage 3: Verdict generation
    verdict = await self.reasoning_and_verdict.decide(normalized_claim, final_evidence)
    
    # Stage 4: Explanation generation
    explanation = await self.explanation.generate(normalized_claim, verdict, final_evidence)
    
    return FactCheckResult(claim, verdict, explanation, final_evidence)
```

**Key Features:**
- Iterative refinement until evidence sufficient or max iterations
- Early termination on sufficiency
- Deduplication of evidence across iterations
- Graceful handling of insufficient evidence

### 4.3 Interface Layer

#### 4.3.1 Command-Line Interface

The CLI (`factcheck_agent.cli`) provides:

- Simple claim input: `factcheck "claim text"`
- Source metadata: `factcheck "claim" --source "social_media"`
- Formatted output with verdict, confidence, explanation, and evidence
- Error handling and user-friendly messages

#### 4.3.2 Programmatic API

The system can be used programmatically:

```python
from factcheck_agent import FactCheckingPipeline, get_default_llm_client
from factcheck_agent.models import Claim

llm = get_default_llm_client()
pipeline = FactCheckingPipeline(llm=llm)
claim = Claim(id="c1", raw_text="Saudi Arabia is the largest oil producer")
result = await pipeline.process(claim)
```

#### 4.3.3 Jupyter Notebook Integration

Interactive notebooks provide:
- Step-by-step pipeline execution
- Claim detection from long texts
- Batch fact-checking of detected claims
- Visualization of intermediate results

---

## 5. Technical Details

### 5.1 Asynchronous Architecture

The system is built entirely on async/await patterns:

- **Rationale**: LLM API calls and web searches are I/O-bound operations
- **Benefits**: 
  - Non-blocking execution
  - Efficient resource utilization
  - Scalability for batch processing

- **Implementation**: All agent methods are async, pipeline uses `asyncio.run()` for execution

### 5.2 Error Handling and Resilience

The system implements multiple layers of error handling:

1. **Configuration Validation**: Checks for required API keys at initialization
2. **API Error Handling**: Graceful degradation when APIs fail
3. **LLM Response Parsing**: Robust JSON extraction with regex fallbacks
4. **Fallback Mechanisms**: Heuristic methods when LLMs unavailable
5. **User Feedback**: Clear error messages and warnings

### 5.3 Configuration Management

Configuration is managed through:

- Environment variables (`.env` file support)
- `Config` class with type-safe access
- Validation methods for required settings
- Sensible defaults for optional parameters

**Key Configuration:**
- LLM API keys (OpenAI, Anthropic)
- Search API keys (Google Custom Search)
- Pipeline parameters (max iterations, retrieval k)
- Confidence thresholds

### 5.4 Testing Strategy

Three offline suites, **89 checks total**, requiring no network access and no API
keys. Every check runs against scripted LLM and search doubles.

| suite | checks | scope |
|---|---|---|
| `tests/test_stage2_behaviour.py` | 43 | retrieval loop iteration, RRF scoring, full-text enrichment and fallback, leakage measurement, article orchestration, query budgets |
| `tests/test_stage3_verdict.py` | 30 | verdict instrumentation, threshold removal, parse-failure counting, decision logging |
| `tests/test_agents_smoke.py` | 16 | every agent constructs, is awaited correctly, and honours its contract |

Two testing practices were adopted after specific failures in this project:

* **Tests must be able to fail.** `test_agents_smoke.py` previously had no
  `__main__` block, so running it executed nothing and exited 0 while five of its
  six functions were broken against a long-changed API. A suite that reports green
  without running is worse than no suite.
* **Assertions must exercise the path they claim to.** Two checks were found
  passing trivially — one because an injected exception was swallowed by an
  agent's own `try/except` before reaching the code under test, another because a
  stubbed method meant the recording list stayed empty. Both were rewritten to
  depend on the behaviour being tested.

Known defects are pinned by test where they are deliberately retained — for
example, the verdict JSON parser cannot handle nested objects, and a test asserts
that it still cannot, so the defect cannot be silently "fixed" outside a measured
change.

## 6. Evaluation

All figures below are measured. Where a number is an estimate, a bound, or a
condition that does not correspond to a deployable configuration, it is labelled
as such. Raw results are in `results/`.

### 6.1 Method

Two benchmarks, chosen because no single dataset spans the pipeline: claim
detection requires source articles, while verification datasets begin from a
claim and work forward to evidence.

* **Stage 1 (claim detection)** — NewsScope, 80 in-domain and 60 out-of-domain
  articles. ROUGE-L and a calibrated BERTScore.
* **Stages 2–3 (retrieval, verdict)** — AVeriTeC dev, 136 of 150 claims scored.
  The fourth AVeriTeC label, `Conflicting Evidence/Cherrypicking`, has no
  equivalent in this system's three-label output and is **excluded from accuracy**
  and reported separately; scoring it would measure a representational gap rather
  than verdict quality.

Two practices are applied throughout, both a response to errors made earlier in
this project:

* **Paired comparison.** Conditions run over the same claims with the same
  evidence, varying one factor. Several apparent gains disappeared once retrieval
  volume was held constant.
* **A majority-class baseline is reported with every accuracy figure.** 71% of
  AVeriTeC dev claims are `Refuted`, so a constant predictor scores **0.706**.
  Macro-F1 is the primary metric under this imbalance.

### 6.2 Stage 1 — claim detection

Three prompt designs, single-variable ablation, ROUGE-L F1 at threshold 0.40:

| prompt | in-domain F1 | out-of-domain F1 | cross-domain drop |
|---|---|---|---|
| A (atomic, ≤20 claims) | 0.370 | 0.255 | −31% |
| **B (strict, 2–4 claims)** | **0.452** | 0.296 | −34% |
| C (journalist-style, 3–5) | 0.435 | **0.322** | **−26%** |

B is strongest in-domain; C degrades least and is best out-of-domain. A
over-generates: highest recall, lowest precision.

**A benchmark artifact worth stating.** Gold sets average 2.45 claims per
article, so precision is capped at `n_gold / n_predicted`. A prompt emitting ~6
claims cannot exceed ~0.41 precision however correct it is. Precision must not be
compared across prompts that differ in output volume.

**A measurement error found and corrected.** An initial BERTScore evaluation
reported F1 up to 0.838. It used raw (un-rescaled) BERTScore with a 0.85
threshold; measured, unrelated sentences score ~0.838 under that metric, so the
threshold sat inside the noise floor and the metric degenerated into a claim-count
ratio — recall was pinned near 1.0 regardless of content. Re-run with baseline
rescaling and an empirically calibrated threshold, BERTScore gives 0.531 for
prompt B, confirming ROUGE-L understates by roughly 0.05–0.08 F1 rather than the
0.3 originally implied. Full analysis: `results/claim_detection/BERTSCORE_ANALYSIS.md`.

### 6.3 Stage 2 — evidence retrieval

Measured on a closed AVeriTeC corpus (973 documents). **Closed-corpus results are
easier than live open-web search and are not comparable to it.**

* **Evidence depth beats breadth.** Replacing ~186-character search snippets with
  fetched article passages raised verdict accuracy **0.559 → 0.640** (McNemar p=0.043)
  and cut abstention 45.6% → 34.6%. Widening the candidate pool instead produced the
  *best retrieval recall of any condition* while making verdicts **worse**. This
  **replicates on live web evidence and more strongly there** (§6.5: +0.132 accuracy,
  p=0.041 at n=76), making it the largest measured improvement in the system.
* **Enrichment saturates at ~3 documents**; `FULL_TEXT_TOP_K=3` is validated
  rather than assumed.
* **Reachability.** 70.7% of real evidence URLs are fetchable, yielding ~17×
  more text than a snippet. Archived copies are *more* reliable than live URLs
  (80.7% vs 64.5%), so a Wayback fallback is used. The ~29% that fail retain their
  snippet, so enrichment can only add.
* **The retrieval loop was a no-op, then was repaired, then was vindicated live.**
  It originally passed an identical claim every round; URL dedup guaranteed round 2
  added nothing, after spending a full round of searches. Repaired — but at matched
  retrieval volume on the closed corpus it is statistically indistinguishable from a
  larger single pass (p=0.43). **That null result is an artifact of the setting**: a
  single 973-document index with round 1's URLs excluded leaves a reformulated query
  nowhere new to reach. Measured live (§6.5) the loop clearly works. The closed-corpus
  verdict against it should not be cited.

### 6.4 Stage 3 — reasoning and verdict

Measured under an oracle condition (gold evidence fed directly), which isolates
the verdict stage from retrieval quality.

| | accuracy | macro-F1 | abstention |
|---|---|---|---|
| always predict `REFUTES` (baseline) | **0.706** | 0.276 | — |
| this system, gold evidence | 0.728 | **0.624** | 27.9% |
| this system, retrieved evidence | 0.544 | 0.470 | 45.6% |

With perfect evidence the stage clears the majority baseline by **+0.022**. On
retrieved evidence it falls **below** it. It does beat the baseline decisively on
macro-F1, because it distributes across classes rather than collapsing to one.

**Over-abstention is the dominant failure: 33 of 37 errors (89%)** are
unwarranted `NOT_ENOUGH_INFO` on gold evidence, against a gold rate of ~6%.

**Three interventions were tested and all rejected:**

1. *Prompt symmetry* — equalising the `SUPPORTS`/`REFUTES` bars raised `SUPPORTS`
   recall (0.562 → 0.719) but cost more `REFUTES` than it gained; abstention
   unchanged (p=1.0).
2. *Reasoning synthesis* — deriving an evidence-to-verdict bridge before judging
   made every metric worse at twice the cost, converting 13 decisions into
   abstentions.
3. *Confidence thresholds* — the pre-existing rules fired **0 of 150** times and
   were removed. Removal is a clarity change, not an improvement.

**Why abstention resists prompting.** Supplying the fact-checker's own
justification raises accuracy to 0.904; controlling for the 50% of justifications
that state the verdict outright, the gain on non-leaking cases is **+0.132
(p=0.0039)**. But the model cannot *generate* that bridge from the same evidence —
attempting it scored −0.059. Together these indicate the justification helps
because it carries information **absent from the evidence**, not because it
reorganises what is present.

Consequently over-abstention is substantially a *reasonable* response to
inferentially incomplete evidence. The remaining levers are retrieval-side.

**Confidence is uninformative.** ECE 0.202, and the model emits only four values
`{0.8, 0.9, 0.95, 1.0}` with 132 of 136 predictions in a single reliability
bucket. Nothing downstream reads it.

### 6.5 Live open-web evaluation — the only non-proxy measurement

All figures above use one of two proxies: an AVeriTeC-derived closed corpus, or gold
answers supplied directly. Neither is what the deployed system retrieves. A live
evaluation was run against Google CSE, accumulated in daily batches within a
100-queries/day free tier: **41 claims, 148 searches, 35 scorable**
(`results/evidence_retrieval/RESULTS.md` §11; `results/verdict/FINDINGS.md` §L).
Queries and results are exported to `query_trail.json` because the run cannot be
reproduced without re-spending quota on a web whose contents change.

**Retrieval (Stage 2).** The iterative loop, which the closed corpus found
indistinguishable from a larger single pass, is vindicated live: **94% of second rounds
reach a domain the first round did not** (mean 3.09 new domains), replicated across
**four independent batches**, and round 2 retrieves fact-checking pages far less often
than round 1 (3 against 45), reaching government, primary and wire sources instead. The
binding constraint is not the loop but the sufficiency gate that triggers it: only
**39% of claims** get a second round at all (31–46% by batch).

**Verdict (Stage 3).** Scored on 76 claims using the majority-vote label over three
scoring runs per condition:

| condition | accuracy | macro-F1 | abstention |
|---|---|---|---|
| closed corpus | 0.592 | 0.552 | — |
| live snippets | 0.684 | 0.552 | 23.7% |
| majority-class baseline | **0.750** | — | — |
| oracle gold evidence | 0.763 | **0.649** | — |
| **live web + full text** | **0.816** | 0.607 | **10.5%** |

**Live evidence with full text beats the majority-class baseline** by +0.066, and the
margin survives the leakage control: on the 43 claims where no fact-checking page was
retrieved it scores **0.837 against their 0.744 baseline**. Full-text enrichment is
worth **+0.132 accuracy and +0.055 macro-F1** over snippets (paired McNemar p=0.041),
cutting abstention from 23.7% to 10.5%.

**The closed corpus understates live verdict quality**, now established over four
independent samples with a stable 3–6:1 discordant ratio (22:5, p=0.0015 at n=76).
Every closed-corpus figure in this document is therefore a lower bound.

**Live full-text retrieval reaches gold-evidence quality.** The oracle condition scores
0.763 against live full text's 0.816, and the paired test cannot separate them (14:10,
p=0.54). The system does not need the benchmark's annotated answers to reach
benchmark-level verdict quality — a stronger result than the retrieval-side pessimism in
§6.4 anticipated.

**What the gain cannot yet be attributed to.** Live full text is **9.7× longer** than a
snippet *and* differently sourced, so "more text" and "better text" are confounded.
Truncating full text to the oracle's ~246 characters would separate them; this is
untested and is the sharpest open question in the evaluation.

**Two methodological findings matter more than the numbers above.**

1. **Verdict scoring is not reproducible.** The verdict call runs at temperature 0.2
   and is not cached, so re-scoring identical claims on identical replayed evidence
   changes 2–4% of labels — enough to move a paired p-value across 0.05 (observed:
   0.012, 0.006, 0.065 on the same data). Measurement now uses `--repeat N` with a
   majority-vote label. Every Stage 3 figure recorded before 2026-10-08 is a single
   run carrying an unquantified ±0.02–0.04 band.
2. **This section's conclusions reversed twice before settling, and the reversals were
   caused by sample size, not by the system changing.** Live-vs-closed-corpus discordant
   pairs ran 6:0, 7:2, 13:4, 22:5 across n=13/35/55/76 — a stable 3–6:1 ratio — while the
   p-value crossed 0.05 twice (0.031, 0.180, 0.049, 0.0015). The n=35 reading declared
   enrichment not to replicate (p=1.000) and Stage 3 to beat no baseline; both were
   3:2-discordant-pair artifacts and are withdrawn. **The practice that eventually got
   this right was reporting the direction and declining to claim significance until the
   sample could support it.**

**The resulting research question** is no longer whether live retrieval suffices — it
reaches gold-evidence quality — but **what makes it suffice**. Full text beats snippets
by +0.132 while being 9.7× longer and differently sourced, so volume and quality are
confounded; truncating to the oracle's length would separate them. The secondary
question is the sufficiency gate in §6.5: the retrieval loop demonstrably improves
sources but fires on only 39% of claims, and nothing has measured the decision that
gates it.

### 6.6 Threats to validity

* Closed-corpus retrieval results are a lower bound and not comparable to live
  search. Live comparison (§6.5) leans the same way in all three samples collected
  (6:0, 7:2, 13:4 discordant), but the p-value has oscillated across 0.05 (0.031,
  0.180, 0.049), so the magnitude is unresolved.
* The live sample (n=89) is **somewhat easier than the benchmark average**: the closed
  corpus scores 0.592 on it against 0.544 on n=136, though the bullet below puts ±0.05
  on both. Cross-sample comparisons with the n=136 figures are indicative only.
* The **majority-class baseline drifts between samples** (0.692 → 0.800 → 0.782 → 0.750
  across the four batches) as the label mix changes. An accuracy from one sample must
  not be compared against a baseline from another; that error produced the n=13
  over-claim described above.
* **Stage 3 figures carry run-to-run variance** from an uncached temperature-0.2 verdict
  call. All §6.5 figures use a 3-run majority vote. Measured spread at n=76: 0.000 for
  the live conditions, 0.013 for the oracle, and **0.053 for the closed corpus**, which
  re-runs temperature-0.3 query generation and so varies in retrieval as well as
  verdict. **Any single-run closed-corpus figure carries ±0.05** — larger than most
  effects measured on that corpus. §6.4's figures are single runs and have not been
  repeated.
* The fact-check leakage detector is domain-based and misses fact-checking content
  hosted elsewhere, so the leakage-free subset is a lower bound on contamination.
* Live results are quota-bound and time-bound: Google CSE's free tier is 100
  queries/day and **the API sunsets 2027-01-01**, so this evaluation is not
  indefinitely reproducible as run. A provider abstraction (`search.py`) exists to
  allow substitution.
* Macro-F1 on the live sample is unstable: 5 `Supported` and 2 `Not Enough
  Evidence` claims mean one flipped claim moves it several points.
* The oracle condition supplies ~2.0 short question-answer pairs per claim; this
  is an upper bound on evidence quality for this benchmark, not a simulation of
  ideal live retrieval.
* n=136 for Stages 2–3. Effects below roughly 0.05 accuracy are not reliably
  detectable.
* Verdict figures use `gpt-4.1-mini`; an earlier record cites GPT-4o-mini, so
  cross-version comparisons are not model-controlled.
* `Conflicting Evidence/Cherrypicking` (9.3% of claims) is outside the system's
  label space and excluded.

## 7. Use Cases and Applications

### 7.1 Single Claim Fact-Checking

The system can verify individual claims:

```bash
factcheck "Saudi Arabia is the largest oil producer in the world."
```

**Output:**
- Verdict (SUPPORTS/REFUTES/NOT_ENOUGH_INFO)
- Confidence score
- Explanation
- Evidence sources with URLs

### 7.2 Batch Claim Processing

The system can process multiple claims from long texts:

1. **Claim Detection**: Extract check-worthy claims from articles/posts
2. **Batch Fact-Checking**: Process all detected claims
3. **Summary Report**: Aggregate results with importance scores

**Workflow:**
```
Long Text → Claim Detection → Multiple Claims → Batch Fact-Checking → Summary
```

### 7.3 Integration Use Cases

- **News Verification**: Verify claims in news articles
- **Social Media Monitoring**: Check viral claims
- **Research Support**: Verify factual claims in research
- **Educational Tools**: Teach critical thinking through fact-checking

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

Measured limitations first, since these are the ones evidence supports.

**Verdict quality is below a constant baseline on retrieved evidence.** 0.544
accuracy against 0.706 for a predictor that always answers `REFUTES` (§6.4). The
system beats that baseline on macro-F1 and with gold evidence, but not on the
configuration a user actually gets.

**Over-abstention dominates the error profile.** 89% of errors on gold evidence
are unwarranted `NOT_ENOUGH_INFO`. Three prompt-level interventions failed to
reduce it, and the evidence indicates it is substantially a reasonable response to
inferentially incomplete evidence rather than a defect that prompting can fix
(§6.4).

**The pipeline has no question-decomposition step.** AVeriTeC's task design runs
`claim → questions → answers → verdict`; this system runs
`claim → search → snippets → verdict`. The measured evidence gap points at this
omission. Untested, and the largest unexplored change.

**Explanations are ungrounded.** `used_evidence_ids` records the snippets *shown*
to the model, not those it used — the prompt contains no evidence IDs, so the
model cannot cite. The explanation stage has no faithfulness check and no
verification that it agrees with the verdict label. It has never been evaluated.

**Confidence carries almost no information.** ECE 0.202, four distinct values,
132 of 136 predictions in one reliability bucket (§6.4).

**The search backend has a hard deadline.** Google's Custom Search JSON API is
closed to new customers and sunsets 2027-01-01. A provider-agnostic `SearchClient`
interface exists; no replacement implementation has been written.

**Retrieval cost scales multiplicatively.** Cost is
(claims per article) × (queries per claim) × (retrieval rounds). On the free
search tier this bounds throughput to a few articles per day, which constrains
end-to-end evaluation more than it constrains the system.

**Fact-check leakage is a standing validity risk.** Retrieval deliberately issues
`"{claim} debunked"`-style queries, which are close to optimal for surfacing
fact-checking articles. This is now measured and filterable, but a system that
retrieves a fact-checker's conclusion is looking up the answer rather than
verifying the claim.

Longer-standing limitations, unmeasured:

1. **Evidence source diversity** — web search only; no academic, government or
   fact-checking databases.
2. **Temporal reasoning** — limited handling of time-sensitive claims.
3. **Multilingual support** — English-focused.
4. **Source credibility** — no weighting by reliability.
5. **Label space** — three labels; real fact-checking (and AVeriTeC) also uses a
   conflicting/cherrypicking category, which this system cannot express.

### 8.2 Future Enhancements

**Short-Term:**
1. **Additional Evidence Sources**: 
   - Fact-checking databases (Snopes, PolitiFact APIs)
   - Academic databases (PubMed, arXiv)
   - Government databases

2. **Improved Query Generation**:
   - Entity-aware query expansion
   - Temporal query modification
   - Multi-lingual query support

3. **Evidence Quality Metrics**:
   - Source credibility scoring
   - Recency weighting
   - Agreement/disagreement detection

**Medium-Term:**
1. **Advanced Reasoning**:
   - Multi-hop reasoning
   - Contradiction resolution
   - Uncertainty quantification

2. **Performance Optimization**:
   - Caching of common claims
   - Parallel evidence retrieval
   - Batch processing optimizations

3. **User Interface**:
   - Web interface
   - API endpoints
   - Real-time fact-checking

**Long-Term:**
1. **Multimodal Fact-Checking**:
   - Image verification
   - Video fact-checking
   - Audio verification

2. **Collaborative Fact-Checking**:
   - Human-in-the-loop verification
   - Community fact-checking
   - Expert review integration

3. **Explainable AI Enhancements**:
   - Evidence attribution
   - Reasoning chains
   - Confidence calibration

---

## 9. Conclusion

This document has presented a comprehensive overview of an agentic automated fact-checking system, from its initial conception through multiple development phases to its current implementation. The system demonstrates:

1. **Modular Architecture**: Clean separation of concerns with specialized agents
2. **Iterative Refinement**: Agentic loop for evidence collection and evaluation
3. **Practical Implementation**: Real-world integration with LLMs and web search
4. **Continuous Improvement**: Responsive to performance analysis and user feedback

The system addresses a critical need in the modern information ecosystem by providing automated, scalable fact-checking capabilities. While current limitations exist, the architecture provides a solid foundation for future enhancements and extensions.

The development process has been iterative and responsive, with improvements driven by empirical evaluation and user needs. The system's evolution from basic placeholders to a production-ready implementation demonstrates the value of incremental development and continuous refinement.

---

## 10. References and Related Work

### 10.1 Related Systems

- **FEVER** (Fact Extraction and VERification): Dataset and baseline for fact-checking
- **ClaimBuster**: Real-time claim detection and verification
- **Full Fact**: Automated fact-checking for UK politics
- **PolitiFact**: Human-in-the-loop fact-checking with automation support

### 10.2 Technical Foundations

- **Anthropic's "Building Effective Agents"**: Framework for agentic AI systems
- **Retrieval-Augmented Generation (RAG)**: Evidence retrieval and LLM integration
- **Pydantic**: Data validation and serialization
- **Async/Await Patterns**: Asynchronous programming in Python

### 10.3 Evaluation Metrics

- **Verdict Accuracy**: Correctness of SUPPORTS/REFUTES/NOT_ENOUGH_INFO
- **Confidence Calibration**: Alignment between confidence scores and accuracy
- **Evidence Quality**: Relevance and credibility of retrieved evidence
- **Response Time**: Latency of fact-checking pipeline

---

## Appendix A: System Requirements

### A.1 Software Dependencies

- Python 3.9+
- Pydantic 2.7.0+
- python-dotenv 1.0.0+
- httpx 0.24.0+ (optional, for web search)
- openai 1.0.0+ (optional, for OpenAI integration)

### A.2 API Requirements

- LLM API key (OpenAI or Anthropic)
- Google Custom Search API key and Engine ID (optional, for web search)

### A.3 Hardware Requirements

- Minimal: Any system capable of running Python 3.9
- Recommended: Systems with network access for API calls

---

## Appendix B: Code Structure

```
factcheck_agent/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── models.py                # Pydantic data models
├── llm_client.py           # LLM abstraction layer
├── pipeline.py             # Main fact-checking pipeline
├── cli.py                  # Command-line interface
└── agents/
    ├── __init__.py
    ├── claim_understanding.py    # Claim normalization and detection
    ├── retrieval.py              # Evidence retrieval
    ├── evidence_selection.py     # Evidence ranking and selection
    ├── evidence_evaluation.py    # Evidence sufficiency evaluation
    ├── reasoning_and_verdict.py  # Verdict generation
    └── explanation.py            # Explanation generation
```

---

**Document Version:** 1.0
**Last Updated:** 2025-01-XX
**Author:** Fact-Check Agent Development Team
**License:** MIT

