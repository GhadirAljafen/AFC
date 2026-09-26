> **ARCHIVED — superseded 2026-09-26.**
>
> This is the original IMPROVEMENT_PLAN.md, kept unchanged as a historical record. Several
> of its claims were later **contradicted by measurement**; see
> `IMPROVEMENT_PLAN.md` for the current version and
> `results/verdict/FINDINGS.md` / `results/evidence_retrieval/RESULTS.md` for the
> evidence. Do not cite figures from this file.

# Fact-Checking System Improvement Plan

## Problem Analysis

The system is currently getting mostly `NOT_ENOUGH_INFO` verdicts and rarely `REFUTES`, even when evidence contradicts claims.

### Root Causes Identified

1. **Prompt Design Issue in ReasoningAndVerdictAgent**
   - The prompt doesn't explicitly guide the LLM to look for refuting evidence
   - It asks to "analyze the claim against evidence" but doesn't emphasize contradictions
   - LLMs tend to be conservative and default to NOT_ENOUGH_INFO when uncertain

2. **Evidence Retrieval Scope**
   - Google Search returns broad, topic-related results that may not directly address the claim
   - No explicit search for refuting evidence (e.g., "claim X is false" or "claim X debunked")
   - Results are often tangential rather than directly contradictory

3. **LLM Conservatism**
   - Models default to NOT_ENOUGH_INFO when uncertain
   - Especially if evidence doesn't clearly contradict the claim

4. **Evidence Quality vs Quantity**
   - Limiting to top 10 snippets may miss refuting evidence
   - No explicit ranking for contradictory evidence

---

## Recommended Solutions

### Priority 1: High Impact, Quick to Implement

#### 1.1 Improve Verdict Prompt (HIGH PRIORITY)

**File:** `src/factcheck_agent/agents/reasoning_and_verdict.py`

**Expected Impact:** Should significantly increase REFUTES verdicts when evidence contradicts claims.

---

#### 1.2 Add Explicit Refutation Search Queries (HIGH PRIORITY)

**File:** `src/factcheck_agent/agents/retrieval.py`

**Expected Impact:** Will retrieve more refuting evidence, leading to more REFUTES verdicts.

---

### Priority 2: Medium Impact, Moderate Effort

#### 2.1 Improve Evidence Selection to Prioritize Contradictions

**File:** `src/factcheck_agent/agents/evidence_selection.py`

---

#### 2.2 Add Confidence Thresholds

**File:** `src/factcheck_agent/agents/reasoning_and_verdict.py`

---

## Implementation Order

1. **Priority 1.1** (Improve Verdict Prompt) - Quickest, highest impact
2. **Priority 1.2** (Refutation Search Queries) - Better evidence retrieval
3. **Priority 2.1** (Evidence Selection) - Better evidence quality
4. **Priority 2.2** (Confidence Thresholds) - Fine-tuning

---

## Expected Outcomes

After implementing Priority 1 changes:
- **REFUTES verdicts:** Should increase from ~5% to ~30-40% for false claims
- **SUPPORTS verdicts:** Should remain similar (~20-30% for true claims)
- **NOT_ENOUGH_INFO:** Should decrease from ~70% to ~30-40% overall

---

## Files to Modify

1. `src/factcheck_agent/agents/reasoning_and_verdict.py` - Prompt improvement
2. `src/factcheck_agent/agents/retrieval.py` - Refutation query generation
3. `src/factcheck_agent/agents/evidence_selection.py` - Prioritize contradictory evidence

---

**Created:** 2025-01-XX
**Status:** Implementation Phase

