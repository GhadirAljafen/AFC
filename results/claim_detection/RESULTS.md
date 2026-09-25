# Claim Detection — Prompt Ablation Results (Clean Re-run)

**Date:** 2026-06-09
**Task:** Check-worthy claim detection on the NewsScope benchmark
**Script:** [`evaluate.py`](evaluate.py) · prompts in [`prompts.py`](prompts.py)

## Summary

A clean, single-variable prompt ablation comparing three detection prompts (A, B, C).
All three arms now run **identical extraction logic** (the base
`ClaimUnderstandingAgent.detect_claims`) and differ **only** by the prompt
template. Earlier runs were confounded: arm A used the base agent while arms
B/C used a separate `PromptAgent` override that (a) silently returned `[]` on
any error, (b) constructed `DetectedClaim` with wrong field names
(`char_start`/`importance_score`) so `importance` was always dropped, and
(c) used different `min_importance` (A=0.5, B/C=0.0) and `max_tokens`
(A=1000, B/C=500). The refactor removed all three confounds.

## Setup

| Item | Value |
|------|-------|
| Metric | **ROUGE-L F-measure**, greedy 1:1 matching, threshold ≥ 0.40, Porter stemming |
| Model | `gpt-4.1-mini` (OpenAI), temperature 0.3, max_tokens 1000 (all arms) |
| Filtering | `min_importance = 0.0` (all arms — prompt governs claim count) |
| Normalization | `normalize_detected = False` |
| Input per article | `headline + key_points` (dataset article URLs were empty) |
| Splits | `test_indomain` (80 articles), `test_oos` (60 articles) |
| Detection failures | 0 / 140 across all arms |

> **Caveat — model:** this run used `gpt-4.1-mini` (verified: code default,
> no override). The earlier record in `prompts.py` noted `GPT-4o-mini`, so the
> original-vs-clean comparison below is *not* strictly model-controlled.
> The near-identical numbers nonetheless indicate the prompt — not the
> confounds — was the dominant variable.

> **Caveat — metric:** ROUGE-L is lexical and under-credits correct
> paraphrases (a semantically-correct claim worded differently than gold can
> score < 0.40 and count as a miss). This depresses recall across all arms
> roughly equally, so relative rankings are sound; absolute values are
> conservative. BERTScore could not be used (PyTorch 2.2.2 / NumPy 2.x conflict).

## Results

### In-domain (`test_indomain`, n = 80)

| Arm | Precision | Recall | F1 |
|-----|-----------|--------|-----|
| A   | 0.277 | 0.602 | 0.370 |
| **B** | 0.408 | 0.525 | **0.452** |
| C   | 0.367 | 0.552 | 0.435 |

### Out-of-domain (`test_oos`, n = 60)

| Arm | Precision | Recall | F1 |
|-----|-----------|--------|-----|
| A   | 0.192 | 0.400 | 0.255 |
| B   | 0.254 | 0.358 | 0.296 |
| **C** | 0.268 | 0.408 | **0.322** |

### Clean vs. original (F1)

| Arm | In-domain (orig → clean) | OOS (orig → clean) |
|-----|--------------------------|--------------------|
| A | 0.386 → 0.370 | 0.254 → 0.255 |
| B | 0.453 → 0.452 | 0.296 → 0.296 |
| C | 0.434 → 0.435 | *(not run)* → 0.322 |

Removing the confounds shifted every in-domain F1 by ≤ 0.016 (most by ≤ 0.001).

## Findings

1. **The original conclusions hold under a clean design.** The confounds
   (importance filter, max_tokens, the field-name bug) moved results
   negligibly, so the prompt was genuinely the driving variable.

2. **Prompt B is best in-domain** (F1 0.452), with C close behind (0.435).

3. **Prompt C is the most robust across domains** — a refinement of the
   original "A most stable" note, which was written without C and under the
   confound. C wins OOS outright (F1 0.322 vs B 0.296) and degrades least:

   | Arm | In-domain → OOS F1 drop |
   |-----|-------------------------|
   | A | −31% |
   | B | −34% |
   | **C** | **−26%** |

4. **Prompt A over-generates.** Up to ~20 claims per article gives it the
   highest recall (0.602 in-domain) but the lowest precision in both domains —
   many false positives.

**Takeaway:** B for in-domain maximization; **C for cross-domain robustness**
(best OOS F1 and smallest domain drop).

## Prompt characteristics

| Arm | Instruction | Target count |
|-----|-------------|--------------|
| A | Aggressive atomic splitting, ≤ 20 claims | high (~6 avg) |
| B | Strict: only 2–4 most significant claims | low (~3.5 avg) |
| C | Journalist-style, 3–5 self-contained claims, combine related facts | medium (~4.5 avg) |

## Files

- `metrics_test_indomain.json` — clean in-domain run (this report)
- `metrics_test_oos.json` — clean OOS run (this report)
- `metrics_test_indomain_n80_original.json` — preserved original 80-article numbers
- `evaluate.py`, `prompts.py` — eval harness and prompt templates
