# BERTScore Evaluation — Diagnostic and Corrected Methodology

**Date:** 2026-09-20
**Status:** ✅ Diagnostic complete and **corrected evaluation run** (see §6).
**Do not cite the BERTScore numbers from the Colab notebook** (see §2) — use §6.

---

## 1. What was run

A BERTScore evaluation of claim detection was run in Google Colab
(`AFC - BERTScore experiment.ipynb`) on the NewsScope benchmark, comparing
prompts A and B. It reported:

| Split | Arm | Precision | Recall | F1 |
|-------|-----|-----------|--------|-----|
| test_indomain (80) | A | 0.510 | 0.990 | 0.650 |
| test_indomain (80) | B | 0.750 | 0.979 | 0.838 |
| test_oos (60) | A | 0.552 | 0.983 | 0.693 |
| test_oos (60) | B | 0.717 | 1.000 | 0.829 |

Its stated conclusion was that "ROUGE-L underestimates system performance …
Prompt B achieves F1=0.838, indicating near-complete semantic coverage."

**That conclusion is not supported.** The numbers are a measurement artifact.

---

## 2. Why the result is invalid

### 2.1 The metric degenerated into a count ratio

The matcher was greedy 1:1 (each prediction claims at most one gold, then
`break`). When the threshold admits essentially every candidate pair,
`tp = min(n_predictions, n_gold)`, and the metric stops measuring semantics —
it measures only **how many claims a prompt emits**.

The NewsScope gold sets are very small (measured directly from the dataset):

| Split | Gold claims per article |
|-------|-------------------------|
| test_indomain | mean **2.45** (min 2, max 3) |
| test_oos | **exactly 2.00** for every article |

Predictions average ~6.0 claims (arm A) and ~3.5 (arm B). Computing what the
scores *would* be if every pair matched:

| | Predicted under "match everything" | Actually observed |
|---|---|---|
| in-domain A | P=0.408, R=**1.000**, F1=0.580 | P=0.510, R=**0.990**, F1=0.650 |
| in-domain B | P=0.700, R=**1.000**, F1=0.824 | P=0.750, R=**0.979**, F1=**0.838** |
| OOS A | P=0.333, R=**1.000**, F1=0.500 | P=0.552, R=**0.983**, F1=0.693 |
| OOS B | P=0.571, R=**1.000**, F1=0.727 | P=0.717, R=**1.000**, F1=0.829 |

Recall predictions are essentially exact, and in-domain B's F1 matches to within
0.014. **Prompt B "won" only because it emits fewer claims**, shrinking its
precision denominator — not because its claims were better.

Recall of **exactly 1.000** on the harder out-of-domain split is the clearest
symptom: every gold claim found a match.

### 2.2 Root cause — raw BERTScore with an uncalibrated cutoff

The notebook used `bert_score(..., lang="en")` with **no baseline rescaling** and
a hard cutoff of `0.85`.

Measured directly (`bertscore_threshold_demo.py`, roberta-large):

| Pair type | Raw F1 | Rescaled F1 |
|-----------|--------|-------------|
| Unrelated (different topics) | 0.835, 0.838, 0.840 → **mean 0.838** | 0.022, 0.042, 0.052 → **mean 0.039** |
| True paraphrase | 0.920, 0.936 → **mean 0.928** | 0.527, 0.623 → **mean 0.575** |
| **Separation** | **0.090** | **0.536** (≈ 5.9× wider) |

**Important nuance.** Maximally unrelated sentences score 0.835–0.840, which is
*just below* 0.85 — so the cutoff was not admitting literal noise. The problem is
that the entire usable scale spans only **0.838 → 0.936**, and 0.85 sits just
**0.010** above the unrelated floor.

Critically, the evaluation never compares unrelated sentences. It compares
predictions against gold claims **from the same article**, which share topic,
named entities, dates and numbers. Such pairs score well above 0.85 *even when
they state different facts*. That is the actual failure mode, and the observed
recall of 0.979–1.000 confirms it empirically.

### 2.3 Secondary problems in the notebook

1. **The ablation was re-confounded.** `PromptBAgent` again overrode the whole
   `detect_claims`, so arm A (base agent: `min_importance=0.5`, `max_tokens=1000`,
   nltk sentence splitting, full error handling) was compared against arm B
   (override: `min_importance=0.0`, `max_tokens=500`, no splitting). This is the
   same confound removed earlier in `evaluate.py`. *(The `DetectedClaim` field
   names were correctly fixed, however.)*
2. **Arm C was omitted** — only A and B were run, though C is the most robust
   arm cross-domain under ROUGE-L.
3. **Stale ROUGE-L baseline** — compared against 0.385/0.454 (the old confounded
   run) rather than the clean 0.370/0.452.
4. **Results were hand-typed** into the saved JSON rather than computed.
5. **Efficiency** — `bert_score()` was called once per pair inside a double loop
   (thousands of separate forward passes instead of one batched call).
6. **Skipped articles were silently dropped**, so the denominator was unknown.

---

## 3. A structural caveat about this benchmark

With only **2–2.45 gold claims per article**, precision is structurally capped
for any prompt that emits more claims than that. Arm A, emitting ~6 claims,
**cannot exceed ~0.41 precision even if every claim is perfect.**

This is a property of the benchmark, not of the prompt, and should be stated
explicitly wherever precision is compared across prompts that differ in output
volume.

---

## 4. Environment: why BERTScore appeared "unavailable"

Earlier notes recorded *"BERTScore unavailable due to PyTorch 2.2.2 / NumPy 2.x
conflict."* That diagnosis was wrong.

| Component | Local version | Verdict |
|-----------|---------------|---------|
| numpy | 1.26.4 | fine — **not** 2.x |
| torch | 2.2.2 | fine, and **cannot be upgraded**: this is an Intel Mac (x86_64) and PyTorch's last macOS x86_64 wheel is 2.2.2 |
| transformers | 5.8.0 | ❌ **the actual blocker** — v5 requires torch ≥ 2.4, so it disabled torch (`is_torch_available() == False`) |

**Fix:** downgrade transformers rather than upgrade torch.

```bash
pip install "transformers<5"      # resolved to 4.57.6 -> is_torch_available: True
```

Nothing else in this project depends on transformers, so this is low-risk.
BERTScore now runs locally; Colab is not required.

---

## 5. Corrected methodology

`evaluate_bertscore.py` replaces the notebook approach:

1. **`rescale_with_baseline=True`** — unrelated pairs sit near 0 instead of 0.85,
   widening the usable margin ~5.9×.
2. **Empirical threshold calibration** — a null distribution is built from
   ~1000 (prediction, gold) pairs drawn from *different* articles, and the
   threshold is set at its 95th percentile (≈5% false-accept rate on unrelated
   pairs). The threshold and null statistics are reported and saved.
3. **Threshold sensitivity sweep** — metrics are reported across a range of
   cutoffs from the same cached pair scores, so conclusions can be shown to hold
   (or not) independently of one arbitrary threshold.
4. **Single-variable ablation** — every arm uses the base agent's `detect_claims`
   and overrides only the prompt, with shared `min_importance=0.0`,
   `normalize_detected=False`, and identical `max_tokens`. **All three prompts.**
5. **One batched `bert_score` call** for all pairs.
6. **Reports `scored_articles`, `detection_failures`, `skipped_articles`**, and
   emits an automatic warning if recall still exceeds 0.95.

### Why calibration uses *hard* nulls

Calibrating on cross-article ("easy") nulls was tried first and **failed**: it
produced a threshold of 0.205 and recall pinned at **1.000**, with recall
unchanged for every cutoff from 0.05 to 0.25 — the same count-ratio degeneration
described in §2. Easy nulls only establish the floor for unrelated *topics*,
whereas the matcher's real task is separating different facts *within* an
article.

The hard null set — two distinct gold claims from the same article — needs no
extra labelling, since the benchmark already asserts they are separate claims.
It is substantially stricter in practice:

| Split | easy-null p95 | hard-null p95 | calibrated threshold |
|-------|---------------|---------------|----------------------|
| test_indomain | 0.202 | **0.427** | 0.4273 (n=152) |
| test_oos | 0.232 | **0.538** | 0.5384 (n=60) |

### Sanity check

**Recall must land clearly below 1.0.** After hard-null calibration it did
(0.31–0.67), and the automatic warning did not fire.

---

## 6. Results (corrected evaluation)

Run 2026-09-20. All articles scored, **0 detection failures, 0 skipped**.

### 6.1 Headline numbers (at each split's calibrated threshold)

**test_indomain (n=80), threshold 0.427**

| Arm | Precision | Recall | F1 |
|-----|-----------|--------|-----|
| A | 0.317 | 0.671 | 0.419 |
| **B** | 0.473 | 0.627 | **0.531** |
| C | 0.409 | 0.625 | 0.488 |

**test_oos (n=60), threshold 0.538**

| Arm | Precision | Recall | F1 |
|-----|-----------|--------|-----|
| A | 0.163 | 0.342 | 0.217 |
| B | 0.219 | 0.308 | 0.255 |
| **C** | 0.226 | 0.342 | **0.272** |

Recall of 0.31–0.67 (not ~1.0) confirms the metric is measuring semantic
match, not claim counts.

### 6.2 Does ROUGE-L underestimate? Yes — but modestly

In-domain, same systems, both metrics:

| Arm | ROUGE-L F1 | BERTScore F1 | Δ |
|-----|-----------|--------------|-----|
| A | 0.370 | 0.419 | +0.049 |
| B | 0.452 | 0.531 | +0.079 |
| C | 0.435 | 0.488 | +0.053 |

So the notebook's *direction* was right — ROUGE-L does under-credit paraphrases
by roughly **0.05–0.08 F1** — but its magnitude (B = 0.838) was an artifact.
The true value is **0.531**.

### 6.3 Ranking stability across thresholds

| Split | Ranking | Stability |
|-------|---------|-----------|
| in-domain | **B > C > A** | holds at **all 10** thresholds (0.05–0.60) |
| OOS | B > C > A | holds at **8 of 10**; C overtakes B only at 0.538 and 0.600 |

**A is the weakest arm everywhere** — both metrics, both splits, every
threshold. It over-generates (~6 claims against 2–2.45 gold), so precision
collapses. This is the single most robust finding.

### 6.4 ⚠️ One ROUGE-L conclusion does NOT survive

The ROUGE-L write-up concluded *"Prompt C is most robust cross-domain."* That
splits into two separate claims, which the BERTScore data treats differently:

1. **"B degrades most cross-domain"** — ✅ **CONFIRMED.** At every common
   threshold, B loses the most going in-domain → OOS:

   | threshold | A drop | B drop | C drop |
   |-----------|--------|--------|--------|
   | 0.20 | +0.7% | −6.0% | −1.1% |
   | 0.25 | −2.1% | −6.3% | −4.7% |
   | 0.30 | −4.1% | −9.7% | −4.9% |
   | 0.40 | −5.0% | **−12.8%** | −8.3% |
   | 0.50 | −21.3% | **−25.6%** | −20.5% |

2. **"C achieves the best OOS F1"** — ❌ **NOT ROBUST.** C leads OOS only at the
   two strictest cutoffs, by 0.017 and 0.002 F1. At the other eight thresholds
   **B is higher**. Worse, C's win occurs exactly at the *least statistically
   reliable* point (see §6.5).

**Recommended wording:** B is the strongest prompt in-domain and degrades most
out-of-domain; C is more stable but the two are statistically indistinguishable
on OOS. Do not claim C is the best OOS prompt.

### 6.5 Caveats that must accompany these numbers

1. **The two splits use different thresholds** (0.427 vs 0.538), because
   calibration is per-split. **Absolute F1 is therefore not comparable across
   splits.** Use the common-threshold table in §6.4 for domain comparison.
2. **The OOS threshold is poorly estimated.** With only 2 gold claims per
   article, OOS yields just **60** hard-null pairs, so its p95 is roughly the
   57th of 60 values — high variance. The jump from p90 = 0.394 to p95 = 0.538
   indicates a heavy tail. The OOS calibrated row is the least trustworthy in
   this report, and it is precisely where C "wins".
3. **The precision ceiling from §3 still applies** and explains arm A entirely.

## Files

- `evaluate_bertscore.py` — corrected, calibrated BERTScore evaluation
- `bertscore_threshold_demo.py` / `.json` — the raw-vs-rescaled evidence in §2.2
- `RESULTS.md` — the clean ROUGE-L ablation (currently the citable result)
