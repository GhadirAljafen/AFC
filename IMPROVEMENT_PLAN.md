# Fact-Checking System — Improvement Plan (v2)

**Version 2, 2026-09-26.** Supersedes `docs/archive/IMPROVEMENT_PLAN_v1_2025.md`,
which is kept unchanged as a historical record.

## Why this was rewritten

Version 1 diagnosed a verdict-distribution problem, proposed four changes, and
recorded expected outcomes. All four were implemented. **None were ever measured.**
When they finally were — against the AVeriTeC benchmark, 136 scored claims — the
central prediction turned out to be wrong in both direction and mechanism:

| v1 prediction | measured |
|---|---|
| `NOT_ENOUGH_INFO` falls 70% → 30–40% | **45.6%** on retrieved evidence, **27.9%** even on gold evidence |
| `REFUTES` rises 5% → 30–40% | **44%** predicted against **71%** gold — the system *under*-refutes |
| Confidence thresholds prevent over-confident verdicts | **Fired 0 of 150 times.** Unreachable, now removed |

The lesson is recorded here deliberately: v1's four priorities were marked
"✅ Complete" and "Ready for Testing" in `CHANGES_IMPLEMENTED.md`, and the testing
step never happened. Predictions then propagated into
`ACADEMIC_DOCUMENTATION.md` §6.3 as "Expected Outcomes" and were read thereafter
as results.

**Every claim in this version cites a measurement or is labelled untested.**

---

## 1. Status of the v1 priorities

### 1.1 Verdict prompt with explicit refutation guidance — SHIPPED, effect unclear
`reasoning_and_verdict.py`. The prompt makes `REFUTES` reachable on "ANY" evidence
and a "MUST", while gating `SUPPORTS` behind "Only … clearly confirms".

Measured: all 14 missed gold-`SUPPORTS` claims are the model declining to confirm,
with no post-processing involved (`FINDINGS.md` §2). A symmetric rewrite was tested
and **rejected** — it raised `SUPPORTS` recall (0.562 → 0.719 oracle) but cost more
`REFUTES` than it gained, leaving abstention unchanged (p=1.0, p=0.45).

**Status: retained, not because it is right but because no tested alternative is
better.** The asymmetry appears to be load-bearing for `REFUTES` recall.

### 1.2 Refutation-focused search queries — SHIPPED, carries a validity risk
`retrieval.py` appends `"{claim} false"` / `"{claim} debunked"`. This is close to
an optimal query for surfacing fact-checking sites, which risks the system
reproducing a fact-checker's conclusion rather than verifying the claim.

Mitigation shipped: fact-check leakage is now **measured** on every run, and
`EXCLUDE_DOMAINS` can filter it. Measured leakage did **not** increase when the
retrieval loop was made to iterate (p=0.157).

**Status: retained, now instrumented.**

### 1.3 Evidence selection prioritising contradictions — SHIPPED, recalibrated
`evidence_selection.py`. The boost was `+0.3 + 0.1 × matches` on a base that was
**always the constant 0.5**, because retrieval never set a score. Now that
retrieval produces real relevance scores spanning 0–1, an unbounded boost would
swamp them — 15 keyword matches would have added +1.8.

**Status: retained, capped and saturating (`REFUTATION_BOOST = 0.25`).**

### 1.4 Confidence thresholds — REMOVED, never had any effect
Demoted a `SUPPORTS` below 0.5 to `NOT_ENOUGH_INFO`; kept a `REFUTES` below 0.4.

Measured: **0 firings in 150 decisions.** The model emits confidences only in
`{0.8, 0.9, 0.95, 1.0}`, so both thresholds were unreachable. The rule never
affected a single verdict.

**Status: removed.** Replaced by a warning log if a low confidence ever appears.
This is a clarity change and is **not** an improvement.

---

## 2. What measurement established instead

Full evidence: `results/evidence_retrieval/RESULTS.md`, `results/verdict/FINDINGS.md`.

1. **Evidence depth beats evidence breadth.** Full-text enrichment raised verdict
   accuracy 0.559 → 0.640 (McNemar p=0.043). Widening the candidate pool improved
   retrieval recall to the best of any condition and made verdicts *worse*.
   `FULL_TEXT_TOP_K=3` is validated; beyond ~3 documents the gain saturates.
2. **The retrieval loop was a no-op that wasted quota.** It re-sent an identical
   claim each round, so URL dedup guaranteed round 2 added nothing after spending
   a full round of searches. Now repaired — though at matched retrieval volume it
   is statistically indistinguishable from a larger single pass.
3. **The verdict stage is not the tractable bottleneck.** It over-abstains at
   27.9% even on gold evidence, and three prompt-level interventions failed to
   move it. Supplying the fact-checker's own justification is worth +0.132
   accuracy (p=0.0039 on non-leaking cases), but the model **cannot generate that
   bridge itself** — attempting to made everything worse at 2× cost. The
   justification helps because it carries information *absent from the evidence*.
4. **Confidence is uninformative.** ECE 0.202; 132 of 136 predictions fall in one
   reliability bucket. Nothing downstream reads it.

---

## 3. Current priorities

### P1 — Replace the search backend (HARD DEADLINE)
Google's Custom Search JSON API is **closed to new customers and sunsets
2027-01-01**. The `SearchClient` interface, disk cache and offline doubles are
built and tested; what remains is choosing a provider (Brave / Serper / Tavily)
and writing one implementation. **This is the only item with an external clock,
and the pipeline stops retrieving when it passes.**

### P2 — Question decomposition (retrieval-side)
AVeriTeC's task design is `claim → questions → answers → verdict`; this pipeline
goes `claim → search → snippets → verdict`, skipping decomposition entirely. The
measured evidence gap (§2.3) points here. **Untested** — the largest unexplored
lever, and explicitly retrieval-side, not verdict-side.

### P3 — Citation grounding (Stage 4 prerequisite)
`used_evidence_ids` currently holds the snippets *shown* to the model, not those
it used; the prompt contains no IDs, so the model cannot cite. Explanations cannot
be grounded until this is real. Requires a prompt change and its own paired
measurement.

### P4 — Explanation faithfulness (Stage 4)
`explanation.py` narrates an already-fixed verdict with no citation grounding, no
faithfulness check, and no verification that it agrees with the verdict label.
Never evaluated.

### P5 — Test the retrieval loop live
The closed corpus cannot expose sources a reformulated query would newly reach,
which is the loop's entire rationale. Until then, whether it earns its cost is
genuinely unknown.

---

## 4. Practices adopted after v1

Recorded because v1's failure mode was methodological, not technical:

- **Measure before changing.** Stage 3 began with diagnosis and refuted its own
  pre-registered hypothesis before any code was touched.
- **Paired comparisons.** Same claims, same evidence, one variable. Several early
  findings evaporated once volume was controlled for.
- **Always report the majority-class baseline.** 71% of AVeriTeC dev is `Refuted`,
  so a constant predictor scores 0.706 — above this system's verdict accuracy on
  retrieved evidence.
- **Prefer macro-F1 under class imbalance**, and report both.
- **Withdraw findings in place.** Three claims have been retracted with their
  reasoning left visible rather than edited away.
- **Label untested changes as untested**, and never report the removal of dead
  code as an improvement.

---

**Status:** measurement-driven. Stages 1–3 complete; Stage 4 not started.
**Evidence:** `results/claim_detection/`, `results/evidence_retrieval/`,
`results/verdict/`.
