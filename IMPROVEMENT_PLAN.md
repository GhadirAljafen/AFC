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

### ~~P5 — Test the retrieval loop live~~ — **done, and it reframed the priorities**
The closed corpus could not expose sources a reformulated query would newly reach,
which is the loop's entire rationale. Measured live on 41 claims
(`results/evidence_retrieval/RESULTS.md` §11): the loop **does** reach new domains —
**94% of second rounds add at least one, replicated across four batches** (n=89) — and
it retrieves fact-checking pages far less often than round 1 (3 against 45).
**Keep it enabled.**

Two consequences, both of which outrank the items above:

### P5a — Raise the second-round trigger rate *(new top retrieval priority)*
The loop fires on only **39% of claims** (31–46% by batch; 33, 31, 46, 46 across the
four). When it fires it reaches better sources 94% of the time. So the binding constraint is not the loop
but the **evidence evaluator's sufficiency gate**, which decides whether round 1 was
enough — and nothing has ever measured that gate. This is the cheapest remaining
retrieval lever and needs no new benchmark.

### P5b — Full-text enrichment on live evidence is the biggest measured win
`results/verdict/FINDINGS.md` §L at n=76, majority vote over 3 scoring runs:

| condition | accuracy | macro-F1 | abstention |
|---|---|---|---|
| closed corpus | 0.592 | 0.552 | — |
| live snippets | 0.684 | 0.552 | 23.7% |
| majority-class baseline | **0.750** | — | — |
| oracle gold evidence | 0.763 | **0.649** | — |
| **live + full text** | **0.816** | 0.607 | **10.5%** |

Enrichment is worth **+0.132 accuracy, +0.055 macro-F1** (paired p=0.041), cuts
abstention by more than half, and **survives the leakage control**. **`FULL_TEXT_TOP_K=3`
should stay on.**

Three results that reframe the project's priorities:

* **Live full-text retrieval reaches gold-evidence quality.** 0.816 vs the oracle's
  0.763, not separable (14:10, p=0.54). The system does not need the benchmark's
  annotations to reach benchmark-level verdict quality — so retrieval is not the
  bottleneck it appeared to be on the closed corpus.
* **The closed corpus understates live quality**, now established across four samples
  (22:5, p=0.0015). Every closed-corpus-based conclusion in this project is a lower
  bound, and §L.5 puts **±0.05** on any single-run closed-corpus figure.
* **Fact-check leakage masks the enrichment effect rather than creating it.** The gain
  is significant on the 43 claims with *no* fact-checker retrieved (10:2, p=0.039) and
  absent on the 33 with one (5:3, p=0.73). An n=35 reading of this said the opposite,
  and it is withdrawn.

For the record, this priority reversed twice as the sample grew: an n=35 reading had
enrichment "not replicating live" (p=1.000) and Stage 3 beating no baseline at all.
Both were 3:2-discordant-pair artifacts.

### P5c — Stage 3 measurement is not reproducible *(methodological)*
The verdict call runs at **temperature 0.2 and is not cached**, so re-scoring identical
claims changes 2–4% of labels. At n=76, three runs per condition:

| condition | spread | unstable labels | variance sources |
|---|---|---|---|
| live snippets / full text | 0.000 | 0–1 / 76 | verdict call |
| oracle gold | 0.013 | 2 / 76 | verdict call |
| **closed corpus** | **0.053** | **7 / 76** | verdict **+ retrieval** |

**The closed corpus is 4× noisier than the oracle**, because query generation at
temperature 0.3 means each run retrieves a different document set — a prediction written
into the harness before the run and confirmed by it. At n=55 this instability moved a
paired p-value across 0.05 on identical data (0.012, 0.006, 0.065); at n=76 the live
conditions are perfectly stable, so **larger samples buy reproducibility as well as
power**.

Both `live_evidence_verdict.py` and `same_claim_baselines.py` now take `--repeat N` and
report per-run accuracy, spread, unstable-label counts and a majority-vote label.

**Remaining:** re-check pre-2026-10-08 Stage 3 findings with near-balanced discordant
counts, closed-corpus-based ones first. One-sided results such as §S.3 (0/9, 0/13) are
robust.

### P5d — Volume or quality? *(the sharpest open question)*
Live full text beats snippets by +0.132 and matches gold evidence, but it is **9.7×
longer** *and* differently sourced, so the active ingredient is unidentified. Truncating
live full text to ~246 characters — the oracle's length — separates the two: if the
advantage survives, it is better text; if it vanishes, it is simply more text. Costs no
search quota and reuses the existing harness.

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
- **Withdraw findings in place.** Five claims have been retracted with their
  reasoning left visible rather than edited away — most recently the n=13
  live-evidence result (`FINDINGS.md` §L), which tripling the sample dissolved.
- **Treat a small-sample result as a hypothesis, and go back and test it.** §L
  reported p=0.031 at n=13 and recommended a larger run; the larger run returned
  p=0.180. The practice that caught it was re-measuring the baselines on the *same*
  claims rather than comparing across samples — and the check that was supposed to
  catch it at n=13 (confirming the sample wasn't unusually easy) gave the wrong
  answer, because 13 claims cannot establish representativeness either.
- **Label untested changes as untested**, and never report the removal of dead
  code as an improvement.

---

**Status:** measurement-driven. Stages 1–3 complete; Stage 4 not started.
**Evidence:** `results/claim_detection/`, `results/evidence_retrieval/`,
`results/verdict/`.
