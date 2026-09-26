<!-- Archived from ~/.claude/plans/ so it lives with the repo. -->

# Stage 3 Plan — Reasoning & Verdict

**Status: COMPLETE.** All plan items and verification points closed. Written 2026-09-25,
executed 2026-09-26. Results are in `results/verdict/FINDINGS.md`.

> **Note on provenance.** Claude Code keeps plans in `~/.claude/plans/`, outside
> the repo and not version-controlled. This is a copy. The Stage 2 plan was
> overwritten in that same file when this one was written, so it no longer exists
> as a document — its substance is captured in
> `results/evidence_retrieval/RESULTS.md`.

## What happened against this plan

| item | status | outcome |
|---|---|---|
| A1 instrument the stage | ✅ done | counters + decision log; 11 tests pin verdicts unchanged |
| A2 calibration | ✅ done | ECE 0.202; confidence only ever `{0.8, 0.9, 0.95, 1.0}` |
| A3 locate the SUPPORTS loss | ✅ done | 14/14 misses are the model, not a rule |
| B1 prompt symmetry | ✅ tested → **rejected** | class trade, p=1.0 / p=0.45 |
| B2 remove dead threshold rules | ✅ done | removed; replaced with a low-confidence warning. Clarity only, not an improvement |
| B3 fix `used_evidence_ids` | ✅ documented | field description corrected; real citation support deferred as a measured Stage 4 change |
| B4 calibration work | ✅ closed | not actionable: nothing downstream reads confidence. Revisit if that changes |
| C paired evaluation | ✅ done | McNemar, oracle + retrieved, majority baseline throughout |
| Verification 4: stale smoke test | ✅ done | rewritten; it ran nothing and exited 0, with 5/6 failing when invoked |

**Two experiments were run beyond this plan**, both prompted by its findings:

* *Evidence sufficiency* — the fact-checker's justification is worth +0.132
  accuracy (p=0.0039 on the non-leaking half). The evidence carries facts but not
  the inferential bridge to a verdict.
* *Reasoning synthesis* — deriving that bridge from the same evidence made
  everything worse at 2× cost. Together these show the justification helps because
  it contains information **absent** from the evidence.

**The plan's central hypothesis was refuted.** It was built to test whether the
`SUPPORTS → NOT_ENOUGH_INFO` demotion rule caused the SUPPORTS deficit; the rule
fires on 0 of 150 decisions. Starting with diagnosis rather than the pre-registered
fix is the reason that was caught before any code changed.

**Conclusion:** the verdict stage is not the tractable bottleneck it appeared to
be. Over-abstention is substantially a reasonable response to inferentially
incomplete evidence and cannot be prompted away. The remaining levers are
retrieval-side.

---

# Stage 3 — Reasoning & Verdict: diagnose first, optimise only what the data implicates

## Context

Stage 2 is complete and committed. Its evaluation left the verdict stage as the
largest remaining block of lost accuracy, and pre-registered the first test
(`results/evidence_retrieval/RESULTS.md` §11, item 4).

What is already established, on AVeriTeC dev (n=136 mappable, closed corpus):

* The stage scores **0.566 accuracy against a 0.706 always-REFUTES baseline** —
  below the majority-class prior (§7.3). It does beat that baseline on macro-F1
  (0.499 vs 0.276), so it distributes across classes rather than collapsing.
* It **over-abstains ~7x** (47% `NOT_ENOUGH_INFO` predicted vs ~7% gold) and
  **under-refutes** (45% predicted vs 71% gold) — the opposite of what the
  `IMPROVEMENT_PLAN` changes were meant to achieve.
* **Over-abstention persists on gold evidence** (27.9%, §8.3), so this is not
  merely a retrieval problem.
* `SUPPORTS` is the weakest class everywhere, even with perfect evidence (0.56).
* Nothing about the verdict stage has ever been measured by the project itself:
  `ACADEMIC_DOCUMENTATION.md` §6.3 is titled **"Expected Outcomes"** and its
  predicted distribution (30–40% REFUTES) is contradicted by the measurements
  above.

### The hypothesis this plan was going to be built on is probably wrong

§8.4 attributed the `SUPPORTS` deficit to the asymmetric rule at
`reasoning_and_verdict.py:85-96`: a low-confidence `SUPPORTS` is rewritten to
`NOT_ENOUGH_INFO`, while a low-confidence `REFUTES` is kept.

But that rule only fires below `confidence < 0.5`, and the saved sweep data shows
mean confidence of **0.910, 0.905, 0.904, 0.905, 0.901** across k=5…15 while
accuracy ranged 0.529–0.566. If the model almost always emits ≈0.9, **the rule
can rarely be firing**, and cannot be the main cause.

That same data exposes a larger, unreported problem: the stage claims ~91%
confidence while being ~55% correct — a **0.35 calibration gap** — and its
confidence does not respond to a 0.09 swing in retrieval recall. Confidence is
close to a constant, carrying almost no information. Nothing in the codebase
computes ECE, Brier score, or a reliability curve; nothing downstream reads
`confidence` except display and those threshold rules.

So the first job is diagnosis, not optimisation. This matches the requested
sequence: **analyse → optimise only if warranted → you evaluate.**

## Non-goals

* **No 4th label.** `Conflicting Evidence/Cherrypicking` stays excluded from
  scoring, as `sweep_k.py:16-27` argues: scoring it against three labels would
  measure a representational gap, not verdict quality. (It is 14 of 150 claims;
  the system currently scatters them across all three labels rather than
  abstaining, so nothing is silently broken.)
* **No changes to retrieval or Stage 1.**
* **No explanation-stage work.** `explanation.py` is a separate pipeline step
  that narrates an already-fixed verdict; it is Stage 4.

---

## Phase A — Diagnose (no behaviour change)

Instrument first so every later claim is evidence-backed. None of this alters
verdicts.

### A1. Make the stage observable — `reasoning_and_verdict.py`
Add counters and structured logging only:
* how often each threshold branch fires (`SUPPORTS` demotion, `REFUTES` low-conf
  note, neither);
* parse failures. The blanket `except Exception` at `:104-111` currently makes an
  API timeout, malformed JSON, and genuine uncertainty **indistinguishable** —
  all three emit `NOT_ENOUGH_INFO`, silently. An eval run cannot presently report
  its own parse-failure rate.
* Replace the `re.search(r'\{[^}]+\}', ...)` at `:66` with the balanced-brace
  parser already written and tested in
  `evidence_evaluation.py:_extract_json_object` — the current regex cannot match
  nested objects and truncates at the first `}` inside `reasoning`. This is a
  correctness fix, not a behaviour change.

### A2. Calibration analysis — new `results/verdict/calibration.py`
Per-claim confidence is currently **not saved** (only aggregates), so this needs
one cheap re-run against the oracle condition.
* Reliability curve, **ECE**, Brier score, accuracy-vs-confidence buckets.
* Confidence distribution per predicted label — specifically **what fraction
  falls below the 0.5 and 0.4 thresholds**, which settles whether the §8.4 rule
  fires at all.

### A3. Locate the SUPPORTS loss — new `results/verdict/error_analysis.py`
For every gold-`Supported` claim the system gets wrong, record what it predicted,
its confidence, whether a threshold rewrote it, and the evidence it saw. Separate
three candidate causes:
* the **prompt asymmetry** at `:38-42` — "If ANY evidence clearly shows the claim
  is FALSE … you **MUST** return REFUTES" versus "**Only** return SUPPORTS if
  evidence **clearly** confirms";
* the **threshold rewrite** at `:85-96`;
* genuine evidence insufficiency.

Run under the **oracle condition** (`error_propagation.py:gold_evidence`) so
retrieval noise is excluded.

**Phase A output:** a short findings note naming which cause the data supports.
No optimisation until then.

---

## Phase B — Optimise only what Phase A implicates

Candidates, in the order the current evidence favours. Each is a separate,
independently measurable change:

* **B1 Prompt symmetry** — rewrite `:38-42` so the evidential bar for `SUPPORTS`
  matches `REFUTES`, and drop the near-duplicate criteria block at `:44-47`.
* **B2 Threshold rules** — remove or symmetrise `:85-96`. Note the magic
  constants: a demoted `SUPPORTS` has its confidence overwritten with `0.4`
  (`:89`), the empty-evidence path emits `0.0` (`:19`), and the exception path
  `0.3` (`:108`) — none of these mean anything comparable.
* **B3 Evidence IDs in the prompt** — evidence is currently rendered as
  `f"[{e.source}] {e.text}"` (`:26`) with **no IDs**, yet `used_evidence_ids`
  (`:28`) is set to "the first 10 snippets we showed", not what the model cited.
  The prompt asks it to "cite specific evidence" while giving it nothing to cite.
* **B4 Calibration** — only if A2 shows the ~0.91 constant is actionable.

Apply the narrowest change the evidence supports, and measure each separately.

---

## Phase C — Evaluate

Reuse the Stage 2 harness rather than writing new scoring:

* `sweep_k.py:63-68` — `LABEL_MAP`, `CLASSES` (single chokepoint for metrics)
* `sweep_k.py:71-81` — `macro_f1`
* `error_propagation.py:62-73` — `gold_evidence`, the oracle pattern
* `fulltext_verdict.py:68-81` — `mcnemar`, exact paired test for binary outcomes
* `summarise()` in either script — abstention rate, prediction distribution,
  per-class recall

**Paired design, as in §10:** run each condition over the *same* claims with the
*same* evidence, varying only the verdict logic. Report accuracy, macro-F1,
abstention, per-class recall, and **always the majority-class baseline
alongside** — 0.706 here, and any accuracy at or below it is not evidence of
skill.

Run both **oracle** (isolates the verdict stage) and **retrieved** (what users
actually get) conditions. The oracle number is the honest ceiling.

## Verification

1. **Behavioural tests** for the new counters and the parser swap, added to
   `tests/test_stage2_behaviour.py` (43 checks currently passing, offline).
   Specifically: nested-brace JSON parses correctly; a parse failure is *counted*
   rather than silently becoming `NOT_ENOUGH_INFO`.
2. **Phase A changes must not alter any verdict** — re-run the oracle condition
   and confirm the label distribution is unchanged before/after instrumentation.
3. **Paired significance** (McNemar) for every Phase B change, against the
   pre-change baseline on identical claims and evidence.
4. **Regression guard:** `tests/test_agents_smoke.py:65-86` is already stale and
   cannot pass — it calls `ReasoningAndVerdictAgent()` with no `llm` and treats
   `decide()` as synchronous. Fix or delete it rather than leaving a broken test
   that masks real breakage.

## Risks

* **The pre-registered hypothesis may not survive.** §8.4 named the demotion rule;
  the confidence data suggests it rarely fires. If A2 confirms that, §8.4 must be
  **withdrawn in `RESULTS.md`** the way two earlier claims already were, with the
  reasoning kept.
* **Changing these rules supersedes `IMPROVEMENT_PLAN` Priorities 1.1/2.2** and
  makes `ACADEMIC_DOCUMENTATION.md` §6.2–6.3 stale. Those sections are already
  contradicted by measurement, so they need rewriting regardless — but the
  rewrite should be deliberate, not incidental.
* **Closed-corpus numbers are a lower bound** (§7.4). The oracle condition
  mitigates this for Stage 3 specifically, since it removes retrieval entirely.
* **Improving abstention may lower headline accuracy.** With 71% of gold claims
  `Refuted`, moving probability mass out of `NOT_ENOUGH_INFO` toward `SUPPORTS`
  can reduce accuracy while improving macro-F1. Report both; prefer macro-F1.
