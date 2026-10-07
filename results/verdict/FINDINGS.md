# Stage 3 Phase A — Verdict Stage Diagnosis

**Date:** 2026-09-26
**Condition:** ORACLE — AVeriTeC gold evidence fed straight to the verdict stage,
so retrieval quality is removed and everything here belongs to Stage 3.
**Scope:** 150 claims, 136 scored (14 `Conflicting Evidence/Cherrypicking`
excluded — no system equivalent).
**Code state:** instrumentation only. Verdicts are unchanged; 11 tests in
`tests/test_stage3_verdict.py` pin that, including one that deliberately confirms
the known JSON-regex defect is still present.

---

## 1. The pre-registered hypothesis is refuted

`results/evidence_retrieval/RESULTS.md` §8.4 attributed the `SUPPORTS` deficit to
the asymmetric rule at `reasoning_and_verdict.py:85-96`, which demotes a
low-confidence `SUPPORTS` to `NOT_ENOUGH_INFO` while keeping a low-confidence
`REFUTES`. §11 pre-registered the test: disable it and re-measure.

**The rule never fires.**

| counter | value |
|---|---|
| decisions | 150 |
| **SUPPORTS demoted (confidence < 0.5)** | **0** |
| REFUTES low-confidence noted (< 0.4) | 0 |
| raw confidence below 0.5 | **0 / 136** |
| raw confidence below 0.4 | **0 / 136** |

The reason is that the model emits only **four distinct confidence values —
0.8, 0.9, 0.95, 1.0** — with a minimum of 0.800. The thresholds at 0.5 and 0.4
are **unreachable**. Those rules are dead code and have been since they were
added.

**§8.4 is withdrawn.** The demotion rule cannot explain the `SUPPORTS` deficit,
and `IMPROVEMENT_PLAN` Priority 2.2 has never had any effect on any verdict.

## 2. Where SUPPORTS is actually lost: the model itself

Of 32 gold-`Supported` claims, 18 correct and **14 missed**:

| | count |
|---|---|
| predicted `NOT_ENOUGH_INFO` | 12 |
| predicted `REFUTES` | 2 |
| **the model's own raw answer, before any rule** | **identical — 12 / 2** |
| a threshold rule rewrote the answer | **0** |

Attribution is unambiguous: **14 of 14 misses are the model never saying
`SUPPORTS` in the first place.** No post-processing is involved.

This points at the prompt (`reasoning_and_verdict.py:38-42`), whose evidential
bars are asymmetric:

> `- If ANY evidence clearly shows the claim is FALSE or INCORRECT, you MUST return "REFUTES"`
> `- Only return "SUPPORTS" if evidence clearly confirms the claim is true`

`REFUTES` is reachable on *any* evidence and is a **MUST**; `SUPPORTS` is gated by
**Only** and requires evidence that **clearly confirms**. A third line —
*"Be decisive: if evidence contradicts the claim, return REFUTES even if
confidence is moderate"* — lowers the bar for `REFUTES` again with no counterpart.

## 3. Over-abstention is 89% of all errors

Confusion matrix (oracle evidence):

| gold ↓ / predicted → | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | total |
|---|---|---|---|---|
| SUPPORTS | **18** | 2 | **12** | 32 |
| REFUTES | 2 | **73** | **21** | 96 |
| NOT_ENOUGH_INFO | 0 | 0 | **8** | 8 |
| total | 20 | 75 | 41 | 136 |

- **33 false `NOT_ENOUGH_INFO`** out of **37 total errors — 89%.** The system's
  dominant failure is declining to answer when handed gold-standard evidence.
- It affects **both** classes, not just `SUPPORTS`: 12 of 32 `SUPPORTS` (38% of the
  class) and 21 of 96 `REFUTES` (22%). In absolute terms more `REFUTES` are lost.
- `NOT_ENOUGH_INFO` recall is **1.00** (8/8) — the system never misses a genuine
  abstention. It simply abstains 41 times when 8 were warranted.

This refines the §8.4 framing further: the problem is not specific to `SUPPORTS`.
It is a general reluctance to commit, which hits `SUPPORTS` proportionally hardest.

## 4. Confidence is uninformative and overconfident

| metric | value |
|---|---|
| mean confidence | 0.930 |
| accuracy | 0.728 |
| **overconfidence gap** | **+0.202** |
| **ECE** | **0.202** |
| Brier | 0.228 |
| values ever emitted | **{0.8, 0.9, 0.95, 1.0}** |

Reliability:

| bucket | n | mean confidence | accuracy |
|---|---|---|---|
| 0.8–0.9 | 4 | 0.800 | **0.000** |
| 0.9–1.0 | 132 | 0.934 | 0.750 |

132 of 136 predictions land in a single bucket, so confidence cannot discriminate
correct from incorrect. Every one of the four claims the model marked 0.800 was
wrong — suggestive of a usable signal at the low end, but n=4.

Nothing downstream reads `confidence` except display and the two dead threshold
rules, so this currently costs nothing — but it also means the system cannot
defer, rank, or flag uncertain verdicts.

## 5. The JSON-regex defect is latent, not active

**0 parse failures and 0 LLM errors in 150 decisions.** 0 label coercions, 0
missing confidence fields.

The `re.search(r'\{[^}]+\}', ...)` at `:66` genuinely cannot handle nested objects
or a `}` inside `reasoning` — proven by test — but on 150 real responses it never
failed. Replacing it is still correct defensively, and would be needed before
adding nested fields to the schema, but it is **not** a source of current error
and should be deprioritised.

## 6. One number that improves on the earlier report

Oracle accuracy **0.728 against the 0.706 majority baseline** — marginally above,
consistent with the 0.750 measured in §8 of the Stage 2 results (the difference is
temperature-0.2 nondeterminism). Macro-F1 **0.624**, well above the 0.276 a
constant `REFUTES` predictor achieves.

So with perfect evidence the stage does beat the prior — barely. The 0.272 gap to
a perfect verdict stage is dominated by the 33 unwarranted abstentions.

---

## Phase B target

The evidence redirects Phase B from thresholds to the **prompt**:

1. **B1 — prompt symmetry (do this).** Equalise the evidential bar for `SUPPORTS`
   and `REFUTES`, and remove the third "be decisive … return REFUTES" line that
   has no counterpart. All 14 `SUPPORTS` misses and the 33 false abstentions are
   attributable to model behaviour under this prompt.
2. **B2 — threshold rules (dead code).** Fires on 0 of 150 decisions. Remove for
   clarity, but expect **no** measurable effect. Do not report removal as an
   improvement.
3. **B3 — evidence IDs in the prompt.** Independent correctness issue:
   `used_evidence_ids` is set to "the first 10 snippets shown" (`:28`), not what
   the model cited, and the prompt asks it to "cite specific evidence" while
   providing no IDs to cite.
4. **B4 — calibration.** Confidence spans only [0.8, 1.0] in four values. Worth
   revisiting only if a use for it exists downstream.

Measurement for any B change: paired, same claims, same evidence, McNemar's exact
test, majority baseline always reported alongside — and both oracle and retrieved
conditions, since the oracle is the honest ceiling.

**Caveat to carry:** improving abstention may *lower* headline accuracy. With 71%
of gold claims `Refuted`, moving mass out of `NOT_ENOUGH_INFO` can cost accuracy
while improving macro-F1. Prefer macro-F1 and report both.

---

# Stage 3 Phase B1 — Prompt symmetry: tested and REJECTED

**Date:** 2026-09-26 · 150 claims, 136 scored, paired (same evidence per claim,
only the prompt varies) · McNemar exact test
**Data:** `prompt_comparison_b1_dev.json` — this section's figures come from the
first, two-arm run. A later three-arm run (adding synthesis) overwrote
`prompt_comparison_dev.json`, so that file holds the *second* run and its `direct`
arm differs slightly here from `original` (temperature-0.2 nondeterminism, not a
code change).

## B1.1 The result

| condition | variant | accuracy | macro-F1 | abstention | SUPPORTS | REFUTES |
|---|---|---|---|---|---|---|
| oracle | original | **0.735** | 0.629 | 29.4% | 0.562 | **0.771** |
| oracle | symmetric | 0.728 | 0.641 | 27.2% | **0.719** | 0.708 |
| retrieved | original | **0.544** | 0.457 | 47.1% | 0.312 | **0.604** |
| retrieved | symmetric | 0.515 | 0.478 | 48.5% | **0.500** | 0.500 |

**Not significant in either condition.** McNemar p=**1.0** (oracle, 9 vs 8
discordant pairs) and p=**0.45** (retrieved, 10 vs 6). Accuracy *fell* (−0.007,
−0.029); macro-F1 rose marginally (+0.012, +0.022), well inside noise.

**Recommendation: keep `prompt_variant="original"` as the default.** The symmetric
prompt is available and tested, but the evidence does not justify switching.

## B1.2 What actually happened: the asymmetry was load-bearing

The rewrite did exactly what it was designed to do on `SUPPORTS` — and paid for it
on `REFUTES`:

| condition | SUPPORTS gained | REFUTES lost | net |
|---|---|---|---|
| oracle | **+5** | **−6** | −1 |
| retrieved | **+6** | **−10** | −4 |

`SUPPORTS` recall improved substantially (0.562 → 0.719 oracle, +28% relative;
0.312 → 0.500 retrieved, +60%), confirming Phase A's diagnosis that the prompt
asymmetry was suppressing confirmations. But the *"If ANY evidence clearly shows
the claim is FALSE … you MUST return REFUTES"* instruction was **propping up
`REFUTES` recall**. Removing it cost more than the `SUPPORTS` fix gained.

This is precisely the failure mode flagged in advance: a gain in one decisive
class paid for by a loss in the other is not an improvement.

## B1.3 The primary target was not hit at all

B1's main goal was reducing the 89%-of-errors over-abstention. It did not:

| condition | abstention | false abstentions |
|---|---|---|
| oracle | 29.4% → 27.2% | 32 → 29 |
| retrieved | 47.1% → **48.5%** | 58 → **60** |

On retrieved evidence abstention got **worse**. The label-change breakdown shows
why: `REFUTES → NOT_ENOUGH_INFO` moved **7** claims (oracle) and **9**
(retrieved). When the model lost its licence to refute on any contradicting
evidence, it did **not** fall back to `SUPPORTS` — it fell back to abstaining.

The explicit instruction narrowing `NOT_ENOUGH_INFO` ("reserve it for evidence
that genuinely does not address the claim — not for evidence that is indirect,
partial, or merely implies the answer") had **no detectable effect**. The model
ignored it.

## B1.4 A weakness in my own experiment

The symmetric variant bundled **three** changes — equalised bars, removed the
one-sided decisiveness nudge, and narrowed `NOT_ENOUGH_INFO`. Because they moved
together, this run cannot attribute the outcome to any one of them. The
`SUPPORTS` gain and the `REFUTES` loss are most plausibly both from the first two,
and the third appears inert — but that is inference, not measurement. Isolating
them needs one variant per change.

## B1.5 What this rules out, and what is left

**Ruled out:** over-abstention is not caused by the prompt's asymmetric evidential
bars, and is not fixable by instructing the model to abstain less. Two different
prompt formulations produced ~the same abstention rate.

Remaining hypotheses, in order of my confidence:

1. **Evidence sufficiency is genuinely marginal.** Even the oracle supplies only
   ~2.0 short QA-style answers per claim (§9 of the Stage 2 results). Abstaining
   on that may be defensible behaviour rather than a defect — which would mean the
   0.272 gap to a perfect verdict stage is smaller than it looks.
2. **`NOT_ENOUGH_INFO` is an attractive safe default that prose cannot override.**
   Few-shot examples showing confident judgements on partial evidence would test
   this where instructions failed.
3. **Model capability.** `gpt-4.1-mini` may simply not adjudicate these claims
   well. Testing a stronger model would bound how much of the gap is addressable
   by prompting at all.

Hypothesis 1 is the cheapest to check and would reframe the whole Stage 3 target,
so it should come first.

---

# Stage 3 — Is the evidence genuinely marginal? YES, in inferential completeness

**Date:** 2026-09-26 · 150 claims, 136 scored · paired, gold evidence under three
framings, original prompt throughout · McNemar exact test
**Data:** `evidence_sufficiency_dev.json`. Figures below are from the second run,
which added the leakage split; an earlier run without it differed by ≤0.008 on
accuracy (temperature-0.2 nondeterminism, not a code change).

## S.1 A flaw in the earlier oracle — real, but immaterial

`error_propagation.gold_evidence` passed each annotated answer's **text only**,
discarding the QUESTION it answers. AVeriTeC evidence is question-answer pairs, and
an answer alone can be close to meaningless: *"It was first published on
Sccopertino"* means little without *"Where was the claim first published"*. So the
§8 "oracle" was never a true oracle.

Correcting it changes nothing significant:

| condition | accuracy | macro-F1 | abstention | SUPPORTS | REFUTES |
|---|---|---|---|---|---|
| answer_only | 0.721 | **0.614** | 30.9% | **0.531** | 0.760 |
| question_answer | 0.743 | 0.604 | 25.0% | 0.469 | **0.823** |

McNemar p=**0.58**. Accuracy +0.022, but macro-F1 *fell* and the classes traded
again (`SUPPORTS` −0.06, `REFUTES` +0.06). The harness flaw was genuine and worth
fixing, but it was not the cause of over-abstention.

## S.2 Supplying the fact-checker's justification transforms performance

| condition | chars | accuracy | macro-F1 | abstention | false abstentions |
|---|---|---|---|---|---|
| question_answer | 662 | 0.743 | 0.604 | 25.0% | 27 |
| **plus_justification** | 788 | **0.904** | **0.802** | **13.2%** | **10** |

22 claims correct **only** with the justification, 0 the other way, p≈0.

Note the size of the input barely changed (662 → 788 characters). This is not a
volume effect — §7 and §8.1 already showed more evidence does not help. It is the
*kind* of content.

## S.3 The gain is not just leakage — controlled and confirmed

**50% of justifications state the verdict outright** ("…so the claim is refuted"),
so condition C is partly "being told the answer". Splitting on that:

| subset | n | question_answer | +justification | delta | McNemar | p |
|---|---|---|---|---|---|---|
| leaky (verdict named) | 68 | 0.706 | 0.897 | **+0.191** | 0 / 13 | 0.00024 |
| **clean (verdict NOT named)** | **68** | 0.779 | **0.912** | **+0.132** | **0 / 9** | **0.0039** |

**The gain survives on non-leaking justifications: +0.132, p=0.0039**, with 9
claims right only with the justification and 0 the other way. Leakage inflates the
effect (+0.191 vs +0.132) but does not create it.

*Caveat:* the leak detector is a keyword heuristic, so some "clean" justifications
may still hint at the direction without using a flagged word. +0.132 is therefore
best read as a strong lower-bound-ish estimate of the reasoning benefit, not an
exact figure. And condition C remains unusable as a system configuration — no
fact-checker justification exists at inference time.

## S.4 What this means: the pipeline is missing an inferential step

The QA evidence supplies **facts**; it does not supply the **bridge** from those
facts to a verdict. The justification supplies the bridge, and doing so is worth
~0.13 accuracy and roughly halves abstention — even when it does not name the
verdict.

So the over-abstention documented in Phase A §3 is **substantially a reasonable
response to inferentially incomplete evidence**, not simply a defect. That
reframes the Stage 3 target, and it explains why B1 failed: the missing ingredient
is not an instruction, so no prompt wording could supply it.

It also exposes an architectural gap. AVeriTeC's task design is:

    claim -> questions -> answers -> justification -> verdict

This pipeline is:

    claim -> search -> snippets -> verdict

There is no question-decomposition step and no reasoning-synthesis step. The
benchmark's own structure asserts both are needed, and this experiment quantifies
what the second one is worth.

## S.5 Recommended next direction

**Add an explicit reasoning step between evidence and verdict** — have the model
first derive, from the retrieved evidence alone, what the evidence establishes
about the claim, then judge on that. This is a legitimate, self-contained change
(no gold data at inference), and it targets the gap this experiment measured
rather than the ones B1 ruled out.

A question-decomposition step (claim -> sub-questions -> per-question answers from
evidence) is the fuller version and matches the benchmark's design, but it is a
larger change and should follow, not precede, the cheaper synthesis test.

Revised ranking of the Phase A/B leftovers:

1. **Evidence-to-verdict reasoning synthesis** — measured motivation, cheap to test.
2. **B3 `used_evidence_ids` correctness** — independent real bug.
3. **Few-shot examples** — still untested, but S.4 suggests the deficit is
   inferential rather than format-related, so expectations should be lower.
4. **B2 remove dead threshold code** — clarity only, 0/150 firings.
5. **Stronger model** — bounds what prompting can achieve at all.

---

# Stage 3 — Reasoning synthesis: tested and REJECTED

**Date:** 2026-09-26 · 150 claims, 136 scored · three arms, paired, same evidence
per claim · McNemar exact test

Motivated by §S.2-S.4: supplying the fact-checker's justification — the
inferential bridge from facts to verdict — was worth +0.132 accuracy (p=0.0039)
even when it never named the verdict. `reasoning_mode="synthesis"` adds a first
LLM call that tries to derive that bridge **from the evidence alone**, then judges
using it.

## R.1 It made things worse, in both conditions

| condition | arm | accuracy | macro-F1 | abstention | SUPPORTS | calls |
|---|---|---|---|---|---|---|
| oracle | direct | **0.728** | 0.618 | 30.1% | 0.531 | 150 |
| oracle | symmetric | **0.735** | **0.643** | **26.5%** | **0.688** | 150 |
| oracle | **synthesis** | **0.669** | **0.556** | **37.5%** | **0.406** | **300** |
| retrieved | direct | **0.544** | 0.470 | 48.5% | 0.344 | 150 |
| retrieved | symmetric | **0.544** | **0.496** | **47.1%** | **0.469** | 150 |
| retrieved | **synthesis** | 0.529 | **0.426** | **50.0%** | **0.219** | **300** |

Synthesis is worse on **every metric in both conditions**, at **2x the LLM calls**.
Oracle: accuracy −0.059, macro-F1 −0.062 (McNemar 13/5, p=0.096). Retrieved:
−0.015, −0.044 (p=0.81). Neither reaches significance, but the direction is
consistent and the cost is certain, so there is no case for adopting it.

`SUPPORTS` recall was hit hardest: 0.531 → 0.406 (oracle) and 0.344 → 0.219
(retrieved).

## R.2 The mechanism is the one predicted in advance

The synthesis prompt asks, among other things, what the claim asserts **that the
evidence does not address**. Making gaps explicit appears to make the model more
willing to abstain. Label changes under synthesis (oracle):

    REFUTES  -> NOT_ENOUGH_INFO   8
    SUPPORTS -> NOT_ENOUGH_INFO   5

**13 decisions converted into abstentions**, against 3 abstentions converted into
decisions. An own-goal: the step intended to reduce abstention increased it.

## R.3 The important inference

Having the inferential bridge is worth +0.132 (§S.3). **Generating it from the
same evidence is worth −0.059.** Those are not in tension — together they say
something sharper than either alone:

> The fact-checker's justification helps because it contains information that is
> **not present in the evidence**, not because it is better-organised reasoning
> over the same facts.

This bounds Stage 3 firmly. The verdict stage cannot be repaired by reasoning
harder over what it currently receives. Three prompt-level interventions have now
been tested — symmetry, narrowed abstention, explicit synthesis — and none moved
the needle. The remaining levers are **more or better evidence**, or **external
knowledge**, neither of which lives in Stage 3.

It also revises §S.5, which recommended exactly this experiment: the recommendation
was reasonable from the evidence available, and the experiment disconfirmed it.
Question decomposition (claim → sub-questions → per-question retrieval) remains
untested and is now the more plausible reading of the AVeriTeC gap — but note it
is a **retrieval-side** change, not a verdict-side one.

## R.4 Where symmetric stands after two runs

| run | condition | macro-F1 delta vs direct | p |
|---|---|---|---|
| first | oracle | +0.012 | 1.0 |
| first | retrieved | +0.022 | 0.45 |
| second | oracle | +0.026 | 1.0 |
| second | retrieved | +0.026 | 1.0 |

Consistently positive on macro-F1 across four independent measurements, never
significant, and accuracy-neutral. The `SUPPORTS`/`REFUTES` trade seen in B1
persists. **Still not adopted** — but the consistency is worth noting if a larger
sample is ever run.

## R.5 Stage 3 conclusion

The verdict stage is **not the tractable bottleneck it appeared to be**. Its
over-abstention is substantially a reasonable response to inferentially incomplete
evidence (§S.3), and cannot be prompted away (B1, R.1). The measured headroom that
looked like a Stage 3 problem is better read as an **evidence problem** that Stage 3
merely surfaces.

Defaults are unchanged: `prompt_variant="original"`, `reasoning_mode="direct"`.
Both alternatives remain implemented, tested and available.

---

# Stage 3 close-out

**Date:** 2026-09-26. The four items left open by `docs/stage3_verdict_plan.md`.

## C.1 B2 — dead threshold rules removed

`SUPPORTS_MIN_CONFIDENCE`/`DEMOTED_CONFIDENCE`/`REFUTES_LOW_CONFIDENCE` and the
branches using them are gone. They fired on **0 of 150** decisions (§1) because the
model's confidence floor is 0.8, so both thresholds were unreachable.

**This is a clarity change and must not be reported as an improvement.** No verdict
behaviour changes for the current model. What replaces them is observability: a
confidence below `LOW_CONFIDENCE_WARN = 0.5` is now logged as a warning rather than
silently rewritten, so if a future model does emit low confidence it surfaces
instead of being masked.

Consequence for the docs: `IMPROVEMENT_PLAN` Priority 2.2 and
`ACADEMIC_DOCUMENTATION.md` §4.1.5/§6.2 describe these rules as an implemented
safeguard. They were never reachable. Those sections are now wrong twice over —
once for the prediction in §6.3, once for describing removed dead code.

`results/verdict/diagnose.py` still runs: the historical threshold values are kept
inside it as constants so the "could the rule ever have fired?" analysis remains
reproducible against the saved run.

## C.2 B3 — `used_evidence_ids` documented honestly, not fixed

The field is set to `[e.id for e in evidence[:10]]` — the snippets **shown** to the
model. The prompt contains no evidence IDs, so the model cannot cite and never
reports which snippets it relied on. Calling it "used" overstates what is known,
and `models.py` previously described it as *"IDs of evidence snippets referenced in
the verdict"*, which was simply false.

Both the field description and the assignment site now state what it actually is.

**Deliberately not converted into a real citation list.** Doing that means putting
IDs into the prompt and asking for them back, which changes verdict behaviour and
therefore needs its own paired measurement — not a quiet edit during close-out.
It is a **prerequisite for Stage 4**: grounding explanations in evidence IDs is
meaningless while the IDs are fabricated.

## C.3 B4 — calibration closed as not actionable

§4 measured ECE 0.202, a +0.202 overconfidence gap, and confidence taking only four
values `{0.8, 0.9, 0.95, 1.0}` with 132 of 136 predictions in one reliability
bucket. Confidence is effectively a constant and carries almost no information.

Closed without action, for a specific reason: **nothing downstream reads it.** With
the threshold rules removed (C.1), `confidence` now feeds only display and the
explanation prompt. Calibrating a number that no decision depends on would be
effort without effect.

It becomes worth revisiting the moment something *does* depend on it — a deferral
gate, evidence ranking, or surfacing uncertainty to a user. Recorded rather than
silently dropped.

## C.4 Verification item 4 — the fake-green smoke test

`tests/test_agents_smoke.py` **exited 0 while executing nothing**: it had no
`__main__` block, so running it asserted nothing at all. Invoking its functions
directly showed **5 of 6 failing** against a pre-LLM API — `RetrievalAgent(corpus=…)`,
sync calls to `async` methods, agents constructed without their required `llm`, and
an assertion expecting the lowercasing that was deliberately removed.

A test that passes without running is worse than no test, because the suite looks
green. Rewritten against the current APIs, now actually executing, asserting
contracts rather than long-replaced stub return values, and covering the two
regressions this project cares about: that normalization preserves case, and that
retrieval populates a relevance score.

## Test suite after close-out

| file | checks | scope |
|---|---|---|
| `test_stage2_behaviour.py` | 43 | retrieval loop, scoring, enrichment, leakage, orchestration |
| `test_stage3_verdict.py` | 30 | verdict instrumentation, threshold removal, parse failures |
| `test_agents_smoke.py` | 16 | every agent constructs, is awaited correctly, honours its contract |

All offline, no network, no API keys.

## Stage 3: complete

Plan items A1-A3, B1-B4, C and all four verification points are closed. The
substantive conclusion is unchanged: the verdict stage is not the tractable
bottleneck, over-abstention is largely a reasonable response to inferentially
incomplete evidence, and the remaining levers are retrieval-side.

> **Amended twice. Current reading (2026-10-08, §L at n=55):** the clause *"the
> remaining levers are retrieval-side"* **holds**, and more strongly than the
> closed-corpus data suggested. Supplying live full-text evidence instead of snippets
> is worth **+0.163 accuracy** (p=0.012) and lifts the stage above its majority
> baseline, the only condition that does so.
>
> ~~Amended 2026-10-04 at n=35: the clause "no longer follows", because live and
> oracle evidence both tied the 0.800 baseline.~~ **Withdrawn.** That rested on a
> single scoring run of 35 claims; the enrichment test behind it was 3:2 discordant,
> which is no power rather than no effect.
>
> One refinement does survive from the n=35 reading: **"better evidence" is not the
> same as "the benchmark's gold evidence."** The oracle scores 0.764, *below* the
> baseline and below live full text — so terse annotated answers do not help, while
> long retrieved article text does. The lever is evidence **volume and form**, not
> closeness to gold annotation.

---

# Stage 3 revisited — verdict quality on LIVE web evidence

**Date:** 2026-10-08 (supersedes 2026-10-04 and 2026-09-28) · **55 scorable claims
of 65 collected** · replayed from cached queries, **0 search quota** · paired
McNemar on identical claims · **3 independent scoring runs**

Every Stage 3 measurement until now used one of two artificial evidence sources:
an AVeriTeC-derived closed corpus, or gold answers handed over directly. Neither
is what the system actually retrieves. This scores the verdict stage on **real web
evidence**.

> ### ⚠ This section has now been revised twice, in opposite directions. Read the history.
>
> | | n=13 | n=35 | **n=55 (current)** |
> |---|---|---|---|
> | live + full text, accuracy | 0.923 | 0.800 | **0.836** (3-run majority vote) |
> | majority-class baseline | 0.692 | 0.800 | **0.782** |
> | margin over baseline | +0.231 | 0.000 | **+0.054** |
> | enrichment (fulltext vs snippets) | "generalises" | **p=1.000** | **p=0.012** |
> | live+fulltext vs closed corpus | p=0.031 sig. | p=0.180 n.s. | **p=0.049 sig.** |
> | clean subset vs its own baseline | +0.084 | −0.046 | **+0.020** |
>
> **The n=13 version over-claimed. The n=35 version then over-corrected, and I stated
> its negative conclusions more firmly than the data supported.** §L.4 declared that
> full-text enrichment "does not replicate on live evidence" on the strength of a
> single run giving 3:2 discordant pairs. At n=55 it is 10:1, p=0.012, stable across
> three scoring runs. **That negative claim was wrong.**
>
> Two causes, different in kind:
>
> 1. **Underpowered n, as suspected.** The 20 claims added in batch 3 split 5:0 for
>    full text. The effect was always present; 35 claims could not see it.
> 2. **A measurement defect I had not accounted for.** The verdict call runs at
>    temperature 0.2 and is **not cached**, so re-scoring identical claims with
>    identical replayed evidence yields different labels. Re-scoring the *same 35
>    claims* moved the paired count from 3:2 (p=1.000) to 5:1 (p=0.219) and accuracy
>    from 0.800 to 0.857. **A single scoring run was never a measurement**, and every
>    Stage 3 figure in this document written before today is a single run. See §L.6.
>
> What has stayed consistent across all three samples is the **direction**: live
> evidence beat the closed corpus in every one (6:0, 7:2, 13:4). The p-value
> oscillates around 0.05 because n is small. **Treat the direction as the finding and
> the significance as unresolved**, rather than trusting whichever side of 0.05 the
> most recent run happened to land on.
>
> The n=13 claim about sample difficulty also still stands corrected: I reported that
> sample was "not unusually easy" because its closed-corpus score (0.462) bracketed
> the 0.544 benchmark average. At n=55 the closed corpus scores **0.673**, so the live
> sample is in fact *easier* than average.
>
> Earlier figures are kept rather than deleted, as with the withdrawn claims in
> `../evidence_retrieval/RESULTS.md` §5.

## L.1 The result at n=55, on identical claims

Live conditions use the **majority-vote label over 3 scoring runs** (§L.5);
baselines are single-run, as noted.

| condition | evidence chars | accuracy | macro-F1 | abstention |
|---|---|---|---|---|
| live web, snippets | 157 | 0.673 | 0.515 | 21.8% |
| closed corpus *(2 single runs)* | — | 0.673, 0.673 | 0.605, 0.576 | — |
| oracle (gold answers) *(2 single runs)* | ~246 | 0.764, **0.782** | 0.633, 0.645 | — |
| **majority-class baseline** | — | **0.782** | — | — |
| **live web + full text** | 1,547 | **0.836** | **0.644** | **7.3%** |

Paired McNemar, identical claims (live side = majority vote; baselines single-run, so
these two rows inherit that instability):

| comparison | base-only right | live-only right | p | |
|---|---|---|---|---|
| live+fulltext vs live snippets | 1 | **10** | **0.012** | significant |
| live+fulltext vs closed corpus | 4 | **13** | **0.049** | significant, both runs |
| live+fulltext vs oracle gold | 8 | 11–12 | 0.503–0.648 | n.s., both runs |

**Live evidence with full text is the only condition that clears the majority-class
baseline** (0.836 vs 0.782, **+0.054**), and it leads on every metric: highest
accuracy, best macro-F1 (0.644), and abstention of 7.3% against 21.8% for snippets.

**On the oracle, I have to be careful — and this is §L.5 biting immediately.** The
first baseline run gave 0.764, below the 0.782 baseline, and I drafted this section
claiming gold evidence fails to beat always-REFUTES. Re-running it gave **0.782,
exactly tying**. The oracle's macro-F1 also moved, 0.633 → 0.645. So:

* **Defensible:** the oracle sits *at* the baseline (0.764–0.782 over two runs) and
  **below live full text** (0.836) — but the paired test cannot separate oracle from
  live full text (11:8, p=0.65), so the 0.054–0.072 gap is suggestive, not shown.
* **Not defensible:** "gold answers are worse verdict input than live pages." The
  direction favours live full text in both runs, but one of those runs would have had
  me asserting a 0.072 gap that re-measurement halved.

The weaker claim that does hold: **~246 characters of gold annotation buys no more
verdict accuracy than the majority class**, while 1,547 characters of retrieved
article text buys +0.054. That is consistent with §S.4's argument that the missing
ingredient is inferential bridging material rather than factual correctness.

## L.2 The closed-corpus comparison: direction consistent, significance unresolved

Live full text beats the closed corpus on **13 claims to 4, p=0.049** — back below
0.05, having been p=0.031 at n=13 and p=0.180 at n=35.

**I am not going to call this settled a third time.** The honest summary of three
samples:

| sample | discordant (live : closed) | p |
|---|---|---|
| n=13 | 6 : 0 | 0.031 |
| n=35 | 7 : 2 | 0.180 |
| n=55 | 13 : 4 | 0.049 |

The direction is the same every time and the discordant ratio is stable at roughly
3:1. What moves is only whether that ratio clears significance at the n available.
**The defensible claim is directional**: the closed corpus understates live verdict
quality, by a margin these sample sizes cannot pin down. The magnitude claimed at
n=13 ("by a wide margin", 0.462 vs 0.923) remains withdrawn — the gap at n=55 is
0.673 vs 0.830.

**Consequence for §7.3.** This has now flipped twice, so stated carefully:

* On **live evidence with full text**, the stage scores 0.830 against a 0.782
  baseline — it **does** beat the majority class, by +0.048.
* On **live snippets** (0.673), the **closed corpus** (0.673), and **oracle gold
  evidence** (0.764), it does **not**.

So §7.3's conclusion holds for every condition except the one that reflects what a
deployed system would actually do. The n=35 version of this section said §7.3 "stands";
that was true of the data then available and is no longer true without the
qualification above.

## L.3 The leakage control — the finding survives it

| subset | n | snippets | full text | majority baseline |
|---|---|---|---|---|
| fact-check page retrieved | 21 | 0.714 | **0.857** | 0.762 |
| **no fact-check page** | **34** | **0.647** | **0.824** | **0.794** |

(Majority-vote labels; the leakage-exposed subset returned 0.714/0.857 identically in
all three runs.)

Leakage is worth **+0.033** on full text (0.857 vs 0.824) — **far less than the
+0.196 measured at n=35**, which was itself far more than the +0.12–0.14 estimated at
n=13. That earlier figure came from a 13-claim split, 6 against 7.

**On the 34 claims where no fact-checker was retrieved, full text scores 0.824
against a 0.794 baseline** — above it, and the paired advantage over snippets holds
there too (7:1, 8:1, 6:2 across runs). At n=35 this subset sat *below* its baseline,
which was the basis for concluding that leakage explained the system's apparent
competence. **It does not.** The enrichment benefit is present with the
fact-checkers removed.

Leakage is still worth controlling for — it inflates the headline by ~3 points and
affects 37% of claims — but it is no longer the dominant explanation of the result.

## L.4 Full-text enrichment DOES help on live evidence — withdrawing §L.4 at n=35

The n=35 version of this section concluded that enrichment "does not replicate on
live evidence" and recommended reading §10 as closed-corpus-specific. **Both of those
are withdrawn.**

Paired on identical claims and identical retrieved evidence, varying only whether the
top-3 documents were fetched in full:

| | n=35 (withdrawn) | **n=55 (current)** |
|---|---|---|
| evidence volume | 158 → 1,439 chars (9.1×) | 157 → 1,547 chars (**9.9×**) |
| accuracy | 0.771 → 0.800 | **0.673 → 0.836** (+0.163) |
| macro-F1 | 0.636 → 0.597 (worse) | **0.515 → 0.644** (+0.129) |
| abstention | 17.1% → 14.3% | **21.8% → 7.3%** |
| paired McNemar | 3 : 2, **p=1.000** | **10 : 1, p=0.012** |
| across 3 scoring runs | not measured | 10:1, 11:1, 9:2 (**p=0.012 / 0.006 / 0.065**) |

**Enrichment is the largest single improvement measured anywhere in this project**:
+0.163 accuracy, +0.129 macro-F1, and abstention cut by two-thirds. It is also the
most consistent — all three scoring runs agree on direction, and the majority-vote
label gives 10:1, p=0.012.

So §10's closed-corpus result (+0.081, p=0.043) **does** generalise to live web pages,
and generalises *more strongly* there (+0.163). The scope limit I added to
`../evidence_retrieval/RESULTS.md` §10 on 2026-10-04 was wrong and has been removed.

Why n=35 missed it: 3:2 is what an underpowered sample produces when the true
discordant ratio is 10:1, and re-scoring those same 35 claims gives 5:1. **I reported
p=1.000 as evidence of no effect. It was evidence of no power.**

Depth is not uniformly monotonic — 2 of 55 claims got no enriched document at all, and
fetch reliability was **143 pages fetched, 22 failed (13%)**, consistent with §9's
29%-unreachable figure given retries and snippet fallback.

## L.5 Measurement stability — the methodological finding

The verdict call runs at **temperature 0.2 and is not cached**. Three scoring runs over
identical claims and identical replayed evidence:

| condition | triple A | triple B | spread | labels unstable |
|---|---|---|---|---|
| live snippets | 0.673 / 0.655 / 0.691 | 0.655 / 0.673 / 0.691 | 0.036 | 2 / 55 (4%) |
| live full text | 0.836 / 0.836 / 0.818 | 0.836 / 0.836 / 0.836 | 0.000–0.018 | 0–1 / 55 |

Two independent triples were run. Full text returned **0.836 in five of six runs**;
snippets never repeated the same value twice within a triple.

Per-claim label instability is low (2–4%), but **at these sample sizes it is enough to
move a paired p-value across 0.05** — observed directly: 0.012, 0.006, 0.065 on the
same data.

Consequences, stated plainly:

* **Every Stage 3 figure in this document written before 2026-10-08 is a single
  scoring run**, and therefore carries an unquantified ±0.02–0.04 accuracy band.
* **Findings with one-sided discordant counts are robust to this.** §S.3's
  justification result (0/9 and 0/13) would need several labels to flip *against* the
  effect to lose significance; two or three flips would not do it.
* **Findings with near-balanced small counts are not measurements.** The n=35
  enrichment test (3:2) is the clearest case — it was within noise of 5:1.
* `live_evidence_verdict.py` now takes **`--repeat N`**, reporting per-run accuracy,
  the spread, and a **majority-vote label per claim**. The majority vote is the
  quotable figure. Future Stage 3 measurement should use it; the baselines in §L.1
  are still single-run and should be repeated next.

**It bit again while this section was being written.** The oracle baseline was
re-measured between drafts and moved **0.764 → 0.782** (macro-F1 0.633 → 0.645),
which changed §L.1 from "the oracle does not beat the baseline" to "the oracle ties
it." The closed corpus held at 0.673 accuracy but its macro-F1 moved 0.605 → 0.576.
A conclusion drafted from the first run would have been wrong, and nothing but the
re-run would have revealed it.

This is a measurement-infrastructure defect that existed from the start of Stage 3
work and went unnoticed because nothing ever re-ran the same condition twice.
## L.6 What this does and does not establish

**Establishes:**
* **Live web evidence with full text beats the majority-class baseline** — 0.836 vs
  0.782 (+0.054), the only condition measured that does, and it holds on the
  leakage-free subset (0.824 vs 0.794).
* **Full-text enrichment is the largest improvement measured anywhere in this
  project**: +0.163 accuracy, +0.129 macro-F1, abstention 21.8% → 7.3%, paired
  p=0.012 and stable across six scoring runs. §10's closed-corpus result generalises
  to the live web, and more strongly there.
* **Gold evidence is not the ceiling.** The oracle scores 0.764–0.782 over two runs
  — *at* the majority baseline, and below live full text (0.836), though the paired
  test cannot separate those two (p=0.65). AVeriTeC's ~246-character answers buy no
  more accuracy than always-REFUTES; 1,547 characters of retrieved article text buys
  +0.054. Consistent with §S.4.
* **Fact-check leakage is real but not the explanation**: +0.033 on full text,
  affecting 37% of claims. The enrichment benefit survives removing them.
* **The verdict stage is not deterministic** and single-run comparisons at these
  sample sizes are unreliable (§L.5). This is the most transferable finding here.

**Does not establish:**
* A settled magnitude for the closed-corpus gap. Three samples give p=0.031, 0.180,
  0.049 — direction consistent, significance not resolved (§L.2).
* Reliable absolute numbers. n=55, with 8 `Supported` and 4 `Not Enough Evidence`
  claims; macro-F1 and per-class recall remain unstable.
* That the §L.1 baselines are noise-free — they are still single-run and should be
  re-run with `--repeat`.
* Anything about Stage 4 (explanation), still unmeasured.

**Caveats:**
* 10 of the 65 live claims are `Conflicting/Cherrypicking` and excluded from scoring.
* The 65-claim live sample is **easier than the benchmark average** on the closed
  corpus (0.673 vs 0.544). Treat cross-sample comparisons with the n=136 figures as
  indicative only.
* The fact-check detector is domain-based and will miss fact-checking content on
  other sites, so the "no fact-check page" subset is a lower bound on leakage.
* Replayed from cached queries: the web as of 2026-09-26, 2026-10-04 and 2026-10-07.

**Next:** continue to n=100 (65 collected, ~1.4 more daily batches). The question has
changed again. At n=35 it was *"can Stage 3 beat always-REFUTES on any evidence source
at all"*; the answer is **yes, on live full-text evidence**. What is open now:

1. Does the +0.054 margin over baseline survive to n=100, and does the closed-corpus
   comparison settle on one side of 0.05?
2. Why does **enrichment help while gold answers do not**? Full text (1,547 chars)
   leads gold answers (~246 chars) by 0.054–0.072, unresolved by the paired test. If
   the mechanism is simply volume, truncating full text to gold length should erase
   the advantage — directly testable, and it would separate "more text" from "better
   text" as the active ingredient.
3. Re-run the oracle and closed-corpus baselines with `--repeat 3`, since §L.5 shows
   single-run baselines carry a ±0.02–0.04 band and both sit within that of 0.782.
## Files

- `diagnose.py` / `diagnosis_dev.json` — Phase A harness and results
- `compare_prompts.py` — prompt/reasoning arm comparison harness
- `prompt_comparison_b1_dev.json` — Phase B1 first run (original vs symmetric)
- `prompt_comparison_dev.json` — later three-arm run (direct / symmetric / synthesis)
- `evidence_sufficiency.py` / `evidence_sufficiency_dev.json` — the S.1-S.3 experiment
- `live_evidence_verdict.py` / `live_evidence_verdict_dev.json` — §L live-evidence verdicts
- `same_claim_baselines.py` / `same_claim_baselines_dev.json` — §L.1-L.2 oracle and
  closed-corpus baselines re-measured on the *same* claims, with paired McNemar.
  Re-run this after every live batch: the live sample grows, and §L shows a sample
  that looked representative at n=13 was not.
- `same_claim_baselines_dev.json` — baselines re-measured on the same 13 claims
