# Stage 2 — Evidence Retrieval: AVeriTeC Closed-Corpus Evaluation

**Date:** 2026-09-25
**Dataset:** AVeriTeC `dev` (500 claims)
**Condition:** offline closed corpus — **not comparable to live open-web search**
**Headline:** the retrieval loop's benefit is **entirely explained by retrieving
more documents**. Query reformulation adds no measurable value over simply taking
more results from the first ranked list, so the cheaper `k` increase is preferable.

> **Correction.** An earlier version of this document claimed the loop was
> "strictly dominated" by a single pass at `k=9`. That rested on an **unpaired
> n=50** comparison. A **paired n=200** test (§5) finds the two conditions
> statistically indistinguishable (recall Δ=+0.020, 95% CI [−0.017, +0.059],
> p=0.43). The dominance claim was an artifact of sample size and is withdrawn.
> A second claim — that the loop increases fact-check leakage — is also withdrawn
> (§5.4).

---

## 1. Summary

The retrieval loop was repaired in Workstream C: it previously passed an identical
claim to `retrieve_evidence` every round, so URL dedup guaranteed round 2 added
nothing after spending an LLM call and up to 15 searches. It now genuinely
iterates — 148 of 200 evaluated claims trigger a gap-driven second round.

**But a working loop is not the same as a useful one.** Both the loop and a
larger single-pass `k` significantly beat the `k=5` baseline, and they are
statistically equivalent to each other at matched retrieval volume. Because the
loop costs two extra LLM calls and an extra search round per claim for no
measurable gain, raising `k` is the better engineering choice.

---

## 2. Dataset scoping

AVeriTeC per claim: `claim`, `label`, `justification`, `fact_checking_article`,
and `questions[] -> answers[] -> {answer, answer_type, source_url, source_medium}`.

### 2.1 `test.json` is unusable

All **2215** records have `label: None` and **zero** questions — it is the blind
shared-task split. Evaluation must use `dev` (500, fully annotated) or `train`
(3068).

### 2.2 Wayback unwrapping is mandatory

**31.9%** of dev gold `source_url`s are `web.archive.org/web/<ts>/<original>`
snapshots, and one is double-wrapped. Live search never returns archive.org URLs,
so comparing raw strings would report near-zero source recall for purely cosmetic
reasons. `averitec.unwrap_url` strips up to three layers.

### 2.3 The scorable subset — 391 of 500 claims

Not every gold answer is retrievable by a text pipeline, and scoring against ones
that aren't would understate performance for reasons unrelated to retrieval:

| Excluded | Share of dev answers | Why |
|---|---|---|
| `Boolean` answers | ~21% | The answer is "Yes"/"No"; no passage can semantically cover it |
| `Unanswerable` | 39 answers | Nothing exists to find |
| Non-`Web text` media (PDF, Image/graphic, Video, Web table, Metadata) | ~25% | Not retrievable/extractable by a text pipeline |
| Empty or non-URL source | ~9% | No target to match |

**Scorable = Extractive/Abstractive + Web text + usable http URL.**

| | count |
|---|---|
| claims | 500 |
| claims with ≥1 scorable answer (**evaluated**) | **391** |
| answers total | 1399 |
| answers scorable | **820 (58.6%)** |
| label distribution | Refuted 305, Supported 122, NEE 35, Conflicting/Cherrypicking 38 |

Every run prints these counts so the denominator is never silently narrowed.

---

## 3. Method

A single index is built from **973 distinct source URLs** across all dev answers
(body = the annotated answer text; questions are deliberately *not* indexed as
titles, since they are written from the claim and would leak the target). For
each claim the gold set is the URLs of its own scorable answers; every other
document is a distractor. The retriever sees only the claim.

Retrieval runs through `FactCheckingPipeline.collect_evidence()` — the real
production loop, extracted so evaluation does not pay for verdict and explanation
generation. Search is served by `CorpusSearchClient` (IDF-weighted overlap), so
different queries genuinely surface different documents and a reformulated query
*can* reach something the first missed.

Settings: 50 claims, 5 results/query, LLM query generation + sufficiency
evaluation enabled, LLM normalization disabled (AVeriTeC claims are already
standalone, so it is a near no-op and only adds a confound).

**This measures closed-corpus retrieval recall, not evidence coverage.** The
corpus is guaranteed to contain the answer, so these numbers are easier than live
open-web search and must be reported as a distinct condition.

---

## 4. Pilot run (n=50, unpaired) — superseded

| condition | recall | precision | any-hit | retrieved |
|---|---|---|---|---|
| rounds=1, k=5 | 0.589 | 0.192 | 0.700 | 5.00 |
| rounds=2, k=5 | 0.649 | 0.136 | 0.760 | 8.54 |
| rounds=1, k=9 (volume control) | 0.716 | 0.144 | 0.820 | 8.44 |

This pilot established the essential control: `rounds=2` retrieved **70% more
documents** than `rounds=1`, and recall is monotonic in documents retrieved, so
the raw comparison could not distinguish better targeting from more volume. That
reasoning still stands, and it is the same trap as the Stage 1 BERTScore
count-ratio artifact.

Its *conclusion*, however, does not: comparing means across independent runs
cannot separate a real effect from per-claim variance. See §5.

---

## 5. Paired analysis (n=200) — primary result

Same 200 claims under every condition; per-claim scores retained; compared with a
paired Wilcoxon signed-rank test and a paired bootstrap 95% CI.

| condition | recall | precision | any-hit | retrieved |
|---|---|---|---|---|
| rounds=1, k=5 | 0.475 | 0.166 | 0.610 | 5.00 |
| rounds=2, k=5 | 0.556 | 0.129 | 0.685 | 8.22 |
| rounds=1, k=9 | 0.576 | 0.124 | 0.695 | 8.48 |

### 5.1 At matched volume, the loop and larger `k` are equivalent

`1round_k9` − `2round_k5`:

| metric | delta | 95% CI | p | W/L/T |
|---|---|---|---|---|
| recall | +0.020 | [−0.017, +0.059] | 0.43 | 19/11/170 |
| any-hit | +0.010 | [−0.035, +0.055] | 0.65 | 11/9/180 |
| precision | −0.005 | [−0.015, +0.004] | 0.97 | 80/47/73 |

**No significant difference on any metric.** The pilot's 0.067 recall gap shrank
to 0.020 with a confidence interval spanning zero.

### 5.2 Both interventions beat the k=5 baseline

| comparison | recall | p | any-hit | p | precision | p |
|---|---|---|---|---|---|---|
| rounds=2,k=5 − rounds=1,k=5 | **+0.081** | 2e-05 | **+0.075** | 0.001 | −0.037 | ~0 |
| rounds=1,k=9 − rounds=1,k=5 | **+0.101** | ~0 | **+0.085** | 0.0001 | −0.042 | ~0 |

Both trade precision for recall by similar amounts. With only **1.70** gold
documents per claim, precision is structurally capped at `1.70 / k`, so raw
precision must never be compared across conditions retrieving different volumes.

### 5.3 Why round 2 underperforms per document

Instrumented over the 148 claims that ran a second round:

| | round 1 | round 2 | hypothesis |
|---|---|---|---|
| claim-token overlap | 0.448 | **0.322** | **H1 vocabulary drift — confirmed** |
| query length (tokens) | 7.47 | 9.10 | longer, more specific |
| gold rate of new documents | 0.167 | **0.040** | **H2 picked-over pool — confirmed** |
| mean corpus rank of new docs | 2.48 | 3.26 | **H2 — confirmed** |
| fact-check documents per round | 0.14 | **0.08** | **H3 leakage steer — refuted** |

Round 2 is **more shots on goal at worse odds**: its documents are 4× less likely
to be gold, drawn from a lower-ranked pool after round 1's URLs are excluded, and
its queries have drifted ~28% away from the claim's vocabulary. It still raises
recall, but only by adding volume — which `k=9` achieves more cheaply from the
better-ranked first list.

### 5.4 Withdrawn: the leakage claim

The pilot noted the claim's own `fact_checking_article` domain being retrieved for
3 / 7 / 5 claims (k=5 / loop / control) and inferred that gap statements steer
toward fact-checking content. The paired test does not support this: the loop
difference is **not significant** (Δ=+0.020, p=0.157), and the per-round
diagnostic shows round 2 retrieving **fewer** fact-check documents than round 1
(0.08 vs 0.14). The pilot's increase was volume, not steering.

Fact-check leakage remains worth monitoring — `EXCLUDE_DOMAINS` and the logged
hit rate exist for that — but this evaluation provides **no evidence that the
loop worsens it**.

---

## 6. What this does and does not establish

**Does not mean the loop fix was wrong.** Before Workstream C the loop was a
no-op that burned quota on discarded results. It now genuinely iterates. That
repair is real and independently verified by 33 behavioural tests.

**Does mean the loop is not worth its cost in this condition.** It is
*equivalent* to a larger single pass, not worse — but equivalence obtained for
two extra LLM calls and an extra search round per claim is a losing trade. This
is a cost-effectiveness argument, not a claim that reformulation is harmful.

**Does not mean reformulation is useless in general.** See caveat 1.

**Three caveats before generalising:**

1. **Closed corpus, lexical retrieval.** Every document sits in one 973-entry IDF
   index. Query reformulation cannot reach anything the first query could not —
   exactly where a loop should pay off on the open web. This setup structurally
   disadvantages the loop, and is the single biggest reason not to retire it on
   this evidence alone.
2. **One corpus, one retrieval model.** Results may differ with a dense/embedding
   retriever, where reformulation changes the embedding neighbourhood rather than
   just the token overlap.
3. **Round 2 excludes already-seen URLs**, forcing it into a lower-ranked pool
   (§5.3). A different exclusion policy — for example allowing re-ranking of
   previously seen documents — might change the result.

---

## 7. k sweep — retrieval saturation and verdict impact

150 claims, same set at every k, rounds=1, verdicts scored against AVeriTeC labels.

| k | evidence delivered | recall | Δrecall | precision | any-hit | verdict acc | macro-F1 |
|---|---|---|---|---|---|---|---|
| 5 | 5.0 | 0.550 | — | 0.188 | 0.687 | 0.566 | 0.486 |
| 7 | 6.9 | 0.582 | +0.033 | 0.152 | 0.713 | 0.544 | 0.478 |
| 9 | 8.4 | 0.604 | +0.021 | 0.129 | 0.740 | 0.559 | 0.481 |
| **12** | 9.7 | **0.641** | +0.038 | 0.125 | **0.767** | 0.559 | **0.499** |
| 15 | 9.8 | 0.629 | −0.012 | 0.123 | 0.760 | 0.529 | 0.467 |

### 7.1 The real bottleneck is the candidate pool, not k

**Evidence delivered saturates at ~9.8 documents.** At k=12 the pipeline returns
9.7 and at k=15 it returns 9.8 — raising k further is a no-op, because there are
no more unique candidates to select from.

The ceiling is set upstream: `max_queries=3` plus the auto-appended refutation
query gives 4 queries × `max_results_per_query=5` = at most 20 results, which
dedupe to roughly 10 unique URLs. **To retrieve more evidence, raise
`max_results_per_query` or `max_queries` — raising `MAX_EVIDENCE_SOURCES` beyond
~12 cannot help.**

Recall accordingly saturates near 0.64. The −0.012 dip at k=15 is noise (the same
9.7 vs 9.8 documents).

### 7.2 Extra evidence neither helps nor hurts the verdict

Verdict accuracy across the whole sweep spans **0.529–0.566, a range of 0.037**,
against a binomial 95% band of **±0.083** at n=136. Every difference is within
noise, and macro-F1 is equally flat (0.467–0.499).

So the answer to "does the added low-precision evidence help or hurt the
verdict?" is **neither**. Retrieval recall rose from 0.550 to 0.641 across the
sweep without moving verdict quality at all. The recall gained by raising k does
not convert into better verdicts in this condition.

### 7.3 The verdict stage underperforms a constant predictor

This is the more serious finding, and it is about Stage 3, not Stage 2.

| predictor | accuracy | macro-F1 |
|---|---|---|
| always predict REFUTES | **0.706** | 0.276 |
| this system (best k) | 0.566 | **0.499** |

96 of the 136 mappable claims are `Refuted`, so a constant predictor scores 70.6%.
**The system scores 56.6% — below the prior.** It does beat that baseline on
macro-F1 (0.499 vs 0.276), because it distributes across classes rather than
collapsing to one, and macro-F1 rewards that. Both facts belong in any write-up:
accuracy alone flatters the constant predictor, macro-F1 alone flatters ours.

Predicted label distribution at k=5 (n=136, gold ≈ 96 Refuted / ~30 Supported /
~10 Not Enough Evidence):

| predicted | count | share |
|---|---|---|
| NOT_ENOUGH_INFO | 64 | 47% |
| REFUTES | 61 | 45% |
| SUPPORTS | 11 | 8% |

**The system over-predicts NOT_ENOUGH_INFO roughly sevenfold** (47% predicted vs
~7% gold) and under-predicts both other classes. Notably it also **under-refutes**
(45% predicted vs 71% gold) — the opposite of what the documented refutation bias
was assumed to cause. On this data the problem is excessive abstention, not
excessive refutation.

### 7.4 Why these verdict numbers are a lower bound

The closed corpus serves **200-character snippets** and this sweep runs with
`full_text_top_k=0`, so the verdict stage sees far thinner evidence than the live
pipeline would. Abstaining on a 200-character fragment is defensible behaviour.
These numbers therefore bound the verdict stage from below and should not be
reported as its performance on real evidence. What they *do* establish, cleanly,
is the **flatness across k** — that comparison is internally valid because every
condition saw the same kind of evidence.

### 7.5 Revised practical recommendation

`MAX_EVIDENCE_SOURCES=5` still leaves recall on the table: k=12 delivers
**recall 0.641 vs 0.550** and **any-hit 0.767 vs 0.687**, at no extra search cost
and — per §7.2 — no verdict penalty. Above 12 there is nothing left to select.

But the larger lever is upstream: the candidate pool caps everything at ~10
documents. Raising `max_results_per_query` from 5 is the change most likely to
move retrieval further, and it is untested.

---

## 8. Error propagation — is the deficit retrieval or verdict?

150 claims, same set, k=12, rounds=1. Four conditions isolate the cause of the
over-abstention found in §7.3.

| condition | evidence | chars/snippet | recall | accuracy | macro-F1 | abstention |
|---|---|---|---|---|---|---|
| A baseline (5/query, 200-char) | 9.7 | 187 | 0.617 | 0.544 | 0.475 | 45.6% |
| B bigger pool (10/query) | 12.0 | 184 | **0.652** | 0.551 | 0.511 | 49.3% |
| **C thicker snippets (500-char)** | 9.8 | **375** | 0.639 | **0.662** | **0.569** | **36.8%** |
| **D ORACLE (gold evidence)** | **2.0** | 246 | 1.000 | **0.750** | **0.638** | 27.9% |

### 8.1 Evidence DEPTH beats evidence BREADTH, decisively

Doubling snippet length (C) raised accuracy **0.544 → 0.662** and cut abstention
**45.6% → 36.8%**, with essentially unchanged retrieval (9.8 documents, recall
0.639 vs 0.617).

Widening the candidate pool (B) did the opposite of what was hoped: recall rose
to 0.652 — the best retrieval of any non-oracle condition — but accuracy barely
moved (0.551, within noise) and **abstention went up** to 49.3%. Precision fell
0.122 → 0.095, so the extra documents were mostly noise, and more noise produced
more abstention.

**More evidence does not help; better evidence does.** This settles the §7.1
question about raising `max_results_per_query`: it improves retrieval metrics and
does not improve verdicts, so it is not worth doing.

It also retroactively validates Workstream D's full-text enrichment, which was
built but never measured end-to-end: serving longer passages is the single
highest-value change found so far.

### 8.2 The oracle: perfect evidence, 5× less of it

The oracle condition wins on **2.0 evidence items**, versus 9.7–12.0 for the
retrieved conditions. Two precise, relevant fragments beat a dozen noisy ones by
a wide margin. Precision dominates volume for verdict quality.

Accuracy decomposition (n=136):

| step | accuracy | gain |
|---|---|---|
| baseline retrieved evidence | 0.544 | — |
| + thicker snippets | 0.662 | +0.118 |
| + perfect evidence | 0.750 | +0.088 |
| + perfect verdict stage | 1.000 | **+0.250** |

Of the headroom above baseline, **26% is recoverable by evidence depth alone**,
a further chunk by evidence quality — and the **largest single remaining block,
0.250, belongs to the verdict stage.**

### 8.3 Confirmed: this is a Stage 3 problem too

**The verdict stage over-abstains even on gold evidence: 27.9%, against a gold
`Not Enough Evidence` rate of ~6% (8 of 136).** Retrieval improvements cannot fix
that. The oracle does clear the majority-class baseline (0.750 vs 0.706), but by
only +0.044.

### 8.4 Per-class recall locates the failure precisely

| condition | SUPPORTS (n=32) | REFUTES (n=96) | NOT_ENOUGH_INFO (n=8) |
|---|---|---|---|
| A baseline | 0.41 | 0.58 | 0.62 |
| C thicker | 0.50 | 0.70 | 0.88 |
| **D oracle** | **0.56** | 0.79 | 1.00 |

**`SUPPORTS` is the weakest class in every condition**, and remains the weakest
even with perfect evidence (0.56). The system is systematically poor at
*confirming* true claims, defaulting to abstention instead.

This points at a specific, asymmetric rule in
`reasoning_and_verdict.py:85-96`:

```python
if label == "SUPPORTS":
    if confidence < 0.5:
        label = "NOT_ENOUGH_INFO"     # demoted
elif label == "REFUTES":
    if confidence < 0.4:
        reasoning = f"Low confidence refutation: {reasoning}"   # kept
```

A low-confidence `SUPPORTS` is **converted into `NOT_ENOUGH_INFO`**; a
low-confidence `REFUTES` is **kept**, with only a note added. That asymmetry
would produce exactly the observed signature: suppressed `SUPPORTS` recall and
inflated abstention. It is a hypothesis consistent with the data, not yet a
demonstrated cause — the test is to disable the demotion and re-run the oracle
condition.

Note this cuts against the assumption behind freezing the verdict logic. The
`IMPROVEMENT_PLAN` rules were added to *reduce* a ~70% `NOT_ENOUGH_INFO` skew;
on AVeriTeC the system abstains on 45.6% of claims and **under-refutes** (44% of
predictions vs 71% gold), so at least one of those rules appears to be working
against its own goal.

---

## 9. Fetchability — can full-text enrichment reach real evidence?

§8.1 established that evidence **depth** is the highest-value lever, but proved it
with longer *corpus snippets*. It never exercised `ContentFetcher`, the machinery
that would deliver depth in production. Depth is only worth building on if the
pages are actually reachable.

150 gold source URLs from AVeriTeC dev, fetched live. No LLM calls, no search
quota — HTTP only, concurrency 4.

### 9.1 Outcomes

| outcome | n | share |
|---|---|---|
| **ok** | **106** | **70.7%** |
| http 4xx (bot-block / dead) | 30 | 20.0% |
| connection error | 11 | 7.3% |
| http 5xx | 2 | 1.3% |
| extraction failed | 1 | 0.7% |

**Fetch success rate: 70.7%.** Roughly three in ten evidence URLs cannot be
retrieved at all, which makes the snippet fallback in `ContentFetcher` load-bearing
rather than defensive: without it, ~30% of sources would simply be lost.

### 9.2 Depth delivered

| | median |
|---|---|
| extracted article text | **2,936 chars** |
| gold answer (what a snippet carries) | 172 chars |
| **depth multiple** | **≈17×** |

Mean extracted length is 10,026 chars with a maximum of **157,392**, so the
`MAX_SOURCE_CHARS` cap and claim-relevant passage selection are necessary, not
optional — an unfiltered page would swamp the verdict prompt.

Extraction used trafilatura for 101 of 106 successes, falling back to
BeautifulSoup for 5. (trafilatura was declared in `pyproject.toml` but not
installed until this session; without it every extraction would have been
boilerplate-laden bs4 output.)

Note the scale gap: condition C in §8.1 raised accuracy +0.118 using **500-char**
snippets. Real full text offers ~2,900 chars. The §8.1 result therefore measures
a *fraction* of what enrichment could deliver — though more text is not
automatically better, and that remains untested.

### 9.3 Archived copies are more reliable than live ones

| URL origin | success rate |
|---|---|
| Wayback-origin | **80.7%** (n=57) |
| plain live URL | 64.5% (n=93) |

Counterintuitive but consistent: archives do not bot-block, paywall, or rot. The
audit's opportunistic Wayback retry recovered **16 of the 106 successes (15%)**.

Domains failing most: `archive.ph` (7 — a different archive service that blocks
automated access), `englandandcompany.co.uk` (4), `businessinsider.com` (4),
`cdc.gov` (3), `apnews.com` (2).

### 9.4 Consequence for the code

`ContentFetcher` did **not** attempt a Wayback retry; the audit script added that
logic ad hoc. Given a 15% recovery rate, it belongs in the fetcher itself.

> **Status:** `wayback_fallback` has been added to `ContentFetcher` and
> **verified** — 6 dedicated assertions in `tests/test_stage2_behaviour.py`
> (43 checks total, all passing) confirm it retries the archive only on failure,
> never re-wraps an already-archived URL, never loops, can be disabled, and is
> skipped entirely when the live fetch succeeds. All fetches in those tests are
> stubbed, so the suite needs no network access.

### 9.5 What this does not tell us

The audit measures *reachability and volume*, not usefulness. Whether ~2,900
characters of article text produces better verdicts than a 200-character snippet
is the next experiment, not a conclusion from this one.

---

## 10. Next steps

1. ~~Paired comparison~~, ~~round-2 diagnostics~~, ~~`k` sweep~~ — **done** (§5, §7).
2. ~~Raise `max_results_per_query`~~ — **done and rejected** (§8.1): improves
   retrieval, does not improve verdicts, increases abstention.
3. **Enable full-text enrichment (`FULL_TEXT_TOP_K > 0`) and measure it.** §8.1
   shows evidence depth is the highest-value lever found so far. The machinery
   exists (Workstream D) and has never been measured end-to-end.
4. **Test the SUPPORTS-demotion hypothesis** (§8.4): disable the
   `SUPPORTS -> NOT_ENOUGH_INFO` rule and re-run the oracle condition. One-line
   change, decisive result. Requires unfreezing the verdict logic.
5. **Test the loop live.** The closed corpus cannot expose sources a reformulated
   query would newly reach.
6. **Try a dense retriever** before concluding reformulation adds nothing.

## Files

- `averitec.py` — loader, Wayback unwrapping, scorable-subset definition
- `fetchability_audit.py` / `fetchability_dev.json` — §9 reachability audit
- `evaluate_retrieval.py` — harness (conditions, volume control, metrics)
- `metrics_retrieval_dev.json` — saved results for the runs above
