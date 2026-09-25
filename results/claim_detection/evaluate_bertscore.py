"""
Claim Detection — BERTScore evaluation with an EMPIRICALLY CALIBRATED threshold.

Why this script exists
----------------------
An earlier BERTScore run used RAW (un-rescaled) BERTScore with a hard cutoff of
0.85. Raw roberta-large BERTScore places two ARBITRARY fluent English sentences
at roughly 0.80-0.88, so 0.85 sits inside the noise floor and admits nearly
every candidate/reference pair. Combined with greedy 1:1 matching, that makes
`tp = min(n_pred, n_gold)` almost always, and the metric degenerates into a pure
COUNT RATIO: recall pinned near 1.0, precision determined only by how many
claims a prompt emits. Observed symptoms were recall of 0.979-1.000 (including
exactly 1.000 on the harder out-of-domain split).

This script fixes the measurement in two ways:

  1. `rescale_with_baseline=True` — re-centers scores so unrelated pairs sit
     near 0 instead of near 0.85, making the scale interpretable.

  2. Empirical threshold CALIBRATION against a HARD null distribution. Two null
     sets are measured:

       easy nulls — prediction vs gold from a DIFFERENT article. Unrelated in
         topic; establishes only the floor for completely unrelated text.
       hard nulls — two DISTINCT gold claims from the SAME article. By
         construction these are different facts sharing topic, entities and
         dates, which is precisely the confusion the matcher must avoid. No
         extra labelling is needed: the benchmark already asserts they are
         separate claims.

     The threshold defaults to the 95th percentile of the HARD nulls. Easy
     nulls proved far too permissive in practice (recall stayed pinned at 1.000
     for every cutoff up to ~0.25), which is the same count-ratio degeneration
     described above. Both distributions are reported and saved.

  3. A THRESHOLD SENSITIVITY SWEEP, computed from the same cached pair scores,
     so conclusions can be shown to hold across cutoffs rather than resting on
     one number.

It also keeps the ablation SINGLE-VARIABLE: every arm runs the base agent's
`detect_claims` and overrides ONLY the prompt, exactly as `evaluate.py` does.
All arms share `min_importance=0.0`, `normalize_detected=False`, and the base
agent's `max_tokens`, so the prompt template is the only thing that differs.

Sanity check to apply to the output
-----------------------------------
With a calibrated threshold, recall should land clearly BELOW 1.0. If recall is
still ~0.99, the threshold is still too permissive and the result is still a
count ratio, not a semantic measurement.

Usage
-----
    python results/claim_detection/evaluate_bertscore.py --split test_indomain --n 80
    python results/claim_detection/evaluate_bertscore.py --split test_oos --n 60
    # override calibration if you want a fixed cutoff:
    python results/claim_detection/evaluate_bertscore.py --threshold 0.35

Requires a torch new enough for current transformers (torch >= 2.4).
"""

import argparse
import asyncio
import json
import os
import random
import sys
from typing import Dict, List, Tuple

# Allow running from the repo root: make this file's dir importable for `prompts`,
# and the project `src/` importable for `factcheck_agent`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

# Fail loudly and usefully if the project tree isn't actually here. On Colab the
# runtime wipes /content, so it is easy to end up with this script sitting in an
# otherwise-empty directory tree.
_required = {
    "agent package": os.path.join(_ROOT, "src", "factcheck_agent", "__init__.py"),
    "claim agent": os.path.join(_ROOT, "src", "factcheck_agent", "agents", "claim_understanding.py"),
    "prompt templates": os.path.join(_HERE, "prompts.py"),
    "NewsScope dataset": os.path.join(_ROOT, "docs", "dataset", "NewsScope"),
}
_missing = {k: v for k, v in _required.items() if not os.path.exists(v)}
if _missing:
    print(f"ERROR: project files are missing under {_ROOT}\n", file=sys.stderr)
    for k, v in _missing.items():
        print(f"  MISSING {k:<18} {v}", file=sys.stderr)
    print("\nThe script is present but the project tree is not. Re-extract the "
          "project archive (on Colab, /content is wiped when the runtime "
          "restarts), then re-run.", file=sys.stderr)
    sys.exit(1)

from factcheck_agent.agents.claim_understanding import (  # noqa: E402
    ClaimUnderstandingAgent,
    ClaimDetectionError,
)
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from prompts import PROMPT_A, PROMPT_B, PROMPT_C  # noqa: E402


# --------------------------------------------------------------------------
# Ablation arm: overrides ONLY the prompt (mirrors evaluate.py's PromptAgent)
# --------------------------------------------------------------------------
class PromptAgent(ClaimUnderstandingAgent):
    """Inherits the full, fixed detect_claims and swaps in a prompt template,
    so every arm shares identical extraction logic."""

    def __init__(self, llm, prompt_template):
        super().__init__(llm=llm)
        self.prompt_template = prompt_template

    def _build_detection_prompt(self, text, min_importance):
        return self.prompt_template.format(text=text, min_importance=min_importance)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolated percentile (avoids a hard numpy dependency)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------
# Stage 1 — run detection for every arm
# --------------------------------------------------------------------------
async def run_detection(articles, arms, agents) -> Tuple[Dict, Dict, Dict]:
    """Returns (records, detection_failures, skipped).

    records[arm] = list of {"preds": [...], "golds": [...]} for scorable articles.
    """
    records = {a: [] for a in arms}
    failures = {a: 0 for a in arms}
    skipped = {a: 0 for a in arms}

    for i, article in enumerate(articles):
        ann = article["annotation"]
        headline = ann.get("headline", "")
        key_points = ann.get("key_points", [])
        gold_claims = [c["claim_text"] for c in ann.get("claims", [])]
        input_text = headline + "\n" + "\n".join(key_points)
        print(f"[{i + 1}/{len(articles)}] {headline[:45]}...")

        for arm in arms:
            try:
                predicted = await agents[arm].detect_claims(
                    input_text, min_importance=0.0, normalize_detected=False
                )
            except ClaimDetectionError as e:
                failures[arm] += 1
                print(f"  !! Arm {arm}: detection failed — {e}")
                continue

            pred_texts = [c.raw_text for c in predicted]
            if not pred_texts or not gold_claims:
                skipped[arm] += 1
                continue
            records[arm].append({"preds": pred_texts, "golds": gold_claims})

    return records, failures, skipped


# --------------------------------------------------------------------------
# Stage 2 — build every pair we need to score (real + null), in ONE batch
# --------------------------------------------------------------------------
def build_pairs(records, arms, n_null=1000, seed=0):
    """Return (pairs, real_index, slices).

    pairs: flat list of (candidate, reference) scored in ONE batched call.
    real_index: (arm, rec_i, pi, gi) aligned with the "real" slice.
    slices: {"real": (lo, hi), "easy_null": (lo, hi), "hard_null": (lo, hi)}

    Two null sets are built, because they answer different questions:

      easy_null — prediction vs gold from a DIFFERENT article. Unrelated in
        topic. Only establishes the floor for completely unrelated text.

      hard_null — two DISTINCT gold claims from the SAME article. By
        construction these are different facts sharing a topic, entities and
        dates: exactly the confusion the matcher must avoid. This needs no
        extra labelling, since the benchmark already asserts these are separate
        claims. Calibrating here is far stricter and much more honest.
    """
    pairs: List[Tuple[str, str]] = []
    real_index: List[Tuple[str, int, int, int]] = []

    for arm in arms:
        for rec_i, rec in enumerate(records[arm]):
            for pi, pred in enumerate(rec["preds"]):
                for gi, gold in enumerate(rec["golds"]):
                    pairs.append((pred, gold))
                    real_index.append((arm, rec_i, pi, gi))
    real_slice = (0, len(pairs))

    # --- easy nulls: different source articles -------------------------------
    rng = random.Random(seed)
    flat_preds, flat_golds = [], []
    for arm in arms:
        for rec_i, rec in enumerate(records[arm]):
            for p in rec["preds"]:
                flat_preds.append((rec_i, p))
            for g in rec["golds"]:
                flat_golds.append((rec_i, g))

    lo = len(pairs)
    tries, added = 0, 0
    while added < n_null and tries < n_null * 20 and flat_preds and flat_golds:
        tries += 1
        ri, pred = rng.choice(flat_preds)
        rj, gold = rng.choice(flat_golds)
        if ri == rj:
            continue
        pairs.append((pred, gold))
        added += 1
    easy_slice = (lo, len(pairs))

    # --- hard nulls: distinct gold claims from the SAME article --------------
    lo = len(pairs)
    seen = set()
    first_arm = arms[0]
    for rec in records[first_arm]:
        golds = rec["golds"]
        for i in range(len(golds)):
            for j in range(i + 1, len(golds)):
                key = (golds[i], golds[j])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((golds[i], golds[j]))
    hard_slice = (lo, len(pairs))

    return pairs, real_index, {
        "real": real_slice, "easy_null": easy_slice, "hard_null": hard_slice,
    }


# --------------------------------------------------------------------------
# Stage 3 — score, calibrate, match
# --------------------------------------------------------------------------
def match_and_score(preds, golds, score_of, threshold):
    """Greedy 1:1 matching, identical in shape to the ROUGE-L version."""
    matched_gold, matched_pred = set(), set()
    for pi in range(len(preds)):
        for gi in range(len(golds)):
            if gi in matched_gold:
                continue
            if score_of(pi, gi) >= threshold:
                matched_gold.add(gi)
                matched_pred.add(pi)
                break
    tp = len(matched_gold)
    fp = len(preds) - len(matched_pred)
    fn = len(golds) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


async def evaluate(split, n, arms, n_null, null_pct, fixed_threshold, batch_size,
                   calibrate_on="hard"):
    from bert_score import score as bert_score  # imported late: heavy

    llm = get_default_llm_client(use_dummy_if_missing_key=False)
    prompt_map = {"A": PROMPT_A, "B": PROMPT_B, "C": PROMPT_C}
    agents = {a: PromptAgent(llm=llm, prompt_template=prompt_map[a]) for a in arms}

    split_file = os.path.join(_ROOT, "docs", "dataset", "NewsScope", f"{split}.jsonl")
    with open(split_file) as f:
        articles = [json.loads(l) for l in f][:n]

    # 1) detection
    records, failures, skipped = await run_detection(articles, arms, agents)

    # 2) pairs
    pairs, real_index, slices = build_pairs(records, arms, n_null=n_null)
    if not pairs:
        print("No pairs to score — nothing was detected.")
        return
    n_real = slices["real"][1]
    print(f"\nScoring {n_real} real + {slices['easy_null'][1] - slices['easy_null'][0]} easy-null "
          f"+ {slices['hard_null'][1] - slices['hard_null'][0]} hard-null pairs "
          f"in one batched BERTScore call...")

    cands = [c for c, _ in pairs]
    refs = [r for _, r in pairs]
    _P, _R, F1 = bert_score(
        cands, refs, lang="en",
        rescale_with_baseline=True,   # <-- the critical fix
        batch_size=batch_size, verbose=True,
    )
    f1s = [float(x) for x in F1]

    def sl(name):
        lo, hi = slices[name]
        return f1s[lo:hi]

    real_f1s, easy_f1s, hard_f1s = sl("real"), sl("easy_null"), sl("hard_null")

    # 3) calibration
    calib_set = hard_f1s if calibrate_on == "hard" else easy_f1s
    if fixed_threshold is not None:
        threshold = fixed_threshold
        basis = "user-specified (--threshold)"
    elif not calib_set:
        threshold = percentile(easy_f1s, null_pct)
        basis = f"{null_pct}th pct of easy nulls (no hard nulls available)"
    else:
        threshold = percentile(calib_set, null_pct)
        basis = f"{null_pct}th percentile of {len(calib_set)} {calibrate_on}-null pairs"

    def describe(label, xs):
        if not xs:
            print(f"{label}: (none)")
            return
        print(f"{label}: n={len(xs)}")
        print(f"    mean={mean(xs):.3f}  p50={percentile(xs, 50):.3f}  "
              f"p90={percentile(xs, 90):.3f}  p95={percentile(xs, 95):.3f}  "
              f"max={max(xs):.3f}")

    print(f"\n{'=' * 64}")
    print("THRESHOLD CALIBRATION (rescaled BERTScore F1)")
    print(f"{'=' * 64}")
    describe("EASY nulls  (pred vs gold, DIFFERENT article)", easy_f1s)
    describe("HARD nulls  (two distinct golds, SAME article)", hard_f1s)
    describe("REAL pairs  (pred vs gold, same article)", real_f1s)
    print(f"\n--> threshold = {threshold:.4f}  ({basis})")
    if hard_f1s and easy_f1s:
        print(f"    hard-null p95 ({percentile(hard_f1s, 95):.3f}) vs easy-null p95 "
              f"({percentile(easy_f1s, 95):.3f}) — the gap is why easy nulls "
              f"under-estimate the cutoff.")

    # 4) match + metrics
    lookup = {key: real_f1s[i] for i, key in enumerate(real_index)}

    def metrics_at(thr):
        """Per-arm aggregates at a given threshold (pair scores already cached,
        so evaluating extra thresholds is essentially free)."""
        out = {}
        for arm in arms:
            per_article = [
                match_and_score(
                    rec["preds"], rec["golds"],
                    lambda pi, gi, _a=arm, _r=rec_i: lookup[(_a, _r, pi, gi)],
                    thr,
                )
                for rec_i, rec in enumerate(records[arm])
            ]
            out[arm] = {
                "precision": mean([r["precision"] for r in per_article]),
                "recall": mean([r["recall"] for r in per_article]),
                "f1": mean([r["f1"] for r in per_article]),
                "n": len(per_article),
            }
        return out

    headline = metrics_at(threshold)
    all_results = {a: [] for a in arms}  # kept for the detailed per-article view
    for arm in arms:
        for rec_i, rec in enumerate(records[arm]):
            all_results[arm].append(match_and_score(
                rec["preds"], rec["golds"],
                lambda pi, gi, _a=arm, _r=rec_i: lookup[(_a, _r, pi, gi)],
                threshold,
            ))

    # --- sensitivity analysis -------------------------------------------------
    # The calibration above uses CROSS-ARTICLE null pairs, which are "easy"
    # negatives. The hard negatives are same-article/different-fact pairs, which
    # we cannot label without ground truth. So rather than resting on a single
    # cutoff, report how the metrics move across a range of thresholds.
    sweep_points = sorted({round(t, 3) for t in
                           [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
                           + [round(threshold, 3)]})
    sweep = {t: metrics_at(t) for t in sweep_points}

    # 5) report
    print(f"\n{'=' * 60}")
    print(f"RESULTS — BERTScore (baseline-rescaled, calibrated) — {split} (n={len(articles)})")
    print(f"{'=' * 60}")
    print(f"{'Metric':<12}", end="")
    for a in arms:
        print(f" {'Prompt ' + a:>12}", end="")
    print()
    print("-" * 60)
    for metric in ["precision", "recall", "f1"]:
        print(f"{metric.capitalize():<12}", end="")
        for a in arms:
            print(f" {mean([r[metric] for r in all_results[a]]):>12.3f}", end="")
        print()

    print(f"\n{'Scored':<12}", end="")
    for a in arms:
        print(f" {len(all_results[a]):>12}", end="")
    print(f"\n{'Failures':<12}", end="")
    for a in arms:
        print(f" {failures[a]:>12}", end="")
    print(f"\n{'Skipped':<12}", end="")
    for a in arms:
        print(f" {skipped[a]:>12}", end="")
    print()

    # --- sensitivity table ---
    print(f"\n{'=' * 60}")
    print("THRESHOLD SENSITIVITY (F1 / recall per arm)")
    print("Threshold is calibrated on HARD nulls (distinct gold claims from the same")
    print("article). This sweep shows whether the ranking holds across cutoffs.")
    print(f"{'=' * 60}")
    print(f"{'thresh':>8}", end="")
    for a in arms:
        print(f" {a + ' F1':>9} {a + ' rec':>9}", end="")
    print()
    print("-" * 60)
    for t in sweep_points:
        mark = " *" if abs(t - round(threshold, 3)) < 1e-9 else "  "
        print(f"{t:>6.3f}{mark}", end="")
        for a in arms:
            print(f" {sweep[t][a]['f1']:>9.3f} {sweep[t][a]['recall']:>9.3f}", end="")
        print()
    print("  (* = calibrated threshold)")

    worst_recall = max(mean([r["recall"] for r in all_results[a]]) for a in arms)
    if worst_recall > 0.95:
        print("\n!! WARNING: recall is still >0.95. The threshold may still be too\n"
              "   permissive — the metric could again be behaving as a count ratio.\n"
              "   Inspect the null vs real score distributions above.")

    # 6) save
    out = os.path.join(_HERE, f"metrics_bertscore_{split}.json")
    with open(out, "w") as f:
        json.dump({
            "split": split,
            "n_articles": len(articles),
            "metric": "BERTScore F1 (roberta-large, rescale_with_baseline=True)",
            "matching": "greedy 1:1, calibrated threshold",
            "calibration": {
                "threshold": threshold,
                "basis": basis,
                "calibrated_on": calibrate_on,
                "null_percentile": None if fixed_threshold is not None else null_pct,
                "easy_null": {"n": len(easy_f1s), "mean": mean(easy_f1s),
                              "p95": percentile(easy_f1s, 95) if easy_f1s else None,
                              "max": max(easy_f1s) if easy_f1s else None},
                "hard_null": {"n": len(hard_f1s), "mean": mean(hard_f1s),
                              "p95": percentile(hard_f1s, 95) if hard_f1s else None,
                              "max": max(hard_f1s) if hard_f1s else None},
                "real_pairs": {"n": len(real_f1s), "mean": mean(real_f1s),
                               "p50": percentile(real_f1s, 50) if real_f1s else None,
                               "max": max(real_f1s) if real_f1s else None},
            },
            "results": {
                a: {
                    "precision": mean([r["precision"] for r in all_results[a]]),
                    "recall": mean([r["recall"] for r in all_results[a]]),
                    "f1": mean([r["f1"] for r in all_results[a]]),
                    "scored_articles": len(all_results[a]),
                    "detection_failures": failures[a],
                    "skipped_articles": skipped[a],
                }
                for a in arms
            },
            "threshold_sensitivity": {
                str(t): {a: {k: v for k, v in sweep[t][a].items()} for a in arms}
                for t in sweep_points
            },
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test_indomain",
                   choices=["test_indomain", "test_oos"])
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--prompts", default="A,B,C")
    p.add_argument("--n-null", type=int, default=1000,
                   help="number of unrelated pairs used to calibrate the threshold")
    p.add_argument("--null-percentile", type=float, default=95.0,
                   help="threshold = this percentile of the null distribution")
    p.add_argument("--threshold", type=float, default=None,
                   help="skip calibration and use this fixed threshold")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--calibrate-on", choices=["hard", "easy"], default="hard",
                   help="hard = distinct gold claims from the SAME article "
                        "(strict, default); easy = cross-article pairs")
    args = p.parse_args()

    arms = [a.strip() for a in args.prompts.split(",")]
    asyncio.run(evaluate(args.split, args.n, arms, args.n_null,
                         args.null_percentile, args.threshold, args.batch_size,
                         args.calibrate_on))
