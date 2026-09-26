"""
Stage 3 Phase A: diagnose the verdict stage. No behaviour change.

Two questions, one run (they need the same data, so combining saves LLM calls).

A2 — CALIBRATION. Saved Stage 2 data shows the stage emitting ~0.91 mean
confidence while being ~0.55 accurate, and that confidence barely moved (0.910 ->
0.901) across a 0.09 swing in retrieval recall. Per-claim confidences were never
saved, only aggregates, so this measures the distribution directly: ECE, Brier
score, a reliability table, and crucially **what fraction of confidences fall
below the 0.5 / 0.4 thresholds**.

That last number settles a pre-registered hypothesis. §8.4 of the Stage 2 results
blamed the SUPPORTS deficit on the rule at `reasoning_and_verdict.py:85-96`, which
demotes a low-confidence SUPPORTS to NOT_ENOUGH_INFO. But the rule only fires
below 0.5. If the model almost always emits ~0.9, **the rule can rarely fire and
cannot be the cause** — and §8.4 must be withdrawn.

A3 — WHERE SUPPORTS IS LOST. For every gold-`Supported` claim the system gets
wrong, record what it predicted, the model's RAW label and confidence before any
rule, and whether a rule rewrote it. That separates three candidate causes:
  1. the prompt asymmetry (model itself never says SUPPORTS)
  2. the threshold rewrite (model says SUPPORTS, the rule destroys it)
  3. genuine evidence insufficiency

Runs under the ORACLE condition — gold evidence fed straight to the verdict stage
— so retrieval quality is removed entirely and anything observed belongs to
Stage 3.

Usage:
    python results/verdict/diagnose.py --limit 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "results", "evidence_retrieval"))

from averitec import AveritecClaim, default_data_dir, evaluable, load_split  # noqa: E402
from error_propagation import gold_evidence  # noqa: E402  (the oracle pattern)
from sweep_k import CLASSES, LABEL_MAP, macro_f1  # noqa: E402

from factcheck_agent.agents.reasoning_and_verdict import (  # noqa: E402
    LOW_CONFIDENCE_WARN, ReasoningAndVerdictAgent,
)

# The thresholds this script was written to investigate have since been REMOVED
# from the agent, on the strength of the finding below: they fired on 0 of 150
# decisions because the model's confidence floor is 0.8. The historical values are
# kept here so the "could they ever have fired?" analysis still runs and remains
# reproducible.
HISTORICAL_SUPPORTS_MIN = 0.5
HISTORICAL_REFUTES_MIN = 0.4
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from factcheck_agent.models import Claim  # noqa: E402

BUCKETS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def ece(rows: List[Dict], n_buckets: int = 10) -> float:
    """Expected Calibration Error: mean |accuracy - confidence| weighted by bin size."""
    total, error = len(rows), 0.0
    if not total:
        return 0.0
    for i in range(n_buckets):
        lo, hi = i / n_buckets, (i + 1) / n_buckets
        bin_rows = [r for r in rows if lo <= r["confidence"] < hi or (i == n_buckets - 1 and r["confidence"] == 1.0)]
        if not bin_rows:
            continue
        acc = sum(1 for r in bin_rows if r["correct"]) / len(bin_rows)
        conf = statistics.mean([r["confidence"] for r in bin_rows])
        error += (len(bin_rows) / total) * abs(acc - conf)
    return error


def brier(rows: List[Dict]) -> float:
    """Brier score treating confidence as P(prediction is correct)."""
    if not rows:
        return 0.0
    return statistics.mean([(r["confidence"] - (1.0 if r["correct"] else 0.0)) ** 2 for r in rows])


async def run(claims: List[AveritecClaim], llm, concurrency: int):
    agent = ReasoningAndVerdictAgent(llm=llm, record_decisions=True)
    sem = asyncio.Semaphore(concurrency)
    rows: List[Dict] = []

    async def one(claim: AveritecClaim):
        evidence = gold_evidence(claim)
        if not evidence:
            return
        async with sem:
            verdict = await agent.decide(
                Claim(id=claim.claim_id, raw_text=claim.claim), evidence
            )
        gold = LABEL_MAP.get(claim.label)
        rows.append({
            "claim_id": claim.claim_id,
            "gold_raw": claim.label,
            "gold": gold,
            "pred": verdict.label,
            "confidence": verdict.confidence,
            "correct": gold == verdict.label if gold else None,
            "n_evidence": len(evidence),
        })

    await asyncio.gather(*(one(c) for c in claims))

    # join the agent's decision log (raw label/confidence pre-rule) onto the rows
    by_id = {d["claim_id"]: d for d in agent.decision_log}
    for r in rows:
        d = by_id.get(r["claim_id"], {})
        r["raw_label"] = d.get("raw_label")
        r["raw_confidence"] = d.get("raw_confidence")
        r["rule_fired"] = d.get("rule_fired")
    return agent, rows


def main_report(agent, rows) -> Dict:
    scored = [r for r in rows if r["gold"]]
    pairs = [(r["gold"], r["pred"]) for r in scored]
    acc = sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0
    gold_counts = Counter(g for g, _ in pairs)
    majority = max(gold_counts.values()) / len(pairs) if pairs else 0

    print(f"{'=' * 86}\nSTAGE 3 PHASE A — VERDICT DIAGNOSIS (oracle evidence)\n{'=' * 86}")
    print(f"claims scored: {len(scored)} (of {len(rows)} run; "
          f"{len(rows) - len(scored)} excluded as Conflicting/Cherrypicking)")
    print(f"accuracy={acc:.4f}  macro-F1={macro_f1(pairs):.4f}  majority-baseline={majority:.4f}")
    print(f"predicted: {dict(Counter(p for _, p in pairs))}")
    print(f"gold:      {dict(gold_counts)}")

    # ---------------- the decisive number ----------------
    st = agent.stats()
    print(f"\n{'=' * 86}\nCOULD THE (NOW-REMOVED) DEMOTION RULE EVER FIRE?\n{'=' * 86}")
    print("  NOTE: the rules were removed after this analysis found 0/150 firings.")
    print("  This section re-derives that from the raw confidences the model emits.")
    print(f"  decisions                      {st['decisions']}")
    print(f"  confidences below {LOW_CONFIDENCE_WARN} (logged)  {st['low_confidence_seen']}"
          f"   ({st['low_confidence_rate']:.1%} of decisions)")
    print(f"  parse failures                 {st['parse_failures']}  ({st['parse_failure_rate']:.1%})")
    print(f"  LLM errors                     {st['llm_errors']}")
    print(f"  label coerced (off-menu)       {st['label_coerced']}")
    print(f"  confidence field missing       {st['missing_confidence']}")
    would_have = sum(1 for r in rows if (r.get("raw_confidence") or 1.0) < HISTORICAL_SUPPORTS_MIN)
    print(f"  would have been demoted under the old rule: {would_have}")
    if would_have == 0:
        print("\n  --> The rule could never have fired. §8.4's hypothesis is refuted:")
        print("      the demotion cannot explain the SUPPORTS deficit.")

    # ---------------- calibration ----------------
    conf = [r["confidence"] for r in scored]
    raw = [r["raw_confidence"] for r in scored if r["raw_confidence"] is not None]
    print(f"\n{'=' * 86}\nCALIBRATION\n{'=' * 86}")
    print(f"  mean confidence {statistics.mean(conf):.4f}   vs accuracy {acc:.4f}"
          f"   -> overconfidence gap {statistics.mean(conf) - acc:+.4f}")
    print(f"  median {statistics.median(conf):.3f}  min {min(conf):.3f}  max {max(conf):.3f}")
    print(f"  ECE {ece(scored):.4f}   Brier {brier(scored):.4f}")
    print(f"\n  raw confidence below the historical thresholds:")
    print(f"    < {HISTORICAL_SUPPORTS_MIN} : {sum(1 for c in raw if c < HISTORICAL_SUPPORTS_MIN)}"
          f" / {len(raw)}")
    print(f"    < {HISTORICAL_REFUTES_MIN} : {sum(1 for c in raw if c < HISTORICAL_REFUTES_MIN)}"
          f" / {len(raw)}")
    print(f"\n  reliability:")
    print(f"    {'bucket':>12} {'n':>5} {'mean conf':>10} {'accuracy':>9}")
    for lo, hi in BUCKETS:
        b = [r for r in scored if lo <= r["confidence"] < hi]
        if not b:
            continue
        print(f"    {f'{lo:.1f}-{hi:.1f}':>12} {len(b):>5} "
              f"{statistics.mean([r['confidence'] for r in b]):>10.3f} "
              f"{sum(1 for r in b if r['correct']) / len(b):>9.3f}")

    # ---------------- where SUPPORTS is lost ----------------
    supported = [r for r in scored if r["gold"] == "SUPPORTS"]
    missed = [r for r in supported if not r["correct"]]
    print(f"\n{'=' * 86}\nWHERE IS SUPPORTS LOST?  (gold=Supported, n={len(supported)})\n{'=' * 86}")
    print(f"  correct: {len(supported) - len(missed)}   missed: {len(missed)}")
    if missed:
        print(f"  what they were predicted as: {dict(Counter(r['pred'] for r in missed))}")
        print(f"  what the MODEL raw-said:     {dict(Counter(r['raw_label'] for r in missed))}")
        rules = Counter(r["rule_fired"] for r in missed)
        print(f"  rule rewrote the answer:     {dict(rules)}")
        blamed_rule = sum(1 for r in missed if r["rule_fired"] is not None)
        blamed_model = sum(1 for r in missed if r["raw_label"] != "SUPPORTS")
        print(f"\n  attribution of the {len(missed)} misses:")
        print(f"    model never said SUPPORTS (prompt/model) : {blamed_model}")
        print(f"    model said SUPPORTS, rule destroyed it   : {blamed_rule}")
        print(f"    other                                    : {len(missed) - blamed_model - blamed_rule}")

    per_class = {}
    for cls in CLASSES:
        n = sum(1 for g, _ in pairs if g == cls)
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        per_class[cls] = {"n": n, "recall": round(tp / n, 3) if n else None}
    print(f"\n  per-class recall: " + "  ".join(
        f"{c}={v['recall']}(n={v['n']})" for c, v in per_class.items()))

    return {
        "claims_scored": len(scored),
        "accuracy": acc, "macro_f1": macro_f1(pairs), "majority_baseline": majority,
        "counters": st,
        "calibration": {
            "mean_confidence": statistics.mean(conf), "median": statistics.median(conf),
            "min": min(conf), "max": max(conf),
            "overconfidence_gap": statistics.mean(conf) - acc,
            "ece": ece(scored), "brier": brier(scored),
            "raw_below_supports_threshold": sum(1 for c in raw if c < HISTORICAL_SUPPORTS_MIN),
            "raw_below_refutes_threshold": sum(1 for c in raw if c < HISTORICAL_REFUTES_MIN),
            "raw_n": len(raw),
        },
        "supports_analysis": {
            "n_gold_supported": len(supported),
            "missed": len(missed),
            "predicted_as": dict(Counter(r["pred"] for r in missed)),
            "model_raw_said": dict(Counter(r["raw_label"] for r in missed)),
            "rule_rewrote": sum(1 for r in missed if r["rule_fired"] is not None),
            "model_never_said_supports": sum(1 for r in missed if r["raw_label"] != "SUPPORTS"),
        },
        "per_class_recall": per_class,
        "pred_distribution": dict(Counter(p for _, p in pairs)),
        "gold_distribution": dict(gold_counts),
    }


async def main_async(args) -> int:
    claims = evaluable(load_split(os.path.join(args.data_dir, f"{args.split}.json")))[: args.limit]
    llm = get_default_llm_client(use_dummy_if_missing_key=False)
    agent, rows = await run(claims, llm, args.concurrency)
    summary = main_report(agent, rows)

    os.makedirs(_HERE, exist_ok=True)
    # Guard: a small smoke run must not clobber the authoritative results file.
    # This happened once — `--limit 4` overwrote the n=136 run and the figures
    # cited in FINDINGS.md and ACADEMIC_DOCUMENTATION.md became untraceable until
    # they were recovered from a log.
    suffix = "" if args.limit >= 100 else f"_n{len(rows)}"
    if suffix:
        print(f"\n  NOTE: limit={args.limit} (<100), writing a scratch file rather than\n"
              f"        overwriting diagnosis_{args.split}.json")
    out = os.path.join(_HERE, f"diagnosis_{args.split}{suffix}.json")
    with open(out, "w") as f:
        json.dump({
            "split": args.split,
            "condition": "ORACLE gold evidence — isolates the verdict stage from retrieval",
            "note": "Conflicting Evidence/Cherrypicking excluded from scoring (no system equivalent)",
            "summary": summary,
            "per_claim": rows,
        }, f, indent=2)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--concurrency", type=int, default=6)
    sys.exit(asyncio.run(main_async(p.parse_args())))
