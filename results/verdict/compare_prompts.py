"""
Stage 3 Phase B1: does prompt symmetry fix the over-abstention?

Phase A established (`FINDINGS.md`):
  * The confidence-threshold rules fire on 0 of 150 decisions — dead code.
  * All 14 missed gold-SUPPORTS claims are the MODEL declining to confirm, with
    no rule involved.
  * 33 of 37 errors (89%) are unwarranted NOT_ENOUGH_INFO on gold evidence.

So the cause is the prompt's asymmetric evidential bars. This measures the
symmetric rewrite against the original, paired.

Design
------
Same claims, same evidence, varying ONLY the prompt. Two evidence conditions:

  ORACLE     gold evidence -> isolates the verdict stage, the honest ceiling
  RETRIEVED  closed-corpus retrieval -> what a user actually gets

In the retrieved condition retrieval runs ONCE per claim and both prompts score
the identical evidence set, so query-generation nondeterminism cannot confound
the prompt comparison.

Significance by McNemar's exact test on paired correct/incorrect outcomes.

The majority-class baseline (0.706 here) is reported with every accuracy figure:
71% of gold claims are Refuted, so accuracy alone rewards a constant predictor.
Macro-F1 is the primary metric, and moving mass out of NOT_ENOUGH_INFO can lower
accuracy while improving macro-F1 — both are reported.

Usage:
    python results/verdict/compare_prompts.py --limit 150
    python results/verdict/compare_prompts.py --limit 150 --oracle-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "results", "evidence_retrieval"))

from averitec import AveritecClaim, default_data_dir, evaluable, load_split  # noqa: E402
from error_propagation import gold_evidence  # noqa: E402
from evaluate_retrieval import build_corpus  # noqa: E402
from fulltext_verdict import mcnemar  # noqa: E402
from sweep_k import CLASSES, LABEL_MAP, macro_f1  # noqa: E402

from factcheck_agent.agents.retrieval import RetrievalAgent  # noqa: E402
from factcheck_agent.agents.reasoning_and_verdict import ReasoningAndVerdictAgent  # noqa: E402
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from factcheck_agent.models import Claim  # noqa: E402
from factcheck_agent.pipeline import FactCheckingPipeline  # noqa: E402
from factcheck_agent.search import CorpusSearchClient  # noqa: E402

# (prompt_variant, reasoning_mode) — named so the report reads clearly.
ARMS = {
    "direct":    {"prompt_variant": "original", "reasoning_mode": "direct"},
    "symmetric": {"prompt_variant": "symmetric", "reasoning_mode": "direct"},
    "synthesis": {"prompt_variant": "original", "reasoning_mode": "synthesis"},
}
VARIANTS = tuple(ARMS)
K = 12


async def evidence_for(claim: AveritecClaim, corpus, llm, mode: str):
    """Build one evidence set per claim, shared by both prompt variants."""
    normalized = Claim(id=claim.claim_id, raw_text=claim.claim)
    if mode == "oracle":
        return normalized, gold_evidence(claim)

    pipeline = FactCheckingPipeline(
        llm=llm, max_iterations=1, initial_retrieval_k=K,
        use_llm_normalization=False, full_text_top_k=0,
        retrieval_agent=RetrievalAgent(
            llm=llm, search_client=CorpusSearchClient(corpus), max_results_per_query=5),
    )
    normalized = pipeline.claim_understanding.normalize_claim(normalized)
    evidence, evaluation, _ = await pipeline.collect_evidence(normalized)
    return normalized, (evaluation.selected_evidence if evaluation else evidence)


async def run_mode(claims, corpus, llm, mode: str, concurrency: int):
    agents = {v: ReasoningAndVerdictAgent(llm=llm, record_decisions=True, **ARMS[v])
              for v in VARIANTS}
    sem = asyncio.Semaphore(concurrency)
    rows: List[Dict] = []

    async def one(claim: AveritecClaim):
        async with sem:
            normalized, evidence = await evidence_for(claim, corpus, llm, mode)
            if not evidence:
                return
            row = {"claim_id": claim.claim_id, "gold": LABEL_MAP.get(claim.label),
                   "n_evidence": len(evidence)}
            for v in VARIANTS:
                verdict = await agents[v].decide(normalized, evidence)
                row[v] = {"pred": verdict.label, "confidence": verdict.confidence}
            rows.append(row)

    await asyncio.gather(*(one(c) for c in claims))
    return agents, [r for r in rows if r["gold"]]


def summarise(rows, variant: str) -> Dict:
    pairs = [(r["gold"], r[variant]["pred"]) for r in rows]
    preds = Counter(p for _, p in pairs)
    per_class = {}
    for cls in CLASSES:
        n = sum(1 for g, _ in pairs if g == cls)
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        per_class[cls] = {"n": n, "recall": round(tp / n, 3) if n else None}
    false_nei = sum(1 for g, p in pairs if p == "NOT_ENOUGH_INFO" and g != "NOT_ENOUGH_INFO")
    errors = sum(1 for g, p in pairs if g != p)
    return {
        "variant": variant, "n": len(pairs),
        "accuracy": sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0,
        "macro_f1": macro_f1(pairs),
        "abstention": preds.get("NOT_ENOUGH_INFO", 0) / len(pairs) if pairs else 0,
        "false_abstentions": false_nei,
        "errors": errors,
        "false_abstention_share_of_errors": false_nei / errors if errors else 0,
        "mean_confidence": statistics.mean([r[variant]["confidence"] for r in rows]) if rows else 0,
        "pred_distribution": dict(preds),
        "per_class_recall": per_class,
    }


def report(mode: str, rows, agents) -> Dict:
    print(f"\n{'=' * 90}\n{mode.upper()} EVIDENCE — n={len(rows)}\n{'=' * 90}")
    summaries = [summarise(rows, v) for v in VARIANTS]
    gold_counts = Counter(r["gold"] for r in rows)
    majority = max(gold_counts.values()) / len(rows) if rows else 0

    print(f"{'variant':>11} {'accuracy':>9} {'macroF1':>8} {'abstain':>8} "
          f"{'false-NEI':>10} {'conf':>6} | per-class recall")
    for s in summaries:
        pc = "  ".join(f"{c[:4]}={s['per_class_recall'][c]['recall']}" for c in CLASSES)
        print(f"{s['variant']:>11} {s['accuracy']:>9.4f} {s['macro_f1']:>8.4f} "
              f"{s['abstention']:>7.1%} {s['false_abstentions']:>10} "
              f"{s['mean_confidence']:>6.3f} | {pc}")
    print(f"{'majority':>11} {majority:>9.4f}  (constant REFUTES; accuracy at or below "
          f"this is not skill)")

    base = [r["gold"] == r["direct"]["pred"] for r in rows]
    base_s = summaries[0]
    tests, changes = {}, {}
    print()
    for i, v in enumerate(VARIANTS[1:], start=1):
        cand = [r["gold"] == r[v]["pred"] for r in rows]
        n_b, n_c, p = mcnemar(base, cand)
        d_acc = summaries[i]["accuracy"] - base_s["accuracy"]
        d_f1 = summaries[i]["macro_f1"] - base_s["macro_f1"]
        sig = "n.s." if (p is None or p >= 0.05) else "SIGNIFICANT"
        print(f"  {v:>10} - direct:  accuracy {d_acc:+.4f}  macro-F1 {d_f1:+.4f}  "
              f"| McNemar direct-only={n_b} {v}-only={n_c} "
              f"p={'n/a' if p is None else round(p, 5)}  {sig}")
        tests[v] = {"direct_only_right": n_b, f"{v}_only_right": n_c, "p": p,
                    "delta_accuracy": d_acc, "delta_macro_f1": d_f1}
        moved = Counter()
        for r in rows:
            a_, b_ = r["direct"]["pred"], r[v]["pred"]
            if a_ != b_:
                moved[f"{a_} -> {b_}"] += 1
        changes[v] = dict(moved.most_common())
        if moved:
            print(f"             label changes: {changes[v]}")

    print("\n  cost (LLM calls for the verdict stage):")
    for v in VARIANTS:
        st = agents[v].stats()
        print(f"    {v:>10}: {st['decisions']} verdict + {st['synthesis_calls']} synthesis"
              f" = {st['decisions'] + st['synthesis_calls']} calls"
              f"  (synthesis failures: {st['synthesis_failures']})")

    return {"mode": mode, "majority_baseline": majority, "summaries": summaries,
            "vs_direct": tests, "label_changes": changes,
            "counters": {v: agents[v].stats() for v in VARIANTS}}


async def main_async(args) -> int:
    all_claims = load_split(os.path.join(args.data_dir, f"{args.split}.json"))
    corpus = build_corpus(all_claims)
    claims = evaluable(all_claims)[: args.limit]

    print(f"{'=' * 90}\nPHASE B1 — PROMPT SYMMETRY, PAIRED\n{'=' * 90}")
    print(f"claims={len(claims)}  variants={VARIANTS}  "
          f"both score the SAME evidence per claim")

    llm = get_default_llm_client(use_dummy_if_missing_key=False)

    modes = ["oracle"] if args.oracle_only else ["oracle", "retrieved"]
    out: List[Dict] = []
    for mode in modes:
        agents, rows = await run_mode(claims, corpus, llm, mode, args.concurrency)
        out.append(report(mode, rows, agents))

    path = os.path.join(_HERE, f"prompt_comparison_{args.split}.json")
    with open(path, "w") as f:
        json.dump({"split": args.split, "claims": len(claims), "k": K,
                   "design": "paired; same evidence per claim, only the prompt varies",
                   "note": "Conflicting Evidence/Cherrypicking excluded (no system equivalent)",
                   "modes": out}, f, indent=2)
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--oracle-only", action="store_true")
    sys.exit(asyncio.run(main_async(p.parse_args())))
