"""
Stage 2/3: sweep evidence-set size k, measuring BOTH retrieval and verdict.

Question
--------
Raising `MAX_EVIDENCE_SOURCES` from 5 to 9 bought +0.101 recall at no extra search
cost, but cost 0.042 precision. Whether that trade is worth making cannot be
decided from retrieval numbers alone — it depends on what the verdict stage does
with extra low-relevance evidence. This sweeps k over {5,7,9,12,15} on the same
claims and reports retrieval saturation alongside verdict quality.

All conditions use rounds=1: the paired analysis found the retrieval loop
statistically indistinguishable from a larger single pass, so holding rounds
fixed isolates k and costs less.

Label mapping (and its limits)
------------------------------
AVeriTeC uses four labels; this system emits three:

    Supported                          -> SUPPORTS
    Refuted                            -> REFUTES
    Not Enough Evidence                -> NOT_ENOUGH_INFO
    Conflicting Evidence/Cherrypicking -> (no equivalent)

The system cannot express "conflicting", so those claims are EXCLUDED from
accuracy and reported separately. Scoring them against any of the three labels
would measure a representational gap, not verdict quality.

Two baselines are reported, because dev is heavily skewed (61% Refuted) and this
system has a documented refutation bias:
  * majority-class: always predict the most common label
  * macro-F1: unweighted per-class F1, which a majority-class predictor cannot win

Usage:
    python results/evidence_retrieval/sweep_k.py --limit 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from averitec import AveritecClaim, default_data_dir, evaluable, load_split  # noqa: E402
from evaluate_retrieval import build_corpus  # noqa: E402

from factcheck_agent.agents.retrieval import RetrievalAgent  # noqa: E402
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from factcheck_agent.models import Claim  # noqa: E402
from factcheck_agent.pipeline import FactCheckingPipeline  # noqa: E402
from factcheck_agent.search import CorpusSearchClient  # noqa: E402

LABEL_MAP = {
    "Supported": "SUPPORTS",
    "Refuted": "REFUTES",
    "Not Enough Evidence": "NOT_ENOUGH_INFO",
}
CLASSES = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]


def macro_f1(pairs: Sequence[tuple]) -> float:
    """Unweighted mean per-class F1 over (gold, pred)."""
    f1s = []
    for cls in CLASSES:
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        fp = sum(1 for g, p in pairs if g != cls and p == cls)
        fn = sum(1 for g, p in pairs if g == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


async def run_claim(claim: AveritecClaim, corpus, k: int, llm) -> Dict:
    search = CorpusSearchClient(corpus)
    agent = RetrievalAgent(llm=llm, search_client=search, max_results_per_query=5)
    pipeline = FactCheckingPipeline(
        llm=llm, retrieval_agent=agent, max_iterations=1,
        initial_retrieval_k=k, use_llm_normalization=False, full_text_top_k=0,
    )
    normalized = pipeline.claim_understanding.normalize_claim(
        Claim(id=claim.claim_id, raw_text=claim.claim)
    )
    evidence, evaluation, _rounds = await pipeline.collect_evidence(normalized)
    final = evaluation.selected_evidence if evaluation else evidence

    # Verdict only; the explanation is skipped since it is not scored here.
    verdict = await pipeline.reasoning_and_verdict.decide(normalized, final)

    gold_urls = {a.source_url for a in claim.scorable_answers}
    urls = {e.id for e in final}
    hits = len(urls & gold_urls)

    return {
        "claim_id": claim.claim_id,
        "k": k,
        "n_evidence": len(final),
        "recall": hits / len(gold_urls) if gold_urls else 0.0,
        "precision": hits / len(urls) if urls else 0.0,
        "any_hit": 1.0 if hits else 0.0,
        "gold_label": claim.label,
        "pred_label": verdict.label,
        "confidence": verdict.confidence,
    }


async def run_k(claims, corpus, k, llm, concurrency) -> List[Dict]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(c):
        async with sem:
            try:
                return await run_claim(c, corpus, k, llm)
            except Exception as e:
                return {"claim_id": c.claim_id, "k": k, "error": f"{type(e).__name__}: {e}"}

    return list(await asyncio.gather(*(guarded(c) for c in claims)))


def summarise(rows: List[Dict]) -> Dict:
    ok = [r for r in rows if "error" not in r]
    mappable = [r for r in ok if r["gold_label"] in LABEL_MAP]
    pairs = [(LABEL_MAP[r["gold_label"]], r["pred_label"]) for r in mappable]

    accuracy = sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0
    gold_counts = Counter(g for g, _ in pairs)
    majority = max(gold_counts.values()) / len(pairs) if pairs else 0.0

    conflicting = [r for r in ok if r["gold_label"] == "Conflicting Evidence/Cherrypicking"]

    return {
        "k": rows[0]["k"],
        "errors": len(rows) - len(ok),
        "n_evidence": statistics.mean([r["n_evidence"] for r in ok]) if ok else 0,
        "recall": statistics.mean([r["recall"] for r in ok]) if ok else 0,
        "precision": statistics.mean([r["precision"] for r in ok]) if ok else 0,
        "any_hit": statistics.mean([r["any_hit"] for r in ok]) if ok else 0,
        "verdict_n": len(pairs),
        "verdict_accuracy": accuracy,
        "majority_baseline": majority,
        "macro_f1": macro_f1(pairs),
        "pred_distribution": dict(Counter(p for _, p in pairs)),
        "conflicting_claims": len(conflicting),
        "conflicting_predictions": dict(Counter(r["pred_label"] for r in conflicting)),
        "mean_confidence": statistics.mean([r["confidence"] for r in ok]) if ok else 0,
    }


async def main_async(args) -> int:
    path = os.path.join(args.data_dir, f"{args.split}.json")
    all_claims = load_split(path)
    corpus = build_corpus(all_claims)
    pool = evaluable(all_claims)[: args.limit]

    ks = [int(x) for x in args.ks.split(",")]
    print(f"{'=' * 88}\nk SWEEP — retrieval saturation and verdict impact (AVeriTeC {args.split})\n{'=' * 88}")
    print(f"corpus={len(corpus)} docs   claims={len(pool)} (same set for every k)   rounds=1   k={ks}")
    print(f"gold labels in pool: {dict(Counter(c.label for c in pool))}")

    llm = get_default_llm_client(use_dummy_if_missing_key=False)

    summaries, per_claim = [], {}
    for k in ks:
        rows = await run_k(pool, corpus, k, llm, args.concurrency)
        per_claim[k] = rows
        s = summarise(rows)
        summaries.append(s)
        print(f"\n--- k={k} ---")
        print(f"  retrieval: recall={s['recall']:.4f} precision={s['precision']:.4f} "
              f"any-hit={s['any_hit']:.4f} evidence={s['n_evidence']:.1f}")
        print(f"  verdict  : acc={s['verdict_accuracy']:.4f} (majority={s['majority_baseline']:.4f}) "
              f"macroF1={s['macro_f1']:.4f} n={s['verdict_n']} errors={s['errors']}")
        print(f"  predicted: {s['pred_distribution']}")

    print(f"\n{'=' * 88}\nSATURATION\n{'=' * 88}")
    print(f"{'k':>4} {'evid':>6} {'recall':>8} {'Δrecall':>9} {'prec':>7} "
          f"{'any-hit':>8} | {'verdict acc':>12} {'macro-F1':>9} {'Δacc':>7}")
    prev_r = prev_a = None
    for s in summaries:
        dr = "" if prev_r is None else f"{s['recall'] - prev_r:+.4f}"
        da = "" if prev_a is None else f"{s['verdict_accuracy'] - prev_a:+.4f}"
        print(f"{s['k']:>4} {s['n_evidence']:>6.1f} {s['recall']:>8.4f} {dr:>9} "
              f"{s['precision']:>7.4f} {s['any_hit']:>8.4f} | "
              f"{s['verdict_accuracy']:>12.4f} {s['macro_f1']:>9.4f} {da:>7}")
        prev_r, prev_a = s["recall"], s["verdict_accuracy"]

    base = summaries[0]
    print(f"\nmajority-class baseline: {base['majority_baseline']:.4f} "
          f"— any accuracy at or below this is not evidence of verdict skill.")
    print(f"Conflicting claims (excluded from accuracy): {base['conflicting_claims']}; "
          f"predicted as {base['conflicting_predictions']}")

    out = os.path.join(_HERE, f"sweep_k_{args.split}.json")
    with open(out, "w") as f:
        json.dump({
            "split": args.split, "claims": len(pool), "corpus_documents": len(corpus),
            "condition": "offline closed corpus, rounds=1 (not comparable to live search)",
            "label_mapping": LABEL_MAP,
            "excluded_label": "Conflicting Evidence/Cherrypicking (no system equivalent)",
            "summaries": summaries,
        }, f, indent=2)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--ks", default="5,7,9,12,15")
    p.add_argument("--concurrency", type=int, default=6)
    sys.exit(asyncio.run(main_async(p.parse_args())))
