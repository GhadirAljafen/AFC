"""
Separating Stage 2 failure from Stage 3 failure on AVeriTeC.

The k sweep found the verdict stage answering NOT_ENOUGH_INFO for ~47% of claims
against ~7% gold, and scoring below the majority-class prior. That could be
caused by either stage: retrieval may be handing over the wrong evidence, or the
verdict stage may be unable to use even the right evidence.

Four conditions isolate the causes. All use k=12 and rounds=1.

  A  baseline            results/query=5,  snippet=200
  B  bigger pool         results/query=10, snippet=200   -> does the candidate
                         pool ceiling (~10 unique URLs) bind retrieval?
  C  thicker evidence    results/query=5,  snippet=500   -> is over-abstention
                         caused by truncated evidence?
  D  ORACLE              gold answers fed straight to the verdict stage, no
                         retrieval at all -> upper bound on Stage 3

D is the decisive one. If the verdict stage still over-abstains on perfect
evidence, the problem is Stage 3 and no amount of retrieval work will fix it. If
it does not, the deficit is attributable to retrieval.

Usage:
    python results/evidence_retrieval/error_propagation.py --limit 150
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

from averitec import AveritecClaim, default_data_dir, evaluable, load_split  # noqa: E402
from evaluate_retrieval import build_corpus  # noqa: E402
from sweep_k import CLASSES, LABEL_MAP, macro_f1  # noqa: E402

from factcheck_agent.agents.retrieval import RetrievalAgent  # noqa: E402
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from factcheck_agent.models import Claim, EvidenceSnippet  # noqa: E402
from factcheck_agent.pipeline import FactCheckingPipeline  # noqa: E402
from factcheck_agent.search import CorpusSearchClient  # noqa: E402

CONDITIONS = [
    {"name": "A_baseline",      "rpq": 5,  "snippet": 200, "oracle": False},
    {"name": "B_bigger_pool",   "rpq": 10, "snippet": 200, "oracle": False},
    {"name": "C_thicker_evid",  "rpq": 5,  "snippet": 500, "oracle": False},
    {"name": "D_oracle_gold",   "rpq": 0,  "snippet": 0,   "oracle": True},
]
K = 12


def gold_evidence(claim: AveritecClaim) -> List[EvidenceSnippet]:
    """The annotated evidence, as the retriever would have delivered it."""
    return [
        EvidenceSnippet(
            id=a.source_url or f"gold-{i}",
            source="gold",
            text=a.answer,
            score=1.0,
            metadata={"url": a.source_url, "domain": a.domain, "rank": "1"},
        )
        for i, a in enumerate(claim.scorable_answers)
    ]


async def run_claim(claim: AveritecClaim, corpus, cond: Dict, llm) -> Dict:
    pipeline = FactCheckingPipeline(
        llm=llm, max_iterations=1, initial_retrieval_k=K,
        use_llm_normalization=False, full_text_top_k=0,
        retrieval_agent=RetrievalAgent(
            llm=llm,
            search_client=CorpusSearchClient(corpus, snippet_chars=cond["snippet"] or 200),
            max_results_per_query=cond["rpq"] or 5,
        ),
    )
    normalized = pipeline.claim_understanding.normalize_claim(
        Claim(id=claim.claim_id, raw_text=claim.claim)
    )

    gold_urls = {a.source_url for a in claim.scorable_answers}

    if cond["oracle"]:
        final = gold_evidence(claim)
        recall = 1.0
        precision = 1.0
    else:
        evidence, evaluation, _ = await pipeline.collect_evidence(normalized)
        final = evaluation.selected_evidence if evaluation else evidence
        urls = {e.id for e in final}
        hits = len(urls & gold_urls)
        recall = hits / len(gold_urls) if gold_urls else 0.0
        precision = hits / len(urls) if urls else 0.0

    verdict = await pipeline.reasoning_and_verdict.decide(normalized, final)

    return {
        "claim_id": claim.claim_id,
        "n_evidence": len(final),
        "evidence_chars": statistics.mean([len(e.text) for e in final]) if final else 0,
        "recall": recall,
        "precision": precision,
        "gold_label": claim.label,
        "pred_label": verdict.label,
        "confidence": verdict.confidence,
    }


async def run_condition(claims, corpus, cond, llm, concurrency) -> List[Dict]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(c):
        async with sem:
            try:
                return await run_claim(c, corpus, cond, llm)
            except Exception as e:
                return {"claim_id": c.claim_id, "error": f"{type(e).__name__}: {e}"}

    return list(await asyncio.gather(*(guarded(c) for c in claims)))


def summarise(name: str, rows: List[Dict]) -> Dict:
    ok = [r for r in rows if "error" not in r]
    mappable = [r for r in ok if r["gold_label"] in LABEL_MAP]
    pairs = [(LABEL_MAP[r["gold_label"]], r["pred_label"]) for r in mappable]
    counts = Counter(g for g, _ in pairs)
    preds = Counter(p for _, p in pairs)

    # per-class recall, to show WHERE the errors are
    per_class = {}
    for cls in CLASSES:
        n = sum(1 for g, _ in pairs if g == cls)
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        per_class[cls] = {"n": n, "recall": tp / n if n else None}

    return {
        "condition": name,
        "errors": len(rows) - len(ok),
        "n_evidence": statistics.mean([r["n_evidence"] for r in ok]) if ok else 0,
        "evidence_chars": statistics.mean([r["evidence_chars"] for r in ok]) if ok else 0,
        "recall": statistics.mean([r["recall"] for r in ok]) if ok else 0,
        "precision": statistics.mean([r["precision"] for r in ok]) if ok else 0,
        "verdict_n": len(pairs),
        "accuracy": sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0,
        "majority": max(counts.values()) / len(pairs) if pairs else 0,
        "macro_f1": macro_f1(pairs),
        "abstention_rate": preds.get("NOT_ENOUGH_INFO", 0) / len(pairs) if pairs else 0,
        "pred_distribution": dict(preds),
        "per_class_recall": per_class,
    }


async def main_async(args) -> int:
    all_claims = load_split(os.path.join(args.data_dir, f"{args.split}.json"))
    corpus = build_corpus(all_claims)
    pool = evaluable(all_claims)[: args.limit]

    print(f"{'=' * 92}\nERROR PROPAGATION — is over-abstention caused by retrieval or by the verdict stage?\n{'=' * 92}")
    print(f"corpus={len(corpus)} docs   claims={len(pool)} (same set every condition)   k={K}   rounds=1")

    llm = get_default_llm_client(use_dummy_if_missing_key=False)

    summaries = []
    for cond in CONDITIONS:
        rows = await run_condition(pool, corpus, cond, llm, args.concurrency)
        s = summarise(cond["name"], rows)
        summaries.append(s)
        print(f"\n--- {cond['name']} ---")
        print(f"  evidence: n={s['n_evidence']:.1f} chars/snippet={s['evidence_chars']:.0f} "
              f"recall={s['recall']:.4f} precision={s['precision']:.4f}")
        print(f"  verdict : acc={s['accuracy']:.4f} (majority={s['majority']:.4f}) "
              f"macroF1={s['macro_f1']:.4f} abstention={s['abstention_rate']:.1%} errors={s['errors']}")
        print(f"  predicted: {s['pred_distribution']}")
        print("  per-class recall: " + "  ".join(
            f"{c}={v['recall']:.2f}(n={v['n']})" if v["recall"] is not None else f"{c}=n/a"
            for c, v in s["per_class_recall"].items()))

    print(f"\n{'=' * 92}\nSUMMARY\n{'=' * 92}")
    print(f"{'condition':>16} {'evid':>6} {'chars':>7} {'recall':>8} | "
          f"{'acc':>7} {'macroF1':>8} {'abstain':>8}")
    for s in summaries:
        print(f"{s['condition']:>16} {s['n_evidence']:>6.1f} {s['evidence_chars']:>7.0f} "
              f"{s['recall']:>8.4f} | {s['accuracy']:>7.4f} {s['macro_f1']:>8.4f} "
              f"{s['abstention_rate']:>7.1%}")

    base, oracle = summaries[0], summaries[-1]
    print(f"\nmajority-class baseline: {base['majority']:.4f}")
    print(f"ORACLE (perfect evidence) accuracy: {oracle['accuracy']:.4f}, "
          f"abstention {oracle['abstention_rate']:.1%}")
    if oracle["abstention_rate"] > 0.25:
        print("  -> The verdict stage over-abstains even on gold evidence: this is a\n"
              "     STAGE 3 problem. Improving retrieval cannot fix it.")
    else:
        print("  -> Gold evidence largely removes over-abstention: the deficit is\n"
              "     attributable to RETRIEVAL, not the verdict stage.")

    out = os.path.join(_HERE, f"error_propagation_{args.split}.json")
    with open(out, "w") as f:
        json.dump({"split": args.split, "claims": len(pool), "k": K,
                   "condition": "offline closed corpus (not comparable to live search)",
                   "summaries": summaries}, f, indent=2)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--concurrency", type=int, default=6)
    sys.exit(asyncio.run(main_async(p.parse_args())))
