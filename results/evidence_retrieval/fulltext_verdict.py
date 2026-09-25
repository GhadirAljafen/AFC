"""
Stage 2 final experiment: does full-text enrichment produce better verdicts?

The open question from §9.5. We know real pages are reachable (70.7%) and carry
~17x more text than a snippet, and §8.1 showed 500-char snippets beat 200-char
ones by +0.118 accuracy. What is untested is `ContentFetcher` itself, end to end,
judged by the only thing that matters: verdict quality.

It has to be judged downstream. §8.1 condition B improved retrieval recall to the
best of any non-oracle condition (0.652) while making verdicts *worse*
(abstention 45.6% -> 49.3%). Retrieval metrics alone cannot tell you whether
evidence is useful.

Design — paired, evidence-set held constant
-------------------------------------------
Retrieval runs ONCE per claim. The resulting evidence set is then scored by the
verdict stage three times:

    snippet      the retrieved snippets, unchanged (current behaviour)
    fulltext@3   top-3 snippets replaced with fetched article passages
    fulltext@8   top-8 replaced

Identical URLs in every condition; only the TEXT differs. Running retrieval
separately per condition would let LLM query-generation nondeterminism change
which documents were found, confounding the comparison.

No verdict code is modified — this evaluates a Stage 2 component using a Stage 3
metric.

Paired significance uses McNemar's exact test, which is the correct test for
paired binary outcomes (correct/incorrect on the same claim).

Usage:
    python results/evidence_retrieval/fulltext_verdict.py --limit 150
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

from averitec import AveritecClaim, default_data_dir, evaluable, load_split  # noqa: E402
from evaluate_retrieval import build_corpus  # noqa: E402
from sweep_k import CLASSES, LABEL_MAP, macro_f1  # noqa: E402

from factcheck_agent.agents.retrieval import RetrievalAgent  # noqa: E402
from factcheck_agent.content_fetch import ContentFetcher  # noqa: E402
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from factcheck_agent.models import Claim  # noqa: E402
from factcheck_agent.pipeline import FactCheckingPipeline  # noqa: E402
from factcheck_agent.search import CorpusSearchClient  # noqa: E402

K = 12
CONDITIONS = ["snippet", "fulltext@3", "fulltext@8"]


def mcnemar(a_correct: List[bool], b_correct: List[bool]):
    """Exact McNemar test for paired binary outcomes. Returns (b, c, p)."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    if b + c == 0:
        return b, c, None
    try:
        # binomtest is two-sided by default, which is what McNemar's exact test
        # needs: under the null, each discordant pair is equally likely to fall
        # either way, so the count of one kind is Binomial(b + c, 0.5).
        from scipy.stats import binomtest
        return b, c, float(binomtest(min(b, c), b + c, 0.5).pvalue)
    except Exception:
        return b, c, None


async def run_claim(claim: AveritecClaim, corpus, llm, fetcher: ContentFetcher) -> Optional[Dict]:
    pipeline = FactCheckingPipeline(
        llm=llm, max_iterations=1, initial_retrieval_k=K,
        use_llm_normalization=False, full_text_top_k=0,   # enrichment done manually below
        retrieval_agent=RetrievalAgent(
            llm=llm, search_client=CorpusSearchClient(corpus), max_results_per_query=5),
    )
    normalized = pipeline.claim_understanding.normalize_claim(
        Claim(id=claim.claim_id, raw_text=claim.claim)
    )
    claim_text = normalized.normalized_text or normalized.raw_text

    # ---- retrieve ONCE; every condition scores the same evidence set ----
    evidence, evaluation, _ = await pipeline.collect_evidence(normalized)
    base = evaluation.selected_evidence if evaluation else evidence
    if not base:
        return None

    variants = {"snippet": base}
    for top_k in (3, 8):
        variants[f"fulltext@{top_k}"] = await fetcher.enrich(base, claim_text, top_k=top_k)

    out: Dict[str, object] = {
        "claim_id": claim.claim_id,
        "gold_label": claim.label,
        "n_evidence": len(base),
    }
    for name, ev in variants.items():
        verdict = await pipeline.reasoning_and_verdict.decide(normalized, ev)
        enriched = sum(1 for e in ev if (e.metadata or {}).get("full_text") == "true")
        out[name] = {
            "pred": verdict.label,
            "confidence": verdict.confidence,
            "mean_chars": statistics.mean([len(e.text) for e in ev]) if ev else 0,
            "total_chars": sum(len(e.text) for e in ev),
            "enriched_docs": enriched,
        }
    return out


def summarise(rows: List[Dict], name: str) -> Dict:
    pairs = [(LABEL_MAP[r["gold_label"]], r[name]["pred"])
             for r in rows if r["gold_label"] in LABEL_MAP]
    preds = Counter(p for _, p in pairs)
    per_class = {}
    for cls in CLASSES:
        n = sum(1 for g, _ in pairs if g == cls)
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        per_class[cls] = {"n": n, "recall": round(tp / n, 3) if n else None}
    return {
        "condition": name,
        "n": len(pairs),
        "accuracy": sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0,
        "macro_f1": macro_f1(pairs),
        "abstention": preds.get("NOT_ENOUGH_INFO", 0) / len(pairs) if pairs else 0,
        "mean_chars": statistics.mean([r[name]["mean_chars"] for r in rows]) if rows else 0,
        "total_chars": statistics.mean([r[name]["total_chars"] for r in rows]) if rows else 0,
        "enriched_docs": statistics.mean([r[name]["enriched_docs"] for r in rows]) if rows else 0,
        "pred_distribution": dict(preds),
        "per_class_recall": per_class,
    }


async def main_async(args) -> int:
    all_claims = load_split(os.path.join(args.data_dir, f"{args.split}.json"))
    corpus = build_corpus(all_claims)
    pool = evaluable(all_claims)[: args.limit]

    print(f"{'=' * 90}\nFULL-TEXT ENRICHMENT — does better evidence produce better verdicts?\n{'=' * 90}")
    print(f"claims={len(pool)}  corpus={len(corpus)} docs  k={K}  rounds=1")
    print("Paired: retrieval runs ONCE per claim; conditions differ only in evidence TEXT.\n")

    llm = get_default_llm_client(use_dummy_if_missing_key=False)
    fetcher = ContentFetcher(
        cache_dir=os.path.join(_HERE, ".cache_pages"),
        max_chars=args.max_chars, timeout=args.timeout, concurrency=args.concurrency,
    )

    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(c):
        async with sem:
            try:
                return await run_claim(c, corpus, llm, fetcher)
            except Exception as e:
                print(f"  !! {c.claim_id}: {type(e).__name__}: {e}")
                return None

    rows = [r for r in await asyncio.gather(*(guarded(c) for c in pool)) if r]
    print(f"claims scored: {len(rows)}")
    print(f"fetcher: {fetcher.fetched} pages extracted, {fetcher.failed} failed, "
          f"{fetcher.wayback_recoveries} recovered via Wayback")

    summaries = [summarise(rows, c) for c in CONDITIONS]

    print(f"\n{'=' * 90}\nRESULTS\n{'=' * 90}")
    print(f"{'condition':>12} {'enriched':>9} {'chars/doc':>10} {'total ch':>9} | "
          f"{'accuracy':>9} {'macroF1':>8} {'abstain':>8}")
    for s in summaries:
        print(f"{s['condition']:>12} {s['enriched_docs']:>9.1f} {s['mean_chars']:>10.0f} "
              f"{s['total_chars']:>9.0f} | {s['accuracy']:>9.4f} {s['macro_f1']:>8.4f} "
              f"{s['abstention']:>7.1%}")

    print(f"\n{'=' * 90}\nPAIRED TESTS (McNemar exact, vs snippet baseline)\n{'=' * 90}")
    base_correct = [LABEL_MAP.get(r["gold_label"]) == r["snippet"]["pred"]
                    for r in rows if r["gold_label"] in LABEL_MAP]
    tests = {}
    for name in CONDITIONS[1:]:
        cond_correct = [LABEL_MAP.get(r["gold_label"]) == r[name]["pred"]
                        for r in rows if r["gold_label"] in LABEL_MAP]
        b, c, p = mcnemar(base_correct, cond_correct)
        verdict = "n.s." if (p is None or p >= 0.05) else "SIGNIFICANT"
        tests[name] = {"snippet_only_correct": b, "fulltext_only_correct": c, "p": p}
        print(f"  {name:>12} vs snippet:  snippet-only-right={b}  fulltext-only-right={c}  "
              f"p={'n/a' if p is None else round(p, 5)}  {verdict}")

    print(f"\n{'=' * 90}\nPER-CLASS RECALL\n{'=' * 90}")
    print(f"{'condition':>12} " + " ".join(f"{c:>18}" for c in CLASSES))
    for s in summaries:
        print(f"{s['condition']:>12} " + " ".join(
            f"{str(s['per_class_recall'][c]['recall']):>18}" for c in CLASSES))

    out = os.path.join(_HERE, f"fulltext_verdict_{args.split}.json")
    with open(out, "w") as f:
        json.dump({
            "split": args.split, "claims_scored": len(rows), "k": K,
            "design": "paired; retrieval once per claim, conditions differ only in evidence text",
            "max_chars": args.max_chars,
            "fetcher": {"extracted": fetcher.fetched, "failed": fetcher.failed,
                        "wayback_recoveries": fetcher.wayback_recoveries},
            "summaries": summaries, "mcnemar_vs_snippet": tests,
        }, f, indent=2)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--max-chars", type=int, default=4000)
    p.add_argument("--timeout", type=float, default=12.0)
    p.add_argument("--concurrency", type=int, default=6)
    sys.exit(asyncio.run(main_async(p.parse_args())))
