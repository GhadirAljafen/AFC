"""
Stage 2: paired comparison + round-2 query diagnostics on AVeriTeC.

Answers two questions the first (n=50, unpaired) run left open:

1. **Is the volume-matched control's advantage real?** Every condition runs over
   the SAME claims, per-claim scores are kept, and conditions are compared with a
   paired Wilcoxon signed-rank test plus a paired bootstrap CI. Comparing means
   across independent runs — as the first evaluation did — cannot separate a real
   effect from per-claim variance.

2. **Why do round-2 queries underperform?** Round-by-round instrumentation records
   the queries issued, which URLs each round contributed, and how many were gold.
   Three hypotheses are tested directly:
     H1 "vocabulary drift" — round-2 queries share fewer tokens with the claim.
     H2 "picked-over pool"  — round 2 is forced into lower-ranked documents
                              because round 1's URLs are excluded.
     H3 "leakage steer"     — gap statements steer queries toward fact-checking
                              content.

Usage:
    python results/evidence_retrieval/paired_analysis.py --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Set

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

_TOKENS = re.compile(r"[a-z0-9]+")


def toks(text: str) -> Set[str]:
    return {t for t in _TOKENS.findall(text.lower()) if len(t) > 2}


# ----------------------------------------------------------------- conditions
CONDITIONS = [
    {"name": "1round_k5", "rounds": 1, "k": 5},
    {"name": "2round_k5", "rounds": 2, "k": 5},
    {"name": "1round_k9", "rounds": 1, "k": 9},
]


async def run_one(claim: AveritecClaim, corpus, rounds: int, k: int, llm) -> Dict:
    """Run one claim under one condition, returning per-claim detail."""
    search = CorpusSearchClient(corpus)
    agent = RetrievalAgent(llm=llm, search_client=search, max_results_per_query=5)
    pipeline = FactCheckingPipeline(
        llm=llm, retrieval_agent=agent, max_iterations=rounds,
        initial_retrieval_k=k, use_llm_normalization=False, full_text_top_k=0,
    )
    normalized = pipeline.claim_understanding.normalize_claim(
        Claim(id=claim.claim_id, raw_text=claim.claim)
    )
    evidence, _evaluation, per_round = await pipeline.collect_evidence(normalized)

    gold = {a.source_url for a in claim.scorable_answers}
    urls = [e.id for e in evidence]
    hits = len(set(urls) & gold)

    # Attribute evidence to rounds: collect_evidence extends in round order.
    rounds_detail = []
    cursor = 0
    for round_index, count in enumerate(per_round):
        chunk = evidence[cursor:cursor + count]
        cursor += count
        chunk_urls = [e.id for e in chunk]
        rounds_detail.append({
            "round": round_index + 1,
            "queries": agent.query_rounds[round_index] if round_index < len(agent.query_rounds) else [],
            "n_new": len(chunk_urls),
            "n_new_gold": len(set(chunk_urls) & gold),
            "mean_rank": statistics.mean(
                [int((e.metadata or {}).get("rank", 0)) for e in chunk]
            ) if chunk else 0.0,
            "n_fact_check": sum(
                1 for e in chunk if (e.metadata or {}).get("fact_check_source") == "True"
            ),
        })

    return {
        "claim_id": claim.claim_id,
        "n_gold": len(gold),
        "n_retrieved": len(set(urls)),
        "hits": hits,
        "recall": hits / len(gold) if gold else 0.0,
        "precision": hits / len(set(urls)) if urls else 0.0,
        "any_hit": 1.0 if hits else 0.0,
        "fc_article_domain_hit": 1.0 if claim.fact_check_domain and any(
            (e.metadata or {}).get("domain") == claim.fact_check_domain for e in evidence
        ) else 0.0,
        "rounds": rounds_detail,
        "claim_tokens": sorted(toks(claim.claim))[:0],  # placeholder, not serialised
    }


async def run_condition(claims, corpus, cond, llm, concurrency) -> Dict[str, Dict]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(c):
        async with sem:
            try:
                return await run_one(c, corpus, cond["rounds"], cond["k"], llm)
            except Exception as e:
                return {"claim_id": c.claim_id, "error": f"{type(e).__name__}: {e}"}

    out = await asyncio.gather(*(guarded(c) for c in claims))
    return {r["claim_id"]: r for r in out}


# ------------------------------------------------------------------ statistics
def wilcoxon(a: Sequence[float], b: Sequence[float]):
    """Paired Wilcoxon signed-rank. Returns (statistic, p) or (None, None)."""
    try:
        from scipy.stats import wilcoxon as w
        diffs = [x - y for x, y in zip(a, b)]
        if not any(diffs):
            return None, None
        stat, p = w(a, b, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return None, None


def bootstrap_ci(a: Sequence[float], b: Sequence[float], n=10000, seed=0):
    """Paired bootstrap 95% CI for mean(a) - mean(b)."""
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n_obs = len(diffs)
    means = []
    for _ in range(n):
        means.append(sum(diffs[rng.randrange(n_obs)] for _ in range(n_obs)) / n_obs)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def compare(label: str, res_a: Dict, res_b: Dict, ids: List[str], metric: str):
    a = [res_a[i][metric] for i in ids]
    b = [res_b[i][metric] for i in ids]
    delta = statistics.mean(a) - statistics.mean(b)
    lo, hi = bootstrap_ci(a, b)
    stat, p = wilcoxon(a, b)
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    ties = len(ids) - wins - losses
    sig = "" if p is None else ("  SIGNIFICANT" if p < 0.05 else "  n.s.")
    print(f"  {label:34} {metric:10} delta={delta:+.4f}  "
          f"95%CI=[{lo:+.4f},{hi:+.4f}]  p={p if p is None else round(p,5)}"
          f"  W/L/T={wins}/{losses}/{ties}{sig}")
    return {"metric": metric, "delta": delta, "ci_low": lo, "ci_high": hi,
            "p": p, "wins": wins, "losses": losses, "ties": ties}


# ----------------------------------------------------------------- diagnostics
def diagnose(res: Dict[str, Dict], claims_by_id: Dict[str, AveritecClaim]):
    """Test H1 (drift), H2 (picked-over pool), H3 (leakage steer)."""
    r1_overlap, r2_overlap, r1_len, r2_len = [], [], [], []
    r1_gold_rate, r2_gold_rate = [], []
    r1_rank, r2_rank = [], []
    r1_fc, r2_fc = [], []
    n_with_r2 = 0

    for cid, rec in res.items():
        if "error" in rec or not rec.get("rounds"):
            continue
        claim_tokens = toks(claims_by_id[cid].claim)
        if not claim_tokens:
            continue
        for rd in rec["rounds"]:
            qs = rd["queries"]
            if not qs:
                continue
            overlap = statistics.mean(
                [len(claim_tokens & toks(q)) / len(claim_tokens) for q in qs]
            )
            length = statistics.mean([len(toks(q)) for q in qs])
            gold_rate = rd["n_new_gold"] / rd["n_new"] if rd["n_new"] else 0.0
            if rd["round"] == 1:
                r1_overlap.append(overlap); r1_len.append(length)
                r1_gold_rate.append(gold_rate); r1_rank.append(rd["mean_rank"])
                r1_fc.append(rd["n_fact_check"])
            else:
                n_with_r2 += 1
                r2_overlap.append(overlap); r2_len.append(length)
                r2_gold_rate.append(gold_rate); r2_rank.append(rd["mean_rank"])
                r2_fc.append(rd["n_fact_check"])

    def m(xs):
        return statistics.mean(xs) if xs else float("nan")

    print(f"\n  claims that ran a 2nd round: {n_with_r2}")
    print(f"  {'':28} {'round 1':>10} {'round 2':>10}")
    print(f"  {'H1 claim-token overlap':28} {m(r1_overlap):>10.3f} {m(r2_overlap):>10.3f}")
    print(f"  {'H1 query length (tokens)':28} {m(r1_len):>10.2f} {m(r2_len):>10.2f}")
    print(f"  {'H2 gold rate of new docs':28} {m(r1_gold_rate):>10.3f} {m(r2_gold_rate):>10.3f}")
    print(f"  {'H2 mean corpus rank':28} {m(r1_rank):>10.2f} {m(r2_rank):>10.2f}")
    print(f"  {'H3 fact-check docs/round':28} {m(r1_fc):>10.2f} {m(r2_fc):>10.2f}")
    return {
        "claims_with_second_round": n_with_r2,
        "h1_claim_overlap": {"round1": m(r1_overlap), "round2": m(r2_overlap)},
        "h1_query_len": {"round1": m(r1_len), "round2": m(r2_len)},
        "h2_gold_rate": {"round1": m(r1_gold_rate), "round2": m(r2_gold_rate)},
        "h2_mean_rank": {"round1": m(r1_rank), "round2": m(r2_rank)},
        "h3_fact_check": {"round1": m(r1_fc), "round2": m(r2_fc)},
    }


async def main_async(args) -> int:
    path = os.path.join(args.data_dir, f"{args.split}.json")
    all_claims = load_split(path)
    corpus = build_corpus(all_claims)
    pool = evaluable(all_claims)[: args.limit]
    by_id = {c.claim_id: c for c in pool}

    print(f"{'=' * 78}\nPAIRED COMPARISON + ROUND-2 DIAGNOSTICS — AVeriTeC {args.split}\n{'=' * 78}")
    print(f"corpus docs={len(corpus)}  claims={len(pool)}  (same claims in every condition)")

    llm = get_default_llm_client(use_dummy_if_missing_key=False)

    results: Dict[str, Dict[str, Dict]] = {}
    for cond in CONDITIONS:
        print(f"\n--- {cond['name']} (rounds={cond['rounds']}, k={cond['k']}) ---")
        results[cond["name"]] = await run_condition(pool, corpus, cond, llm, args.concurrency)
        ok = [r for r in results[cond["name"]].values() if "error" not in r]
        errs = len(results[cond["name"]]) - len(ok)
        print(f"  recall={statistics.mean([r['recall'] for r in ok]):.4f}  "
              f"precision={statistics.mean([r['precision'] for r in ok]):.4f}  "
              f"any-hit={statistics.mean([r['any_hit'] for r in ok]):.4f}  "
              f"retrieved={statistics.mean([r['n_retrieved'] for r in ok]):.2f}  errors={errs}")

    # claims scored successfully under EVERY condition -> valid paired set
    ids = [cid for cid in by_id
           if all("error" not in results[c["name"]].get(cid, {"error": 1}) for c in CONDITIONS)]
    print(f"\nPaired set (scored under all conditions): {len(ids)} claims")

    print(f"\n{'=' * 78}\nPAIRED TESTS\n{'=' * 78}")
    comparisons = {}
    for a, b in (("1round_k9", "2round_k5"), ("2round_k5", "1round_k5"), ("1round_k9", "1round_k5")):
        print(f"\n{a}  vs  {b}")
        comparisons[f"{a}_vs_{b}"] = [
            compare(f"{a} - {b}", results[a], results[b], ids, metric)
            for metric in ("recall", "any_hit", "precision", "fc_article_domain_hit")
        ]

    print(f"\n{'=' * 78}\nROUND-2 DIAGNOSTICS (2round_k5)\n{'=' * 78}")
    diag = diagnose(results["2round_k5"], by_id)

    out = os.path.join(_HERE, f"paired_analysis_{args.split}.json")
    with open(out, "w") as f:
        json.dump({
            "split": args.split, "claims": len(pool), "paired_claims": len(ids),
            "corpus_documents": len(corpus),
            "condition": "offline closed corpus (not comparable to live search)",
            "summary": {
                c["name"]: {
                    m: statistics.mean([r[m] for r in results[c["name"]].values() if "error" not in r])
                    for m in ("recall", "precision", "any_hit", "n_retrieved", "fc_article_domain_hit")
                } for c in CONDITIONS
            },
            "paired_tests": comparisons,
            "round2_diagnostics": diag,
        }, f, indent=2)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=6)
    sys.exit(asyncio.run(main_async(p.parse_args())))
