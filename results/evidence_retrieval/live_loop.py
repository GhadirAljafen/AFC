"""
Stage 2, final open question: does the retrieval loop help on the LIVE open web?

The closed-corpus evaluation found the loop statistically indistinguishable from a
larger single pass, and diagnosed why: round 2 drew from a picked-over pool
(4% gold vs 16.7% in round 1) because round 1's URLs were excluded from a single
973-document index. That setup structurally cannot show reformulation reaching
sources a first query could not — which is the loop's entire rationale. This is
the only fair test.

Design
------
One live run per claim with rounds=2, recording which URLs each round contributed.
Both conditions are then derived from the SAME run:

    round 1 only   = evidence after the first retrieval round
    rounds 1+2     = evidence after the gap-driven second round

This halves search spend versus running two conditions, and makes the comparison
exactly paired: the round-1 queries are literally the same queries, not merely
matched ones.

What this measures, and what it cannot
--------------------------------------
PRIMARY (well powered even at small n, because it is within-claim and descriptive):
  * does round 2 retrieve URLs and DOMAINS that round 1 did not?
  * how many, and are they relevant?
This is precisely what the closed corpus could not show.

SECONDARY (weakly powered, reported with that caveat):
  * gold-domain recall, round 1 vs rounds 1+2. Expect low absolute numbers: live
    search returns equally-good alternative sources, and AVeriTeC gold URLs suffer
    link rot. Domain-level matching is used rather than exact URL because an exact
    match is an unreasonable bar for live retrieval.
  * fact-check leakage, which should be far higher here than in the closed corpus
    since the live web is full of fact-checking articles.

NOT MEASURED: whether the loop improves verdict accuracy. Quota makes the sample
far too small for that, and claiming it would repeat the error this project has
already made twice.

Quota
-----
Google CSE free tier is 100 queries/day and the API sunsets 2027-01-01. Cost is
~6 searches per claim (up to 4 in round 1 including the auto-appended refutation
query, up to 2 gap queries in round 2). `--budget` is a HARD cap: the run stops
cleanly when it is reached and reports how far it got. Results are cached on disk,
so re-running the same claims costs nothing.

Usage:
    python results/evidence_retrieval/live_loop.py --limit 12 --budget 60
    python results/evidence_retrieval/live_loop.py --dry-run     # no quota spent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List, Set

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from averitec import AveritecClaim, default_data_dir, evaluable, load_split  # noqa: E402

from factcheck_agent.agents.retrieval import RetrievalAgent, _domain_of, _is_fact_check_source  # noqa: E402
from factcheck_agent.config import get_config  # noqa: E402
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from factcheck_agent.models import Claim  # noqa: E402
from factcheck_agent.pipeline import FactCheckingPipeline  # noqa: E402
from factcheck_agent.search import CachedSearchClient, GoogleCSEClient  # noqa: E402


class BudgetExceeded(Exception):
    """Raised to stop a run cleanly once the search budget is spent."""


class BudgetedSearch(CachedSearchClient):
    """Cached search that refuses to exceed a hard live-query budget.

    Cache hits are free and do not count, so re-running the same claims costs
    nothing. Only calls that actually reach the provider are charged.
    """

    def __init__(self, inner, cache_dir: str, budget: int):
        super().__init__(inner, cache_dir=cache_dir)
        self.budget = budget
        self.live_queries = 0

    async def search(self, query: str, k: int = 5):
        path = self._path(query, k)
        if not os.path.exists(path):
            if self.live_queries >= self.budget:
                raise BudgetExceeded(
                    f"search budget of {self.budget} live queries is spent"
                )
            self.live_queries += 1
        return await super().search(query, k=k)


async def run_claim(claim: AveritecClaim, llm, search) -> Dict:
    """One live run, recording what each retrieval round contributed."""
    agent = RetrievalAgent(llm=llm, search_client=search)
    pipeline = FactCheckingPipeline(
        llm=llm, retrieval_agent=agent, max_iterations=2,
        initial_retrieval_k=get_config().MAX_EVIDENCE_SOURCES,
        use_llm_normalization=False, full_text_top_k=0,
    )
    normalized = pipeline.claim_understanding.normalize_claim(
        Claim(id=claim.claim_id, raw_text=claim.claim)
    )
    before = search.live_queries
    evidence, evaluation, per_round = await pipeline.collect_evidence(normalized)
    spent = search.live_queries - before

    # collect_evidence extends in round order, so per_round slices it cleanly
    rounds, cursor = [], 0
    for i, count in enumerate(per_round):
        chunk = evidence[cursor:cursor + count]
        cursor += count
        rounds.append({
            "round": i + 1,
            "queries": agent.query_rounds[i] if i < len(agent.query_rounds) else [],
            "urls": [e.id for e in chunk],
            "domains": sorted({(e.metadata or {}).get("domain", "") for e in chunk} - {""}),
            "fact_check": sum(1 for e in chunk
                              if (e.metadata or {}).get("fact_check_source") == "True"),
        })

    gold_domains = set(claim.gold_domains)
    r1 = rounds[0] if rounds else {"urls": [], "domains": [], "queries": [], "fact_check": 0}
    r2 = rounds[1] if len(rounds) > 1 else None

    r1_domains: Set[str] = set(r1["domains"])
    r2_domains: Set[str] = set(r2["domains"]) if r2 else set()
    new_domains = r2_domains - r1_domains

    return {
        "claim_id": claim.claim_id,
        "claim": claim.claim[:120],
        "gold_label": claim.label,
        "gold_domains": sorted(gold_domains),
        "searches_spent": spent,
        "ran_second_round": r2 is not None,
        "round1": {"n_urls": len(r1["urls"]), "domains": sorted(r1_domains),
                   "queries": r1["queries"], "fact_check": r1["fact_check"]},
        "round2": ({"n_urls": len(r2["urls"]), "domains": sorted(r2_domains),
                    "new_domains": sorted(new_domains), "queries": r2["queries"],
                    "fact_check": r2["fact_check"]} if r2 else None),
        "gold_hit_r1": len(gold_domains & r1_domains),
        "gold_hit_r1r2": len(gold_domains & (r1_domains | r2_domains)),
        "n_gold_domains": len(gold_domains),
    }


def report(rows: List[Dict], search) -> Dict:
    ran2 = [r for r in rows if r["ran_second_round"]]
    print(f"\n{'=' * 88}\nLIVE OPEN-WEB RETRIEVAL LOOP — n={len(rows)} claims\n{'=' * 88}")
    print(f"live searches spent: {search.live_queries}   "
          f"cache hits: {search.hits}   (mean {statistics.mean([r['searches_spent'] for r in rows]):.1f} live/claim)")
    print(f"claims that ran a second round: {len(ran2)}/{len(rows)}")

    print(f"\n{'=' * 88}\nPRIMARY: does round 2 reach sources round 1 could not?\n{'=' * 88}")
    if ran2:
        new_counts = [len(r["round2"]["new_domains"]) for r in ran2]
        any_new = sum(1 for c in new_counts if c > 0)
        print(f"  claims where round 2 added a NEW domain: {any_new}/{len(ran2)}"
              f"  ({100 * any_new / len(ran2):.0f}%)")
        print(f"  new domains per second round: mean={statistics.mean(new_counts):.2f}  "
              f"max={max(new_counts)}")
        print(f"  round 1 domains/claim: {statistics.mean([len(r['round1']['domains']) for r in ran2]):.2f}")
        examples = [(r["claim"][:52], r["round2"]["new_domains"][:3])
                    for r in ran2 if r["round2"]["new_domains"]][:4]
        for claim_text, doms in examples:
            print(f"    + {doms}  ← {claim_text}...")
    else:
        print("  no claim ran a second round (round 1 judged sufficient every time)")

    print(f"\n{'=' * 88}\nSECONDARY (weakly powered — treat as descriptive)\n{'=' * 88}")
    scored = [r for r in rows if r["n_gold_domains"]]
    if scored:
        rec1 = statistics.mean([r["gold_hit_r1"] / r["n_gold_domains"] for r in scored])
        rec2 = statistics.mean([r["gold_hit_r1r2"] / r["n_gold_domains"] for r in scored])
        hit1 = sum(1 for r in scored if r["gold_hit_r1"])
        hit2 = sum(1 for r in scored if r["gold_hit_r1r2"])
        print(f"  gold-DOMAIN recall  round1={rec1:.3f}  rounds1+2={rec2:.3f}  "
              f"(delta {rec2 - rec1:+.3f})")
        print(f"  claims with >=1 gold domain found: {hit1} -> {hit2} of {len(scored)}")
        print("  NOTE: low absolute recall is expected. Live search returns equally-good")
        print("        alternative sources, and AVeriTeC gold URLs suffer link rot.")
    fc1 = sum(r["round1"]["fact_check"] for r in rows)
    fc2 = sum(r["round2"]["fact_check"] for r in ran2) if ran2 else 0
    print(f"\n  fact-check results retrieved: round1={fc1}  round2={fc2}")
    print("  (the live web carries far more fact-checking content than a closed corpus)")

    return {
        "claims": len(rows), "live_searches": search.live_queries,
        "cache_hits": search.hits,
        "ran_second_round": len(ran2),
        "new_domain_rate": (sum(1 for r in ran2 if r["round2"]["new_domains"]) / len(ran2)) if ran2 else 0,
        "mean_new_domains": statistics.mean([len(r["round2"]["new_domains"]) for r in ran2]) if ran2 else 0,
        "gold_domain_recall_r1": statistics.mean([r["gold_hit_r1"] / r["n_gold_domains"] for r in scored]) if scored else 0,
        "gold_domain_recall_r1r2": statistics.mean([r["gold_hit_r1r2"] / r["n_gold_domains"] for r in scored]) if scored else 0,
        "fact_check_round1": fc1, "fact_check_round2": fc2,
    }


async def main_async(args) -> int:
    claims = evaluable(load_split(os.path.join(args.data_dir, f"{args.split}.json")))[: args.limit]
    cfg = get_config()

    print(f"{'=' * 88}\nLIVE RETRIEVAL-LOOP EVALUATION (spends real search quota)\n{'=' * 88}")
    print(f"claims requested: {len(claims)}   hard budget: {args.budget} live searches")
    print(f"estimated cost: ~{cfg.MAX_QUERIES_PER_CLAIM + 1 + cfg.MAX_GAP_QUERIES} searches/claim "
          f"-> ~{args.budget // (cfg.MAX_QUERIES_PER_CLAIM + 1 + cfg.MAX_GAP_QUERIES)} claims within budget")
    print("Google CSE free tier is 100 queries/day. Cached queries are free.")

    if args.dry_run:
        print("\n--dry-run: no searches issued, nothing spent.")
        return 0

    inner = GoogleCSEClient()
    if not inner.is_configured():
        print("ERROR: Google Search is not configured.", file=sys.stderr)
        return 1

    search = BudgetedSearch(inner, cache_dir=cfg.SEARCH_CACHE_DIR, budget=args.budget)
    llm = get_default_llm_client(use_dummy_if_missing_key=False)

    rows: List[Dict] = []
    for i, claim in enumerate(claims, 1):
        try:
            rows.append(await run_claim(claim, llm, search))
            print(f"  [{i}/{len(claims)}] spent={search.live_queries:>3}  {claim.claim[:60]}...")
        except BudgetExceeded as e:
            print(f"\n  BUDGET REACHED after {len(rows)} claims: {e}")
            break
        except Exception as e:
            print(f"  [{i}] !! {type(e).__name__}: {e}")

    if not rows:
        print("No claims completed.", file=sys.stderr)
        return 1

    summary = report(rows, search)
    out = os.path.join(_HERE, f"live_loop_{args.split}.json")
    with open(out, "w") as f:
        json.dump({
            "split": args.split,
            "condition": "LIVE open-web search via Google CSE",
            "design": "one run per claim at rounds=2; both conditions derived from the "
                      "same run, so round-1 queries are identical by construction",
            "not_measured": "verdict accuracy — the quota-limited sample is far too "
                            "small to support that claim",
            "summary": summary, "per_claim": rows,
        }, f, indent=2)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--budget", type=int, default=60,
                   help="HARD cap on live searches; run stops cleanly when reached")
    p.add_argument("--dry-run", action="store_true", help="plan only, spend nothing")
    sys.exit(asyncio.run(main_async(p.parse_args())))
