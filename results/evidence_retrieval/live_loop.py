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
Google CSE free tier is 100 queries/day and the API sunsets 2027-01-01. Observed
cost is ~3.7 live searches per claim (up to 4 in round 1 including the
auto-appended refutation query, up to 2 gap queries in round 2; some are cache
hits). `--budget` is a HARD cap: the run stops cleanly when it is reached and
reports how far it got.

Incremental runs — why this script resumes rather than restarts
---------------------------------------------------------------
A sample large enough to carry weight needs more claims than one day's quota
buys, so the run accumulates across days. Two hazards make a plain re-run wrong,
not merely wasteful:

1. **Re-running finished claims is NOT free.** Cached searches cost nothing, but
   query generation runs at temperature 0.3, so replaying a finished claim emits
   *different* queries, misses the cache and spends quota on work already done.
   So claims already present in the output file are skipped outright — never
   re-generated and re-searched.
2. **A plain re-run would destroy the earlier days' data**, since the output file
   is rewritten. This project has lost result files to exactly that four times.
   New claims are therefore MERGED into the existing file, a per-run provenance
   record is appended, and a write that would reduce the claim count is refused
   unless `--fresh` is passed explicitly.

Summary statistics are recomputed over the full accumulated sample, so the file
always describes every claim collected, not just the latest batch.

Usage:
    python results/evidence_retrieval/live_loop.py --target 100 --budget 95
    python results/evidence_retrieval/live_loop.py --dry-run     # no quota spent
    python results/evidence_retrieval/live_loop.py --fresh ...   # discard & restart
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter
from datetime import date
from typing import Dict, List, Optional, Set

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


def load_prior(path: str) -> Dict:
    """Read an earlier run's output so this run can extend it.

    Returns empty structures when the file does not exist yet, so the first run
    and a resumed run take the same code path.
    """
    if not os.path.exists(path):
        return {"per_claim": [], "runs": []}
    with open(path) as f:
        prior = json.load(f)
    # Older files predate the `runs` provenance list; synthesise one entry for
    # them so the history is complete rather than silently missing its first day.
    if "runs" not in prior:
        rows = prior.get("per_claim", [])
        prior["runs"] = [{
            "date": "unrecorded (pre-dates run provenance)",
            "claims_added": len(rows),
            "live_searches": prior.get("summary", {}).get("live_searches"),
            "claim_ids": [r["claim_id"] for r in rows],
        }]
    return prior


def report(rows: List[Dict], total_searches: int) -> Dict:
    ran2 = [r for r in rows if r["ran_second_round"]]
    print(f"\n{'=' * 88}\nLIVE OPEN-WEB RETRIEVAL LOOP — n={len(rows)} claims (cumulative)\n{'=' * 88}")
    print(f"live searches spent across all runs: {total_searches}   "
          f"(mean {statistics.mean([r['searches_spent'] for r in rows]):.1f} live/claim)")
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
        "claims": len(rows), "live_searches": total_searches,
        "ran_second_round": len(ran2),
        "new_domain_rate": (sum(1 for r in ran2 if r["round2"]["new_domains"]) / len(ran2)) if ran2 else 0,
        "mean_new_domains": statistics.mean([len(r["round2"]["new_domains"]) for r in ran2]) if ran2 else 0,
        "gold_domain_recall_r1": statistics.mean([r["gold_hit_r1"] / r["n_gold_domains"] for r in scored]) if scored else 0,
        "gold_domain_recall_r1r2": statistics.mean([r["gold_hit_r1r2"] / r["n_gold_domains"] for r in scored]) if scored else 0,
        "fact_check_round1": fc1, "fact_check_round2": fc2,
    }


async def main_async(args) -> int:
    cfg = get_config()
    out = os.path.join(_HERE, f"live_loop_{args.split}.json")
    prior = {"per_claim": [], "runs": []} if args.fresh else load_prior(out)
    prior_rows: List[Dict] = prior["per_claim"]
    done_ids = {r["claim_id"] for r in prior_rows}

    all_claims = evaluable(load_split(os.path.join(args.data_dir, f"{args.split}.json")))
    remaining = [c for c in all_claims if c.claim_id not in done_ids]
    need = max(0, args.target - len(prior_rows))
    queue = remaining[: need if args.limit is None else min(need, args.limit)]

    print(f"{'=' * 88}\nLIVE RETRIEVAL-LOOP EVALUATION (spends real search quota)\n{'=' * 88}")
    print(f"already collected: {len(prior_rows)} claims over {len(prior['runs'])} run(s)"
          f"{'  [--fresh: IGNORING and restarting]' if args.fresh else '  (will be skipped, not re-run)'}")
    print(f"target: {args.target} claims  ->  {need} still needed, "
          f"{len(remaining)} unused claims available in the split")
    print(f"this run will attempt: {len(queue)} claims   hard budget: {args.budget} live searches")
    print(f"observed cost ~3.7 live searches/claim -> ~{int(args.budget / 3.7)} claims within budget")
    print("Google CSE free tier is 100 queries/day. Cached queries are free.")

    if args.dry_run:
        print("\n--dry-run: no searches issued, nothing spent.")
        return 0
    if not queue:
        print(f"\nTarget of {args.target} already met ({len(prior_rows)} claims). Nothing to do.")
        return 0

    inner = GoogleCSEClient()
    if not inner.is_configured():
        print("ERROR: Google Search is not configured.", file=sys.stderr)
        return 1

    search = BudgetedSearch(inner, cache_dir=cfg.SEARCH_CACHE_DIR, budget=args.budget)
    llm = get_default_llm_client(use_dummy_if_missing_key=False)

    new_rows: List[Dict] = []
    failed: List[Dict] = []
    for i, claim in enumerate(queue, 1):
        try:
            new_rows.append(await run_claim(claim, llm, search))
            print(f"  [{i}/{len(queue)}] spent={search.live_queries:>3}  "
                  f"total={len(prior_rows) + len(new_rows):>3}  {claim.claim[:52]}...")
        except BudgetExceeded as e:
            print(f"\n  BUDGET REACHED after {len(new_rows)} new claims: {e}")
            break
        except Exception as e:
            print(f"  [{i}] !! {type(e).__name__}: {e}")
            failed.append({"claim_id": claim.claim_id, "error": f"{type(e).__name__}: {e}"})

    if not new_rows:
        print("No new claims completed; leaving the existing file untouched.", file=sys.stderr)
        return 1

    rows = prior_rows + new_rows
    # Guard the overwrite hazard that has cost this project four result files: a
    # write must never shrink the accumulated sample unless explicitly requested.
    if not args.fresh and len(rows) < len(prior_rows):
        print("REFUSING to write: merged result has fewer claims than the existing "
              "file. Pass --fresh if discarding it is intended.", file=sys.stderr)
        return 1

    total_searches = sum(r["searches_spent"] for r in rows)
    summary = report(rows, total_searches)
    runs = list(prior["runs"]) + [{
        "date": str(date.today()),
        "claims_added": len(new_rows),
        "live_searches": search.live_queries,
        "cache_hits": search.hits,
        "budget": args.budget,
        "failed": failed,
        "claim_ids": [r["claim_id"] for r in new_rows],
    }]

    with open(out, "w") as f:
        json.dump({
            "split": args.split,
            "condition": "LIVE open-web search via Google CSE",
            "design": "one run per claim at rounds=2; both conditions derived from the "
                      "same run, so round-1 queries are identical by construction",
            "accumulation": f"Collected incrementally over {len(runs)} run(s) against a "
                            "100 queries/day free tier. Claims are never re-run: query "
                            "generation is stochastic, so replaying a finished claim "
                            "would miss the cache and spend quota for no new data. "
                            "Summary statistics cover all claims listed here.",
            "target": args.target,
            "not_measured": "verdict accuracy — measured separately in "
                            "results/verdict/live_evidence_verdict.py, which replays "
                            "these recorded queries from cache at zero quota",
            "runs": runs,
            "summary": summary, "per_claim": rows,
        }, f, indent=2)

    print(f"\nthis run: +{len(new_rows)} claims for {search.live_queries} live searches"
          f"{f', {len(failed)} failed' if failed else ''}")
    if len(rows) < args.target:
        togo = args.target - len(rows)
        print(f"cumulative: {len(rows)}/{args.target} claims  ({togo} to go, "
              f"~{togo * 3.7 / 100:.1f} more days at the 100/day free tier)")
    else:
        print(f"cumulative: {len(rows)} claims — TARGET OF {args.target} REACHED")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--target", type=int, default=100,
                   help="cumulative claim count to work toward across runs")
    p.add_argument("--limit", type=int, default=None,
                   help="optional cap on NEW claims attempted this run (default: "
                        "as many as the target and budget allow)")
    p.add_argument("--budget", type=int, default=95,
                   help="HARD cap on live searches; run stops cleanly when reached")
    p.add_argument("--fresh", action="store_true",
                   help="discard the accumulated file and start over (destructive)")
    p.add_argument("--dry-run", action="store_true", help="plan only, spend nothing")
    sys.exit(asyncio.run(main_async(p.parse_args())))
