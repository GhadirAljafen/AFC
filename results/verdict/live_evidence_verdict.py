"""
Does LIVE web evidence change verdict quality? (Stage 2 x Stage 3)

Every Stage 3 verdict measurement so far used one of two artificial evidence
sources:

    closed corpus (AVeriTeC answers indexed)   accuracy 0.544
    oracle (gold answers handed over)          accuracy 0.728

Neither is what a user gets. This scores the verdict stage on **real web evidence**
retrieved live, which nothing has measured.

The trap this is designed around
--------------------------------
Live retrieval surfaces fact-checking articles — measured at 4.8% of all results,
and **7 of these 15 claims retrieved at least one**. A fact-check page often states
the verdict outright, so live evidence could score *higher* simply because the
system looked up the answer. A naive "live is better" number would be measuring
circularity, not capability.

So results are **split by whether a fact-check page was retrieved**, the same
control applied to the justification-leakage question in FINDINGS.md section S.3.

Zero quota
----------
The 15 claims were already run live and their queries are cached. But query
generation runs at temperature 0.3, so replaying would emit *different* queries,
miss the cache and spend quota. Instead this forces the **exact queries recorded**
in live_loop_dev.json, so retrieval replays deterministically from disk: same
queries, same results, same ranking and fusion as production, nothing spent.

Sample size
-----------
Whatever `live_loop_dev.json` currently holds, minus Conflicting/Cherrypicking
claims, which are excluded from scoring (see sweep_k.py). That file is collected
incrementally against a 100 queries/day free tier, so this script is re-run after
each batch and its sample grows. The first batch scored 13 claims and was
explicitly descriptive-only; the caveat emitted below is derived from the actual
n rather than asserted, so it stops claiming "too small" once it isn't.

Usage:
    python results/verdict/live_evidence_verdict.py
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

from averitec import default_data_dir, evaluable, load_split  # noqa: E402
from sweep_k import CLASSES, LABEL_MAP, macro_f1  # noqa: E402

from factcheck_agent.agents.retrieval import RetrievalAgent  # noqa: E402
from factcheck_agent.config import get_config  # noqa: E402
from factcheck_agent.content_fetch import ContentFetcher  # noqa: E402
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from factcheck_agent.models import Claim  # noqa: E402
from factcheck_agent.agents.evidence_selection import EvidenceSelectionAgent  # noqa: E402
from factcheck_agent.agents.reasoning_and_verdict import ReasoningAndVerdictAgent  # noqa: E402
from factcheck_agent.search import CachedSearchClient, GoogleCSEClient  # noqa: E402


class NoLiveSearch(CachedSearchClient):
    """Cache-only search. Raises rather than spending quota on a miss."""

    def __init__(self, cache_dir: str):
        super().__init__(GoogleCSEClient(), cache_dir=cache_dir)
        self.misses_blocked = 0

    async def search(self, query: str, k: int = 5):
        if not os.path.exists(self._path(query, k)):
            self.misses_blocked += 1
            return []          # degrade rather than spend; counted and reported
        return await super().search(query, k=k)


class ReplayRetrievalAgent(RetrievalAgent):
    """Retrieval that reuses the exact queries recorded from the live run.

    Overriding only query generation keeps everything else — cached search, RRF
    fusion, lexical overlap, domain filtering, leakage counting — identical to
    production, so this is a faithful replay rather than a reimplementation.
    """

    def __init__(self, *args, recorded_queries: List[str], **kwargs):
        super().__init__(*args, **kwargs)
        self._recorded = list(recorded_queries)

    async def _generate_queries(self, claim_text: str, gaps: Optional[str] = None):
        return self._recorded


async def score_claim(record: Dict, claim_obj, llm, search, fetcher) -> Optional[Dict]:
    """Rebuild the live evidence for one claim and take a verdict on it."""
    queries = list(record["round1"]["queries"])
    if record.get("round2"):
        queries += record["round2"]["queries"]
    if not queries:
        return None

    agent = ReplayRetrievalAgent(
        llm=llm, search_client=search, recorded_queries=queries,
        max_results_per_query=5,
    )
    normalized = Claim(id=record["claim_id"], raw_text=claim_obj.claim)
    evidence = await agent.retrieve_evidence(normalized, max_queries=len(queries))
    if not evidence:
        return None

    selected = EvidenceSelectionAgent().select(
        normalized, evidence, k=get_config().MAX_EVIDENCE_SOURCES
    )
    verdict_agent = ReasoningAndVerdictAgent(llm=llm, record_decisions=True)

    out = {
        "claim_id": record["claim_id"],
        "gold": LABEL_MAP.get(record["gold_label"]),
        "n_evidence": len(selected),
        "retrieved_fact_check": bool(
            record["round1"]["fact_check"] or (record.get("round2") or {}).get("fact_check")
        ),
    }

    snippet_verdict = await verdict_agent.decide(normalized, selected)
    out["snippet"] = {"pred": snippet_verdict.label,
                      "chars": statistics.mean([len(e.text) for e in selected])}

    enriched = await fetcher.enrich(
        selected, normalized.normalized_text or normalized.raw_text, top_k=3
    )
    full_verdict = await verdict_agent.decide(normalized, enriched)
    out["fulltext"] = {"pred": full_verdict.label,
                       "chars": statistics.mean([len(e.text) for e in enriched]),
                       "enriched": sum(1 for e in enriched
                                       if (e.metadata or {}).get("full_text") == "true")}
    return out


def summarise(rows: List[Dict], cond: str) -> Dict:
    pairs = [(r["gold"], r[cond]["pred"]) for r in rows]
    preds = Counter(p for _, p in pairs)
    per_class = {}
    for cls in CLASSES:
        n = sum(1 for g, _ in pairs if g == cls)
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        per_class[cls] = {"n": n, "recall": round(tp / n, 3) if n else None}
    return {
        "condition": cond, "n": len(pairs),
        "accuracy": sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0,
        "macro_f1": macro_f1(pairs),
        "abstention": preds.get("NOT_ENOUGH_INFO", 0) / len(pairs) if pairs else 0,
        "mean_chars": statistics.mean([r[cond]["chars"] for r in rows]) if rows else 0,
        "pred_distribution": dict(preds), "per_class_recall": per_class,
    }


async def main_async(args) -> int:
    live = json.load(open(os.path.join(
        _ROOT, "results", "evidence_retrieval", "live_loop_dev.json")))
    by_id = {c.claim_id: c for c in
             evaluable(load_split(os.path.join(args.data_dir, f"{args.split}.json")))}

    cfg = get_config()
    search = NoLiveSearch(cfg.SEARCH_CACHE_DIR)
    fetcher = ContentFetcher(cache_dir=os.path.join(_ROOT, ".cache", "pages"),
                             max_chars=cfg.MAX_SOURCE_CHARS)
    llm = get_default_llm_client(use_dummy_if_missing_key=False)

    print(f"{'=' * 90}\nVERDICT QUALITY ON LIVE WEB EVIDENCE (replayed from cache, 0 quota)\n{'=' * 90}")

    rows = []
    for rec in live["per_claim"]:
        claim = by_id.get(rec["claim_id"])
        if not claim or not LABEL_MAP.get(rec["gold_label"]):
            continue
        r = await score_claim(rec, claim, llm, search, fetcher)
        if r:
            rows.append(r)

    if not rows:
        print("no claims scorable", file=sys.stderr)
        return 1

    print(f"scored {len(rows)} claims  |  cache misses blocked (would have cost quota): "
          f"{search.misses_blocked}  |  pages fetched: {fetcher.fetched}, failed: {fetcher.failed}")

    summaries = [summarise(rows, c) for c in ("snippet", "fulltext")]
    gold_counts = Counter(r["gold"] for r in rows)
    majority = max(gold_counts.values()) / len(rows)

    # The honesty of this line matters more than its wording: a 30-claim sample
    # supports a paired test that a 13-claim one does not, and the label should
    # follow the data rather than be frozen at whatever the first batch allowed.
    if len(rows) < 25:
        power = "DESCRIPTIVE ONLY, far too small for significance"
    elif len(rows) < 60:
        power = "paired tests possible; only large effects detectable"
    else:
        power = "adequately powered for moderate paired effects"
    print(f"\n{'=' * 90}\nRESULTS — n={len(rows)} ({power})\n{'=' * 90}")
    print(f"{'condition':>22} {'chars':>7} {'accuracy':>9} {'macroF1':>8} {'abstain':>8} | per-class recall")
    for s in summaries:
        pc = "  ".join(f"{c[:4]}={s['per_class_recall'][c]['recall']}" for c in CLASSES)
        print(f"{'live ' + s['condition']:>22} {s['mean_chars']:>7.0f} {s['accuracy']:>9.3f} "
              f"{s['macro_f1']:>8.3f} {s['abstention']:>7.1%} | {pc}")
    print(f"{'majority baseline':>22} {'':>7} {majority:>9.3f}")
    print(f"\n  for reference, already measured on the SAME benchmark:")
    print(f"    closed-corpus retrieved   accuracy 0.544   (RESULTS.md section 8)")
    print(f"    oracle gold evidence      accuracy 0.728   (FINDINGS.md section 1)")

    print(f"\n{'=' * 90}\nTHE CONTROL: does a retrieved fact-check page explain the score?\n{'=' * 90}")
    split = {}
    for name, subset in (("fact-check retrieved", [r for r in rows if r["retrieved_fact_check"]]),
                         ("no fact-check", [r for r in rows if not r["retrieved_fact_check"]])):
        if not subset:
            continue
        acc_s = sum(1 for r in subset if r["gold"] == r["snippet"]["pred"]) / len(subset)
        acc_f = sum(1 for r in subset if r["gold"] == r["fulltext"]["pred"]) / len(subset)
        print(f"  {name:>22} (n={len(subset):>2}):  snippet {acc_s:.3f}   fulltext {acc_f:.3f}")
        split[name] = {"n": len(subset), "snippet_accuracy": acc_s, "fulltext_accuracy": acc_f}
    if len(split) == 2:
        a, b = split["fact-check retrieved"], split["no fact-check"]
        gap = a["snippet_accuracy"] - b["snippet_accuracy"]
        print(f"\n  gap: {gap:+.3f} accuracy when a fact-check page was retrieved.")
        print("  A large positive gap means the score is partly the system reading a")
        print("  fact-checker's conclusion rather than judging evidence itself.")

    out = os.path.join(_HERE, f"live_evidence_verdict_{args.split}.json")
    with open(out, "w") as f:
        json.dump({
            "split": args.split, "claims_scored": len(rows),
            "condition": "LIVE web evidence, replayed from cache using the exact "
                         "queries recorded in live_loop_dev.json (0 search quota)",
            "caveat": f"n={len(rows)}. {power}.",
            "reference_baselines": {"closed_corpus_retrieved": 0.544, "oracle_gold": 0.728},
            "majority_baseline": majority,
            "summaries": summaries, "fact_check_split": split, "per_claim": rows,
        }, f, indent=2)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    sys.exit(asyncio.run(main_async(p.parse_args())))
