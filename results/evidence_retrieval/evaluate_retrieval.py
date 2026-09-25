"""
Stage 2 (evidence retrieval) evaluation on AVeriTeC — offline, closed corpus.

What this measures
------------------
**Closed-corpus retrieval recall.** A single index is built from every annotated
answer in the split (keyed by Wayback-unwrapped source URL). For each claim, the
gold set is the URLs of its own scorable answers; every other document is a
distractor. The retriever sees only the claim and must surface the right URLs.

This is deliberately NOT called "evidence coverage". The corpus is guaranteed to
contain the answer, so these numbers are easier than open-web search and are not
comparable to a live run. What it does measure cleanly, for free and
deterministically, is *ranking and loop behaviour*: whether a gap-driven second
retrieval round surfaces gold documents that the first round missed.

Headline experiment
-------------------
The same claims are run at `--rounds 1` (single pass, approximating the old
no-op loop) and `--rounds N` (loop active). The delta in recall is the direct
measure of what Workstream C bought.

Scoping (see averitec.py for why)
---------------------------------
Only claims with at least one *scorable* gold answer are evaluated
(Extractive/Abstractive + Web text + usable URL): 391 of 500 dev claims. Boolean
answers, unanswerable answers, and PDF/image/video sources are excluded because
no text retriever could match them, and charging for them would understate
performance for reasons unrelated to retrieval quality.

Usage
-----
    python results/evidence_retrieval/evaluate_retrieval.py --limit 50
    python results/evidence_retrieval/evaluate_retrieval.py --limit 50 --rounds 3
    python results/evidence_retrieval/evaluate_retrieval.py --limit 20 --no-llm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from averitec import AveritecClaim, default_data_dir, describe, evaluable, load_split  # noqa: E402

from factcheck_agent.agents.retrieval import RetrievalAgent  # noqa: E402
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from factcheck_agent.models import Claim  # noqa: E402
from factcheck_agent.pipeline import FactCheckingPipeline  # noqa: E402
from factcheck_agent.search import CorpusSearchClient  # noqa: E402


def build_corpus(claims: Sequence[AveritecClaim]) -> List[Dict[str, str]]:
    """One document per distinct source URL, body = its annotated answer text(s).

    Questions are deliberately NOT used as document titles: they are written from
    the claim, so indexing them would leak the target into the corpus.
    """
    by_url: Dict[str, List[str]] = {}
    for claim in claims:
        for answer in claim.answers:
            if not answer.source_url.startswith("http") or not answer.answer:
                continue
            by_url.setdefault(answer.source_url, []).append(answer.answer)
    return [{"url": url, "title": "", "text": " ".join(texts)}
            for url, texts in by_url.items()]


def score_claim(retrieved_urls: Sequence[str], gold_urls: Sequence[str]) -> Dict[str, float]:
    retrieved, gold = set(retrieved_urls), set(gold_urls)
    if not gold:
        return {}
    hits = len(retrieved & gold)
    return {
        "recall": hits / len(gold),
        "precision": hits / len(retrieved) if retrieved else 0.0,
        "any_hit": 1.0 if hits else 0.0,
        "n_retrieved": len(retrieved),
        "n_gold": len(gold),
    }


async def run_condition(
    claims: Sequence[AveritecClaim],
    corpus: Sequence[Dict[str, str]],
    rounds: int,
    llm,
    concurrency: int = 4,
    results_per_query: int = 5,
    evidence_k: Optional[int] = None,
) -> Dict[str, object]:
    """Run every claim at a fixed number of retrieval rounds."""
    semaphore = asyncio.Semaphore(concurrency)
    per_claim: List[Dict[str, float]] = []
    fc_domain_hits = 0
    gap_rounds_used = 0

    async def one(claim: AveritecClaim):
        nonlocal fc_domain_hits, gap_rounds_used
        async with semaphore:
            search = CorpusSearchClient(corpus)
            agent = RetrievalAgent(
                llm=llm, search_client=search, max_results_per_query=results_per_query
            )
            pipeline = FactCheckingPipeline(
                llm=llm,
                retrieval_agent=agent,
                max_iterations=rounds,
                initial_retrieval_k=evidence_k,
                # AVeriTeC claims are already standalone, so LLM normalization is a
                # near no-op here; skipping it removes a confound and a cost.
                use_llm_normalization=False,
                full_text_top_k=0,          # corpus documents have no live pages
            )
            normalized = pipeline.claim_understanding.normalize_claim(
                Claim(id=claim.claim_id, raw_text=claim.claim)
            )
            evidence, _evaluation, per_round = await pipeline.collect_evidence(normalized)

        if len(per_round) > 1:
            gap_rounds_used += 1
        urls = [e.id for e in evidence]
        if claim.fact_check_domain and any(
            (e.metadata or {}).get("domain") == claim.fact_check_domain for e in evidence
        ):
            fc_domain_hits += 1

        scored = score_claim(urls, [a.source_url for a in claim.scorable_answers])
        if scored:
            per_claim.append(scored)

    outcomes = await asyncio.gather(*(one(c) for c in claims), return_exceptions=True)
    errors = [o for o in outcomes if isinstance(o, BaseException)]
    for err in errors[:3]:
        print(f"    !! {type(err).__name__}: {err}")

    def mean(key: str) -> float:
        vals = [c[key] for c in per_claim if key in c]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "rounds": rounds,
        "evidence_k": evidence_k,
        "claims_scored": len(per_claim),
        "errors": len(errors),
        "recall": mean("recall"),
        "precision": mean("precision"),
        "any_hit_rate": mean("any_hit"),
        "avg_retrieved": mean("n_retrieved"),
        "avg_gold": mean("n_gold"),
        "claims_using_a_second_round": gap_rounds_used,
        "fact_check_article_domain_retrieved": fc_domain_hits,
    }


async def main_async(args) -> int:
    path = os.path.join(args.data_dir, f"{args.split}.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Set --data-dir or AVERITEC_DIR.", file=sys.stderr)
        return 1

    all_claims = load_split(path)
    stats = describe(all_claims)
    print(f"{'=' * 70}\nAVeriTeC {args.split}: {path}\n{'=' * 70}")
    for key, value in stats.items():
        print(f"  {key:32} {value}")

    if stats["answers_scorable"] == 0:
        print("\nThis split has no scorable answers (test.json is the blind split).",
              file=sys.stderr)
        return 1

    corpus = build_corpus(all_claims)
    pool = evaluable(all_claims)[: args.limit]
    print(f"\nCorpus documents (distinct source URLs): {len(corpus)}")
    print(f"Claims evaluated: {len(pool)} (of {len(evaluable(all_claims))} evaluable)")

    llm = None
    if not args.no_llm:
        try:
            llm = get_default_llm_client(use_dummy_if_missing_key=False)
        except RuntimeError as e:
            print(f"\nERROR: {e}\nUse --no-llm for a rule-based baseline.", file=sys.stderr)
            return 1
    else:
        print("Mode: --no-llm (rule-based queries; the loop cannot produce gaps)")

    conditions = sorted({1, args.rounds})
    results = []
    for rounds in conditions:
        print(f"\n--- condition: rounds={rounds} ---")
        outcome = await run_condition(
            pool, corpus, rounds, llm,
            concurrency=args.concurrency, results_per_query=args.results_per_query,
            evidence_k=args.evidence_k,
        )
        results.append(outcome)
        print(f"  recall={outcome['recall']:.3f}  precision={outcome['precision']:.3f}  "
              f"any-hit={outcome['any_hit_rate']:.3f}  "
              f"avg retrieved={outcome['avg_retrieved']:.1f}")
        print(f"  claims that used a 2nd round: {outcome['claims_using_a_second_round']}"
              f"   errors: {outcome['errors']}")

    if args.control_k:
        print(f"\n--- control: rounds=1 at k={args.control_k} (volume-matched) ---")
        control = await run_condition(
            pool, corpus, 1, llm,
            concurrency=args.concurrency, results_per_query=args.results_per_query,
            evidence_k=args.control_k,
        )
        control["label"] = f"rounds=1 (k={args.control_k}, volume control)"
        results.append(control)
        print(f"  recall={control['recall']:.3f}  precision={control['precision']:.3f}  "
              f"any-hit={control['any_hit_rate']:.3f}  "
              f"avg retrieved={control['avg_retrieved']:.1f}")

    print(f"\n{'=' * 70}\nLOOP EFFECT (closed corpus)\n{'=' * 70}")
    print(f"{'rounds':>7} {'k':>4} {'recall':>9} {'precision':>10} {'any-hit':>9} {'retrieved':>10}")
    for outcome in results:
        print(f"{outcome['rounds']:>7} {str(outcome.get('evidence_k') or '-'):>4} "
              f"{outcome['recall']:>9.3f} "
              f"{outcome['precision']:>10.3f} {outcome['any_hit_rate']:>9.3f} "
              f"{outcome['avg_retrieved']:>10.1f}")
    if len(results) > 1:
        delta = results[-1]["recall"] - results[0]["recall"]
        print(f"\nrecall delta (rounds={results[-1]['rounds']} vs 1): {delta:+.3f}")
        if delta <= 0:
            print("  NOTE: the extra round did not improve recall on this corpus.")

    out_path = os.path.join(_HERE, f"metrics_retrieval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump({
            "split": args.split,
            "condition": "offline closed corpus (NOT comparable to live open-web search)",
            "dataset_stats": stats,
            "corpus_documents": len(corpus),
            "claims_evaluated": len(pool),
            "results_per_query": args.results_per_query,
            "llm": "none (rule-based queries)" if args.no_llm else "configured default",
            "conditions": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "train", "test"])
    parser.add_argument("--data-dir", default=default_data_dir())
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--results-per-query", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--evidence-k", type=int, default=None,
                        help="Evidence kept per round (default: MAX_EVIDENCE_SOURCES)")
    parser.add_argument("--control-k", type=int, default=None,
                        help="Also run rounds=1 at this k, to match the multi-round "
                             "retrieved-document count and isolate targeting from volume")
    parser.add_argument("--no-llm", action="store_true",
                        help="Rule-based queries; free, but the loop cannot produce gaps")
    sys.exit(asyncio.run(main_async(parser.parse_args())))
