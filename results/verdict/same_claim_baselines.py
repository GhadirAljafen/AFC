"""
Re-measure the oracle and closed-corpus baselines on the SAME claims as the
live-evidence run, then compare them paired.

Why this is separate from live_evidence_verdict.py. That script scores the verdict
stage on live web evidence for whichever claims have been collected live so far,
and compares the result against baselines measured earlier on n=136. Those are
different samples, so the comparison would only hold if the live claims were of
average difficulty — which is an assumption, not a measurement. This re-runs both
baselines on exactly the same claim IDs and applies McNemar on identical claims.

The live sample grows across days as quota allows, so re-run this after every
batch: a sample that was representative at n=13 is not guaranteed to stay so.

The check mattered: it confirmed the sample is not unusually easy (closed-corpus
accuracy here brackets the n=136 figure), which is what makes the live-evidence
comparison in FINDINGS.md section L trustworthy.

Costs no search quota — the oracle needs no retrieval, and the closed corpus is a
local index.

Run-to-run variance
-------------------
FINDINGS.md section L.5 established that the verdict call runs at temperature 0.2 and
is not cached, so a single scoring run is not a measurement: the oracle baseline moved
0.764 -> 0.782 between two single runs, which was enough to change a stated conclusion.
`--repeat N` scores each baseline N times and reports per-run accuracy, the spread, and
a majority-vote label, which is the quotable figure.

The two baselines are not equally noisy. The oracle supplies gold evidence directly, so
its variance is the verdict call alone. The closed-corpus condition re-runs retrieval as
well, and query generation is temperature 0.3, so a re-run also sees a different
document set — its spread mixes both sources.

Usage:
    python results/verdict/same_claim_baselines.py --repeat 3
"""

import argparse, asyncio, json, os, statistics, sys  # noqa: E401
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "results", "evidence_retrieval"))
sys.path.insert(0, os.path.join(ROOT, "results", "verdict"))

from averitec import default_data_dir, evaluable, load_split
from error_propagation import gold_evidence
from evaluate_retrieval import build_corpus
from sweep_k import CLASSES, LABEL_MAP, macro_f1

from factcheck_agent.agents.retrieval import RetrievalAgent
from factcheck_agent.agents.reasoning_and_verdict import ReasoningAndVerdictAgent
from factcheck_agent.llm_client import get_default_llm_client
from factcheck_agent.models import Claim
from factcheck_agent.pipeline import FactCheckingPipeline
from factcheck_agent.search import CorpusSearchClient


async def score_oracle(claims, llm):
    """Gold evidence, no retrieval. Variance here is the verdict call alone."""
    agent = ReasoningAndVerdictAgent(llm=llm)
    rows = []
    for c in claims:
        ev = gold_evidence(c)
        if not ev:
            continue
        v = await agent.decide(Claim(id=c.claim_id, raw_text=c.claim), ev)
        rows.append((c.claim_id, LABEL_MAP[c.label], v.label))
    return rows


async def score_closed(claims, llm, corpus):
    """Closed-corpus retrieval + verdict.

    Note this condition has TWO sources of run-to-run variance, not one: query
    generation runs at temperature 0.3, so a re-run also retrieves a different
    document set. Its spread is therefore not comparable to the oracle's.
    """
    rows = []
    for c in claims:
        pipe = FactCheckingPipeline(
            llm=llm, max_iterations=1, initial_retrieval_k=12,
            use_llm_normalization=False, full_text_top_k=0,
            retrieval_agent=RetrievalAgent(
                llm=llm, search_client=CorpusSearchClient(corpus), max_results_per_query=5),
        )
        norm = pipe.claim_understanding.normalize_claim(Claim(id=c.claim_id, raw_text=c.claim))
        ev, ev_res, _ = await pipe.collect_evidence(norm)
        final = ev_res.selected_evidence if ev_res else ev
        v = await pipe.reasoning_and_verdict.decide(norm, final)
        rows.append((c.claim_id, LABEL_MAP[c.label], v.label))
    return rows


def vote(runs):
    """Majority-vote prediction per claim across runs, preserving run-1 order."""
    gold = {cid: g for run in runs for cid, g, _ in run}
    preds = {}
    for cid, _, _ in runs[0]:
        labels = [p for run in runs for c2, _, p in run if c2 == cid]
        preds[cid] = Counter(labels).most_common(1)[0][0]
    return [(cid, gold[cid], preds[cid]) for cid, _, _ in runs[0]]


async def main(args):
    live = json.load(open(f"{ROOT}/results/verdict/live_evidence_verdict_dev.json"))
    ids = [r["claim_id"] for r in live["per_claim"]]
    all_claims = load_split(os.path.join(default_data_dir(), "dev.json"))
    corpus = build_corpus(all_claims)
    by_id = {c.claim_id: c for c in evaluable(all_claims)}
    claims = [by_id[i] for i in ids if i in by_id]
    print(f"re-measuring baselines on the SAME {len(claims)} claims, "
          f"{args.repeat} run(s) each\n")

    llm = get_default_llm_client(use_dummy_if_missing_key=False)
    runs = {"oracle_gold": [], "closed_corpus": []}
    for attempt in range(args.repeat):
        runs["oracle_gold"].append(await score_oracle(claims, llm))
        runs["closed_corpus"].append(await score_closed(claims, llm, corpus))
        if args.repeat > 1:
            accs = {k: sum(1 for _, g, p in runs[k][-1] if g == p) / len(runs[k][-1])
                    for k in runs}
            print(f"  run {attempt + 1}/{args.repeat}: "
                  f"oracle {accs['oracle_gold']:.3f}  closed {accs['closed_corpus']:.3f}")

    variance = None
    if args.repeat > 1:
        variance = {"runs": args.repeat, "per_run_accuracy": {}, "spread": {},
                    "unstable_claims": {}, "majority_vote_accuracy": {}}
        print(f"\nrun-to-run variance over {args.repeat} runs:")
        for name, rs in runs.items():
            per_run = [sum(1 for _, g, p in r if g == p) / len(r) for r in rs]
            maps = [{cid: p for cid, _, p in r} for r in rs]
            common = set.intersection(*[set(m) for m in maps])
            unstable = sum(1 for cid in common if len({m[cid] for m in maps}) > 1)
            voted = vote(rs)
            variance["per_run_accuracy"][name] = [round(a, 4) for a in per_run]
            variance["spread"][name] = round(max(per_run) - min(per_run), 4)
            variance["unstable_claims"][name] = unstable
            variance["majority_vote_accuracy"][name] = round(
                sum(1 for _, g, p in voted if g == p) / len(voted), 4)
            print(f"  {name:>14}: per-run {[round(a, 3) for a in per_run]}  "
                  f"spread {variance['spread'][name]:.3f}  "
                  f"unstable {unstable}/{len(common)}  "
                  f"majority-vote {variance['majority_vote_accuracy'][name]:.3f}")
        print("  NOTE closed_corpus also re-runs retrieval (temperature-0.3 query")
        print("       generation), so its spread mixes retrieval and verdict variance;")
        print("       oracle_gold isolates the verdict call.")
        print("  -> figures below use the MAJORITY-VOTE label per claim.")

    out = {k: [(g, p) for _, g, p in vote(rs)] for k, rs in runs.items()}
    order = [cid for cid, _, _ in vote(runs["closed_corpus"])]

    gold = Counter(g for g, _ in out["oracle_gold"])
    maj = max(gold.values()) / len(out["oracle_gold"])
    print(f"{'condition':>26} {'accuracy':>9} {'macroF1':>8}   (n={len(claims)})")
    results = {}
    for name, pairs in out.items():
        acc = sum(1 for g, p in pairs if g == p) / len(pairs)
        f1 = macro_f1(pairs)
        results[name] = {"accuracy": acc, "macro_f1": f1}
        print(f"{name:>26} {acc:>9.3f} {f1:>8.3f}")
    print(f"{'majority baseline':>26} {maj:>9.3f}")
    print()
    for x in live["summaries"]:
        print(f"{'live ' + x['condition']:>26} {x['accuracy']:>9.3f} {x['macro_f1']:>8.3f}")
    print()
    print("on n=136 these baselines were: closed 0.544, oracle 0.728")
    print(f"on these {len(claims)} claims they are: closed {results['closed_corpus']['accuracy']:.3f}, "
          f"oracle {results['oracle_gold']['accuracy']:.3f}")

    sys.path.insert(0, os.path.join(ROOT, "results", "evidence_retrieval"))
    from fulltext_verdict import mcnemar
    live_by_id = {r["claim_id"]: r for r in live["per_claim"]}
    correct = {
        "closed_corpus": [g == p for g, p in out["closed_corpus"]],
        "oracle_gold":   [g == p for g, p in out["oracle_gold"]],
        "live_snippet":  [live_by_id[i]["gold"] == live_by_id[i]["snippet"]["pred"] for i in order],
        "live_fulltext": [live_by_id[i]["gold"] == live_by_id[i]["fulltext"]["pred"] for i in order],
    }
    print("\npaired McNemar (identical claims):")
    tests = {}
    for base in ("closed_corpus", "oracle_gold"):
        n = min(len(correct[base]), len(correct["live_fulltext"]))
        b, c, pv = mcnemar(correct[base][:n], correct["live_fulltext"][:n])
        sig = "n.s." if (pv is None or pv >= 0.05) else "SIGNIFICANT"
        print(f"  live_fulltext vs {base:>14}: {base}-only={b}  live-only={c}  "
              f"p={'n/a' if pv is None else round(pv, 5)}  {sig}")
        tests[f"live_fulltext_vs_{base}"] = {"base_only_right": b, "live_only_right": c, "p": pv}
    results["paired_tests"] = tests
    json.dump({"claim_ids": ids, "n": len(claims), "majority": maj,
               "run_to_run_variance": variance,
               "same_claim_baselines": results},
              open(f"{ROOT}/results/verdict/same_claim_baselines_dev.json", "w"), indent=2)
    print("\nsaved -> results/verdict/same_claim_baselines_dev.json")


_p = argparse.ArgumentParser()
_p.add_argument("--repeat", type=int, default=1,
                help="score each baseline this many times and report the spread plus "
                     "a majority-vote label; 1 run is not a measurement (FINDINGS.md L.5)")
asyncio.run(main(_p.parse_args()))
