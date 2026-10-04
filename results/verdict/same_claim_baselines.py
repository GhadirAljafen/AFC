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

Usage:
    python results/verdict/same_claim_baselines.py
"""

import asyncio, json, os, statistics, sys  # noqa: E401
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


async def main():
    live = json.load(open(f"{ROOT}/results/verdict/live_evidence_verdict_dev.json"))
    ids = [r["claim_id"] for r in live["per_claim"]]
    all_claims = load_split(os.path.join(default_data_dir(), "dev.json"))
    corpus = build_corpus(all_claims)
    by_id = {c.claim_id: c for c in evaluable(all_claims)}
    claims = [by_id[i] for i in ids if i in by_id]
    print(f"re-measuring baselines on the SAME {len(claims)} claims\n")

    llm = get_default_llm_client(use_dummy_if_missing_key=False)
    out = {}

    # ---- oracle: gold evidence, no retrieval ----
    agent = ReasoningAndVerdictAgent(llm=llm)
    pairs = []
    for c in claims:
        ev = gold_evidence(c)
        if not ev:
            continue
        v = await agent.decide(Claim(id=c.claim_id, raw_text=c.claim), ev)
        pairs.append((LABEL_MAP[c.label], v.label))
    out["oracle_gold"] = pairs
    ids_o = [c.claim_id for c in claims if gold_evidence(c)]

    # ---- closed corpus ----
    pairs = []
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
        pairs.append((LABEL_MAP[c.label], v.label))
    out["closed_corpus"] = pairs

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
    s = live["summaries"]
    for x in s:
        print(f"{'live ' + x['condition']:>26} {x['accuracy']:>9.3f} {x['macro_f1']:>8.3f}")
    print()
    print("on n=136 these baselines were: closed 0.544, oracle 0.728")
    print(f"on these {len(claims)} claims they are: closed {results['closed_corpus']['accuracy']:.3f}, "
          f"oracle {results['oracle_gold']['accuracy']:.3f}")
    # paired McNemar against the live conditions on identical claims
    sys.path.insert(0, os.path.join(ROOT, "results", "evidence_retrieval"))
    from fulltext_verdict import mcnemar
    live_by_id = {r["claim_id"]: r for r in live["per_claim"]}
    order = [c.claim_id for c in claims]
    correct = {
        "closed_corpus": [g == p for g, p in out["closed_corpus"]],
        "oracle_gold":   [g == p for g, p in out["oracle_gold"]],
        "live_snippet":  [live_by_id[i]["gold"] == live_by_id[i]["snippet"]["pred"] for i in order],
        "live_fulltext": [live_by_id[i]["gold"] == live_by_id[i]["fulltext"]["pred"] for i in order],
    }
    print("\npaired McNemar (identical claims):")
    tests = {}
    for base in ("closed_corpus", "oracle_gold"):
        b, c, pv = mcnemar(correct[base], correct["live_fulltext"])
        sig = "n.s." if (pv is None or pv >= 0.05) else "SIGNIFICANT"
        print(f"  live_fulltext vs {base:>14}: {base}-only={b}  live-only={c}  "
              f"p={'n/a' if pv is None else round(pv,5)}  {sig}")
        tests[f"live_fulltext_vs_{base}"] = {"base_only_right": b, "live_only_right": c, "p": pv}
    results["paired_tests"] = tests
    json.dump({"claim_ids": ids, "n": len(claims), "majority": maj,
               "same_claim_baselines": results},
              open(f"{ROOT}/results/verdict/same_claim_baselines_dev.json", "w"), indent=2)
    print("\nsaved -> results/verdict/same_claim_baselines_dev.json")


asyncio.run(main())
