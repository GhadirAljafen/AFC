"""
Stage 3: is the evidence genuinely marginal, or was the oracle under-serving it?

Phase B1 ruled out the prompt as the cause of over-abstention. The remaining
leading hypothesis was that the evidence is genuinely too thin to judge — that
abstaining is defensible rather than defective.

Checking that surfaced a flaw in the earlier harness. `error_propagation.gold_evidence`
passes each annotated answer's **text only**, discarding the QUESTION it answers.
AVeriTeC evidence is question-answer pairs, and an answer alone can be close to
meaningless:

    question: "Where was the claim first published"
    answer:   "It was first published on Sccopertino"

Sent as bare text, "It was first published on Sccopertino" loses the context that
makes it evidence. So the §8 "oracle" was never a true oracle, and its 27.9%
abstention may be partly an artifact of the harness rather than a property of the
verdict stage.

Three conditions, paired, same claims, original prompt throughout:

  A  answer_only   replicates the existing oracle baseline
  B  question+answer   the actual gold evidence, properly framed
  C  +justification    ALSO includes the fact-checker's own justification

**Condition C is a LEAKY diagnostic ceiling, not a fair evaluation.** The
justification states the reasoning behind the gold verdict, so a model given it is
close to being told the answer. It is included only to bound what is achievable:
if the stage still abstains even then, the limit is the model, not the evidence.
C must never be reported as system performance.

Reading the outcome:
  B >> A   the harness was under-serving evidence; abstention was partly artifactual
  C >> B   the evidence genuinely lacks what is needed; abstention is defensible
  C ~= B ~= A   the limit is the model or prompt, not the evidence

Usage:
    python results/verdict/evidence_sufficiency.py --limit 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter
import re as _re
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "results", "evidence_retrieval"))

from averitec import AveritecClaim, default_data_dir, evaluable, load_split  # noqa: E402
from fulltext_verdict import mcnemar  # noqa: E402
from sweep_k import CLASSES, LABEL_MAP, macro_f1  # noqa: E402

from factcheck_agent.agents.reasoning_and_verdict import ReasoningAndVerdictAgent  # noqa: E402
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from factcheck_agent.models import Claim, EvidenceSnippet  # noqa: E402

CONDITIONS = ("answer_only", "question_answer", "plus_justification")

# A justification "leaks" when it states the verdict or its polarity outright,
# rather than only supplying the reasoning. 47% of dev justifications do. Results
# for condition C must be split on this, or the gain cannot be distinguished from
# simply being told the answer.
_LEAK = _re.compile(
    r"\b(refut|support|debunk|false|true|correct|incorrect|misleading|not enough|insufficient)",
    _re.I,
)


def justification_leaks(claim: AveritecClaim) -> bool:
    return bool(claim.justification and _LEAK.search(claim.justification))


def build_evidence(claim: AveritecClaim, condition: str) -> List[EvidenceSnippet]:
    """Render the gold evidence under one framing."""
    snippets: List[EvidenceSnippet] = []
    for i, a in enumerate(claim.scorable_answers):
        if condition == "answer_only":
            text = a.answer
        else:
            # Pair each answer with the question it answers.
            text = f"Q: {a.question}\nA: {a.answer}" if a.question else a.answer
        snippets.append(EvidenceSnippet(
            id=a.source_url or f"gold-{i}",
            source="gold",
            text=text,
            score=1.0,
            metadata={"url": a.source_url, "domain": a.domain, "rank": "1"},
        ))

    if condition == "plus_justification" and claim.justification:
        # LEAKY. Ceiling diagnostic only — see module docstring.
        snippets.append(EvidenceSnippet(
            id=f"justification-{claim.claim_id}",
            source="fact-checker-justification",
            text=claim.justification,
            score=1.0,
            metadata={"leaky": "true"},
        ))
    return snippets


async def run(claims, llm, concurrency: int):
    agents = {c: ReasoningAndVerdictAgent(llm=llm, record_decisions=True) for c in CONDITIONS}
    sem = asyncio.Semaphore(concurrency)
    rows: List[Dict] = []

    async def one(claim: AveritecClaim):
        gold = LABEL_MAP.get(claim.label)
        if not gold:
            return
        normalized = Claim(id=claim.claim_id, raw_text=claim.claim)
        row = {"claim_id": claim.claim_id, "gold": gold,
               "justification_leaks": justification_leaks(claim)}
        async with sem:
            for cond in CONDITIONS:
                evidence = build_evidence(claim, cond)
                if not evidence:
                    return
                verdict = await agents[cond].decide(normalized, evidence)
                row[cond] = {
                    "pred": verdict.label,
                    "confidence": verdict.confidence,
                    "chars": sum(len(e.text) for e in evidence),
                    "n_evidence": len(evidence),
                }
        rows.append(row)

    await asyncio.gather(*(one(c) for c in claims))
    return rows


def summarise(rows, cond: str) -> Dict:
    pairs = [(r["gold"], r[cond]["pred"]) for r in rows]
    preds = Counter(p for _, p in pairs)
    per_class = {}
    for cls in CLASSES:
        n = sum(1 for g, _ in pairs if g == cls)
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        per_class[cls] = {"n": n, "recall": round(tp / n, 3) if n else None}
    errors = sum(1 for g, p in pairs if g != p)
    false_nei = sum(1 for g, p in pairs if p == "NOT_ENOUGH_INFO" and g != "NOT_ENOUGH_INFO")
    return {
        "condition": cond, "n": len(pairs),
        "accuracy": sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0,
        "macro_f1": macro_f1(pairs),
        "abstention": preds.get("NOT_ENOUGH_INFO", 0) / len(pairs) if pairs else 0,
        "false_abstentions": false_nei,
        "errors": errors,
        "mean_chars": statistics.mean([r[cond]["chars"] for r in rows]) if rows else 0,
        "mean_confidence": statistics.mean([r[cond]["confidence"] for r in rows]) if rows else 0,
        "pred_distribution": dict(preds),
        "per_class_recall": per_class,
    }


async def main_async(args) -> int:
    claims = evaluable(load_split(os.path.join(args.data_dir, f"{args.split}.json")))[: args.limit]
    print(f"{'=' * 92}\nIS THE EVIDENCE GENUINELY MARGINAL?  (all conditions use gold evidence)\n{'=' * 92}")
    print(f"claims={len(claims)}  prompt=original  paired (same claims, only evidence framing varies)")
    print("NOTE: 'plus_justification' is a LEAKY ceiling diagnostic, not system performance.\n")

    llm = get_default_llm_client(use_dummy_if_missing_key=False)
    rows = await run(claims, llm, args.concurrency)
    summaries = [summarise(rows, c) for c in CONDITIONS]

    gold_counts = Counter(r["gold"] for r in rows)
    majority = max(gold_counts.values()) / len(rows) if rows else 0

    print(f"{'=' * 92}\nRESULTS (n={len(rows)})\n{'=' * 92}")
    print(f"{'condition':>20} {'chars':>7} {'accuracy':>9} {'macroF1':>8} {'abstain':>8} "
          f"{'falseNEI':>9} | per-class recall")
    for s in summaries:
        pc = "  ".join(f"{c[:4]}={s['per_class_recall'][c]['recall']}" for c in CLASSES)
        print(f"{s['condition']:>20} {s['mean_chars']:>7.0f} {s['accuracy']:>9.4f} "
              f"{s['macro_f1']:>8.4f} {s['abstention']:>7.1%} {s['false_abstentions']:>9} | {pc}")
    print(f"{'majority baseline':>20} {'':>7} {majority:>9.4f}")

    print(f"\n{'=' * 92}\nPAIRED TESTS (McNemar exact)\n{'=' * 92}")
    tests = {}
    for a, b in (("question_answer", "answer_only"), ("plus_justification", "question_answer")):
        ca = [r["gold"] == r[a]["pred"] for r in rows]
        cb = [r["gold"] == r[b]["pred"] for r in rows]
        n_b, n_c, p = mcnemar(cb, ca)   # cb = baseline, ca = candidate
        sig = "n.s." if (p is None or p >= 0.05) else "SIGNIFICANT"
        tests[f"{a}_vs_{b}"] = {"baseline_only_right": n_b, "candidate_only_right": n_c, "p": p}
        print(f"  {a:>20} vs {b:<20} {b}-only-right={n_b}  {a}-only-right={n_c}  "
              f"p={'n/a' if p is None else round(p, 5)}  {sig}")

    a0, b0, c0 = summaries
    print(f"\n{'=' * 92}\nINTERPRETATION\n{'=' * 92}")
    print(f"  answer_only      -> question_answer : abstention {a0['abstention']:.1%} -> {b0['abstention']:.1%}"
          f"   accuracy {a0['accuracy']:.4f} -> {b0['accuracy']:.4f}")
    print(f"  question_answer  -> +justification  : abstention {b0['abstention']:.1%} -> {c0['abstention']:.1%}"
          f"   accuracy {b0['accuracy']:.4f} -> {c0['accuracy']:.4f}")
    qa_gain = b0["accuracy"] - a0["accuracy"]
    just_gain = c0["accuracy"] - b0["accuracy"]
    if qa_gain > 0.03:
        print("\n  --> Question framing matters: the earlier oracle was UNDER-SERVING the")
        print("      evidence, so §8's abstention figure was partly a harness artifact.")
    if just_gain > 0.05:
        print("  --> The justification adds a lot, so the QA evidence genuinely lacks what")
        print("      is needed to judge. Abstaining on it is defensible, not defective.")
    elif abs(just_gain) <= 0.05:
        print("  --> Even the fact-checker's own justification barely helps: the limit is")
        print("      the model or the prompt, NOT evidence sufficiency.")

    # --- does the justification gain survive when it does NOT leak the verdict? ---
    clean = [r for r in rows if not r["justification_leaks"]]
    leaky = [r for r in rows if r["justification_leaks"]]
    print(f"\n{'=' * 92}\nSPLITTING CONDITION C BY LEAKAGE\n{'=' * 92}")
    print(f"  justifications that name the verdict outright: {len(leaky)}/{len(rows)}"
          f" ({100 * len(leaky) / len(rows):.0f}%)")
    split_out = {}
    for name, subset in (("leaky", leaky), ("clean", clean)):
        if not subset:
            continue
        qa = sum(1 for r in subset if r["gold"] == r["question_answer"]["pred"]) / len(subset)
        pj = sum(1 for r in subset if r["gold"] == r["plus_justification"]["pred"]) / len(subset)
        cb = [r["gold"] == r["question_answer"]["pred"] for r in subset]
        ca = [r["gold"] == r["plus_justification"]["pred"] for r in subset]
        n_b, n_c, p = mcnemar(cb, ca)
        sig = "n.s." if (p is None or p >= 0.05) else "SIGNIFICANT"
        print(f"  {name:>6} (n={len(subset):>3}): question_answer {qa:.4f} -> "
              f"+justification {pj:.4f}  ({pj - qa:+.4f})  "
              f"McNemar {n_b}/{n_c} p={'n/a' if p is None else round(p, 5)}  {sig}")
        split_out[name] = {"n": len(subset), "qa_accuracy": qa, "plus_just_accuracy": pj,
                           "delta": pj - qa, "p": p,
                           "qa_only_right": n_b, "plus_just_only_right": n_c}
    if "clean" in split_out and (split_out["clean"]["p"] or 1) < 0.05:
        print("\n  --> The gain survives on non-leaking justifications, so supplying the")
        print("      REASONING (not the answer) genuinely helps. The evidence lacks the")
        print("      inferential bridge, and abstaining on it is partly defensible.")
    elif "clean" in split_out:
        print("\n  --> On non-leaking justifications the gain is NOT significant, so most of")
        print("      condition C's advantage was leakage. It does not show the evidence")
        print("      lacks reasoning; treat 0.904 as an artifact.")

    out = os.path.join(_HERE, f"evidence_sufficiency_{args.split}.json")
    with open(out, "w") as f:
        json.dump({
            "split": args.split, "claims_scored": len(rows),
            "design": "paired; gold evidence under three framings, original prompt",
            "warning": "plus_justification is LEAKY (contains the fact-checker's rationale) "
                       "and is a ceiling diagnostic only, never system performance",
            "majority_baseline": majority,
            "summaries": summaries, "mcnemar": tests,
            "leakage_split": split_out, "per_claim": rows,
        }, f, indent=2)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--concurrency", type=int, default=6)
    sys.exit(asyncio.run(main_async(p.parse_args())))
