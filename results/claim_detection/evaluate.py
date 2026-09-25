"""
Claim Detection Evaluation Script
Usage: python results/claim_detection/evaluate.py --split test_indomain --n 80
"""
import asyncio, json, re, argparse
from factcheck_agent.llm_client import get_default_llm_client
from factcheck_agent.agents.claim_understanding import (
    ClaimUnderstandingAgent,
    ClaimDetectionError,
)
from rouge_score import rouge_scorer
from prompts import PROMPT_A, PROMPT_B, PROMPT_C


class PromptAgent(ClaimUnderstandingAgent):
    """Ablation arm: inherits the full (fixed) detect_claims from the base agent
    and overrides ONLY the prompt. This guarantees every arm shares identical
    parsing, sentence-splitting, error handling, and field names — so the prompt
    template is the sole independent variable in the experiment."""

    def __init__(self, llm, prompt_template):
        super().__init__(llm=llm)
        self.prompt_template = prompt_template

    def _build_detection_prompt(self, text, min_importance):
        return self.prompt_template.format(text=text, min_importance=min_importance)


def compute_metrics(pred_texts, gold_claims, scorer, threshold=0.40):
    matched_gold = set()
    matched_pred = set()
    for pi, pred in enumerate(pred_texts):
        for gi, gold in enumerate(gold_claims):
            if gi in matched_gold:
                continue
            s = scorer.score(gold, pred)
            if s['rougeL'].fmeasure >= threshold:
                matched_gold.add(gi)
                matched_pred.add(pi)
                break
    tp = len(matched_gold)
    fp = len(pred_texts) - len(matched_pred)
    fn = len(gold_claims) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


async def evaluate(split, n, prompts_to_run):
    llm = get_default_llm_client(use_dummy_if_missing_key=False)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    split_file = f"docs/dataset/NewsScope/{split}.jsonl"
    with open(split_file) as f:
        articles = [json.loads(l) for l in f][:n]

    prompt_map = {"A": PROMPT_A, "B": PROMPT_B, "C": PROMPT_C}
    # Every arm now runs the SAME extraction logic (the base agent's fixed
    # detect_claims) and differs only by its prompt template — a clean ablation.
    agents = {
        name: PromptAgent(llm=llm, prompt_template=prompt_map[name])
        for name in prompts_to_run
    }

    all_results = {name: [] for name in prompts_to_run}
    detection_failures = {name: 0 for name in prompts_to_run}

    for i, article in enumerate(articles):
        ann = article["annotation"]
        headline = ann.get("headline", "")
        key_points = ann.get("key_points", [])
        gold_claims = [c["claim_text"] for c in ann.get("claims", [])]
        input_text = headline + "\n" + "\n".join(key_points)
        print(f"[{i+1}/{len(articles)}] {headline[:45]}...")

        for name, agent in agents.items():
            try:
                predicted = await agent.detect_claims(
                    input_text, min_importance=0.0, normalize_detected=False
                )
            except ClaimDetectionError as e:
                detection_failures[name] += 1
                print(f"  ⚠️  Arm {name}: detection failed — {e}")
                continue
            pred_texts = [c.raw_text for c in predicted]
            if not pred_texts or not gold_claims:
                continue
            metrics = compute_metrics(pred_texts, gold_claims, scorer)
            all_results[name].append(metrics)

    def avg(results, key):
        return sum(r[key] for r in results) / len(results) if results else 0

    print(f"\n{'='*55}")
    print(f"RESULTS �{split} ({len(articles)} articles)")
    print(f"{'Metric':<12}", end="")
    for name in prompts_to_run:
        print(f" {'Prompt '+name:>12}", end="")
    print()
    print("-" * 55)
    for metric in ["precision", "recall", "f1"]:
        print(f"{metric.capitalize():<12}", end="")
        for name in prompts_to_run:
            print(f" {avg(all_results[name], metric):>12.3f}", end="")
        print()

    if any(detection_failures.values()):
        print("\nDetection failures (counted, not scored as zero):")
        for name in prompts_to_run:
            print(f"  Arm {name}: {detection_failures[name]}/{len(articles)}")

    output_file = f"results/claim_detection/metrics_{split}.json"
    with open(output_file, "w") as f:
        json.dump({
            "split": split,
            "n_articles": len(articles),
            "results": {
                name: {
                    "precision": avg(all_results[name], "precision"),
                    "recall":    avg(all_results[name], "recall"),
                    "f1":        avg(all_results[name], "f1"),
                    "scored_articles": len(all_results[name]),
                    "detection_failures": detection_failures[name],
                }
                for name in prompts_to_run
            }
        }, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test_indomain",
                        choices=["test_indomain", "test_oos"])
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--prompts", default="A,B,C")
    args = parser.parse_args()
    prompts_to_run = [p.strip() for p in args.prompts.split(",")]
    asyncio.run(evaluate(args.split, args.n, prompts_to_run))
