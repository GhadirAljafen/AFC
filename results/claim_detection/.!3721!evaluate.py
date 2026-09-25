"""
Claim Detection Evaluation Script
Usage: python results/claim_detection/evaluate.py --split test_indomain --n 80
"""
import asyncio, json, re, argparse
from factcheck_agent.llm_client import get_default_llm_client
from factcheck_agent.agents.claim_understanding import ClaimUnderstandingAgent
from factcheck_agent.models import DetectedClaim
from rouge_score import rouge_scorer
from prompts import PROMPT_A, PROMPT_B, PROMPT_C


class PromptAgent(ClaimUnderstandingAgent):
    def __init__(self, llm, prompt_template):
        super().__init__(llm=llm)
        self.prompt_template = prompt_template

    async def detect_claims(self, text, min_importance=0.0):
        prompt = self.prompt_template.format(text=text, min_importance=min_importance)
        try:
            response = await self.llm.complete(prompt, temperature=0.3, max_tokens=500)
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if not match:
                return []
            data = json.loads(match.group())
            claims = []
            for item in data:
                claims.append(DetectedClaim(
                    id=f"c{len(claims)}",
                    raw_text=item["claim_text"],
                    normalized_text=item["claim_text"],
                    sentence_index=item.get("sentence_index", 0),
                    char_start=0,
                    char_end=len(item["claim_text"]),
                    importance_score=item.get("importance", 0.8)
                ))
            return claims
        except Exception:
            return []


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
    agents = {
        name: (ClaimUnderstandingAgent(llm=llm) if name == "A"
               else PromptAgent(llm=llm, prompt_template=prompt_map[name]))
        for name in prompts_to_run
    }

    all_results = {name: [] for name in prompts_to_run}

    for i, article in enumerate(articles):
        ann = article["annotation"]
        headline = ann.get("headline", "")
        key_points = ann.get("key_points", [])
        gold_claims = [c["claim_text"] for c in ann.get("claims", [])]
        input_text = headline + "\n" + "\n".join(key_points)
        print(f"[{i+1}/{len(articles)}] {headline[:45]}...")

        for name, agent in agents.items():
            predicted = await agent.detect_claims(input_text, normalize_detected=False)
            pred_texts = [c.raw_text for c in predicted]
            if not pred_texts or not gold_claims:
                continue
            metrics = compute_metrics(pred_texts, gold_claims, scorer)
            all_results[name].append(metrics)

    def avg(results, key):
        return sum(r[key] for r in results) / len(results) if results else 0

    print(f"\n{'='*55}")
