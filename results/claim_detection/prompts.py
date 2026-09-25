"""
Claim Detection Prompts — Ablation Study
Evaluated on NewsScope benchmark (test_indomain: 80, test_oos: 60)
"""

PROMPT_A = """You are a fact-checking assistant. Analyze the following text and identify factual claims that are worth fact-checking.
CRITICAL: Each claim must contain ONLY ONE atomic fact. If a sentence contains multiple facts, split them into separate claims.
IMPORTANT OUTPUT LIMIT:
- Return AT MOST 20 claims total.
- Prefer the most check-worthy, specific, and verifiable claims (numbers, dates, named entities, concrete events).
- If you find more than 20 candidate claims, select the best 20 by importance.
OUTPUT STYLE (match benchmark "claim_text"):
- Each claim must be a short, standalone, decontextualized factual statement.
- Do NOT write narrative framing like "The article says..." / "The case alleges...".
- Prefer direct statements like "X happened on DATE" / "Person Y said Z" / "Charges were dropped because ...".
- Avoid hedging words unless present in the text.
A check-worthy claim is:
- A factual assertion that can be verified
- Specific enough to be fact-checked (has dates, numbers, names, locations)
- Not an opinion or subjective statement
- Not a question
- Contains ONLY ONE atomic fact (not multiple facts combined)
Examples:
- GOOD (atomic): "Saudi Arabia is the world's largest oil producer"
- GOOD (atomic): "Saudi Arabia aims to diversify its energy sources"
- BAD (compound): "Saudi Arabia, which is the world's largest oil producer, aims to diversify its energy sources"
  Split into:
    1. "Saudi Arabia is the world's largest oil producer"
    2. "Saudi Arabia aims to diversify its energy sources"
For each claim you find, provide:
1. The exact claim text (one atomic fact only)
2. An importance score (0.0-1.0) indicating how check-worthy it is
3. The sentence index (0-based) where it appears
Text to analyze:
{text}
Respond with a JSON array of objects, each with:
- "claim_text": the claim text (one atomic fact only)
- "sentence_index": the sentence index (0-based)
- "importance": a float between 0.0 and 1.0
Output format:
[
  {{"claim_text": "...", "sentence_index": 0, "importance": 0.8}},
  ...
]
Only include claims with importance >= {min_importance}. Output ONLY the JSON array, no other text."""


PROMPT_B = """You are a fact-checking assistant. Read the following text and extract the most important factual claims worth fact-checking.

STRICT RULES:
- Extract ONLY 2-4 of the most significant claims
- Each claim must be a complete, standalone factual statement
- Focus on the main facts: key events, decisions, numbers, and outcomes
- Do NOT split one fact into multiple claims
- Do NOT include minor details or background information

Text to analyze:
{text}

Respond with a JSON array:
[
  {{"claim_text": "...", "sentence_index": 0, "importance": 0.9}},
  ...
]
Output ONLY the JSON array, no other text."""


PROMPT_C = """You are an expert fact-checker. Your task is to identify claims from a news article that a fact-checker would prioritize verifying.

GUIDELINES:
- Extract 3-5 claims that represent the most verifiable facts in the article
- Each claim must be self-contained and specific (include names, dates, numbers when available)
- Prioritize claims that are central to the story, not peripheral details
- Combine related facts into one claim if they form a single verifiable assertion
- Write each claim as a journalist would state a fact

Text to analyze:
{text}

Respond with a JSON array:
[
  {{"claim_text": "...", "sentence_index": 0, "importance": 0.9}},
  ...
]
Output ONLY the JSON array, no other text."""


# ORIGINAL RUN (kept for provenance — superseded by RESULTS_CLEAN below).
# This run was CONFOUNDED: arm A used the base agent while arms B/C used a
# separate PromptAgent override that (a) silently returned [] on any error,
# (b) dropped `importance` via wrong DetectedClaim field names, and (c) used
# different min_importance (A=0.5, B/C=0.0) and max_tokens (A=1000, B/C=500).
# Arm C was never run out-of-domain. Numbers are still broadly valid because
# the ROUGE-L matcher only compares claim text, but the A-vs-B-vs-C comparison
# was not single-variable. See RESULTS_CLEAN for the corrected ablation.
RESULTS = {
    "metric": "ROUGE-L F1, matching threshold >= 0.40",
    "model": "GPT-4o-mini via OpenAI API",
    "temperature": 0.3,
    "max_tokens": {"A": 1000, "B": 500, "C": 500},
    "indomain": {
        "n_articles": 80,
        "prompt_a": {"precision": 0.304, "recall": 0.585, "f1": 0.386, "avg_predictions": 6.0},
        "prompt_b": {"precision": 0.407, "recall": 0.529, "f1": 0.453, "avg_predictions": 3.5},
        "prompt_c": {"precision": 0.365, "recall": 0.554, "f1": 0.434, "avg_predictions": 4.5},
    },
    "out_of_domain": {
        "n_articles": 60,
        "prompt_a": {"precision": 0.196, "recall": 0.381, "f1": 0.254},
        "prompt_b": {"precision": 0.256, "recall": 0.358, "f1": 0.296},
    },
    "notes": [
        "BERTScore unavailable due to PyTorch 2.2.2 / NumPy 2.x conflict — ROUGE-L used instead",
        "Input: headline + key_points (article URLs were empty in dataset)",
        "Prompt B best F1 on indomain (0.453), but largest drop on oos (-34%)",
        "Prompt A most stable across domains",
    ]
}


# CLEAN RE-RUN (2026-06-09) — single-variable ablation. Every arm now runs the
# SAME extraction logic (base ClaimUnderstandingAgent.detect_claims) and differs
# ONLY by its prompt template. Confounds above were removed: unified
# min_importance=0.0 and max_tokens=1000 for all arms, the field-name bug fixed,
# detection failures counted rather than silently scored as zero (0/140 failed).
# Removing the confounds shifted every in-domain F1 by <= 0.016, confirming the
# prompt — not the confounds — was the driving variable. Full write-up: RESULTS.md.
RESULTS_CLEAN = {
    "metric": "ROUGE-L F1, matching threshold >= 0.40",
    "model": "gpt-4.1-mini via OpenAI API (verified default; original run noted "
             "GPT-4o-mini, so old-vs-clean is not strictly model-controlled)",
    "temperature": 0.3,
    "max_tokens": {"A": 1000, "B": 1000, "C": 1000},
    "min_importance": 0.0,
    "normalize_detected": False,
    "indomain": {
        "n_articles": 80,
        "prompt_a": {"precision": 0.277, "recall": 0.602, "f1": 0.370},
        "prompt_b": {"precision": 0.408, "recall": 0.525, "f1": 0.452},
        "prompt_c": {"precision": 0.367, "recall": 0.552, "f1": 0.435},
    },
    "out_of_domain": {
        "n_articles": 60,
        "prompt_a": {"precision": 0.192, "recall": 0.400, "f1": 0.255},
        "prompt_b": {"precision": 0.254, "recall": 0.358, "f1": 0.296},
        "prompt_c": {"precision": 0.268, "recall": 0.408, "f1": 0.322},
    },
    "notes": [
        "BERTScore still unavailable (PyTorch 2.2.2 / NumPy 2.x conflict) — ROUGE-L used",
        "Input: headline + key_points (article URLs were empty in dataset)",
        "Prompt B best F1 on indomain (0.452); Prompt C close behind (0.435)",
        "CORRECTED FINDING: Prompt C — not A — is most robust across domains: best "
        "oos F1 (0.322) and smallest in->oos drop (A -31%, B -34%, C -26%)",
        "Prompt A over-generates: highest recall, lowest precision in both domains",
        "Arm C measured out-of-domain for the first time here",
    ]
}
