# NewsScope Claim Understanding Evaluation

This folder contains an end-to-end evaluation pipeline for a **Claim Understanding Agent** on the **NewsScope Benchmark**.

## Dependencies

From the repo root with the project venv active:

```bash
pip install httpx trafilatura beautifulsoup4 pyyaml scipy rouge-score openai python-dotenv
```

Optional (BERTScore; needs a working PyTorch stack in the venv):

```bash
pip install bert-score torch
```

If BERTScore is unavailable, `score` falls back to ROUGE-L matching.

## Quickstart (planned CLI)

From the repo root (use the project venv so `OPENAI_API_KEY` and deps match):

```bash
source .venv/bin/activate
# Fetch + cache article text (strict: live -> wayback -> skip)
python -m eval.src.cli fetch --split test_indomain

# Run your Claim Understanding Agent on fetch-success articles only
python -m eval.src.cli run-agent --split test_indomain

# Score predicted claim sets vs gold claim_text
python -m eval.src.cli score --split test_indomain

# End-to-end
python -m eval.src.cli all --split test_indomain
```

## Data location

This repo ships the NewsScope dataset under:

- `docs/dataset/NewsScope/train.jsonl`
- `docs/dataset/NewsScope/test_indomain.jsonl`
- `docs/dataset/NewsScope/test_oos.jsonl`

## Strict integrity rule

If article text cannot be retrieved from live URL **or** Wayback Machine, the article is **skipped** and logged to `unfetchable_articles_log.json`. No snippet-only evaluation and no LLM-based reconstruction is allowed.

