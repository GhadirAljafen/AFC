# NewsScope Benchmark — Claim Understanding Agent Evaluation Plan

This document defines a reproducible evaluation plan and pipeline for a **Claim Understanding Agent** evaluated on the **NewsScope Benchmark Dataset**.

## Goals

- Evaluate how well the agent extracts **atomic, decontextualized claims** from authentic article text.
- Support in-domain (`test_indomain.jsonl`) and out-of-source (`test_oos.jsonl`) evaluation.
- Produce thesis-grade logs for error analysis (false positives, missed claims, near-misses, duplicates).

## Dataset summary

The dataset is provided as JSONL with schema (per line):

```json
{
  "article_id": "unique_id",
  "url": "https://...",
  "domain": "politics|health|science_env|business",
  "source": "Source Name",
  "annotation": {
    "headline": "...",
    "key_points": [...],
    "whos_involved": [{"name": "...", "role": "..."}],
    "how_it_unfolded": [{"date": "...", "event": "..."}],
    "claims": [
      {
        "claim_text": "Ground truth claim (atomic, decontextualized).",
        "evidence_from_article": "Exact snippet/quote from the article.",
        "claim_id": "unique_claim_id"
      }
    ]
  }
}
```

**Critical constraint:** the dataset does **not** include full article text (copyright).

## Strict text retrieval policy (academic integrity)

We evaluate **only** on articles where we successfully retrieve full **authentic** text.

**Hard rule:** DO NOT use an LLM (or any generative method) to reconstruct, expand, or hallucinate missing article text from `evidence_from_article` snippets.

Retrieval attempts, in order:

1. **Attempt 1 (Live URL):** fetch the live `url` and extract the main article text.
2. **Attempt 2 (Wayback fallback):** if live fetch/extraction fails, query the **Wayback Machine** for the closest available snapshot and fetch/extract that snapshot.
3. **Attempt 3 (Final):** if both fail, **SKIP** the article entirely and log it to `unfetchable_articles_log.json` with the reason and failure stage.

**Metric denominators** must include **only fetch-success articles**.

## Evaluation tasks and outputs

### Task A — Claim extraction quality (primary)

**Input:** authentic article text  
**Agent output:** list of extracted claims (strings), ideally atomic and check-worthy.  
**Ground truth:** `annotation.claims[*].claim_text`

We compute:

- **Claim-level matching** between predicted and gold claims
- **Precision/Recall/F1** over matched/unmatched claims
- **Quality diagnostics** (duplicates, overly broad claims, compound claims, entity errors)

### Optional Task B — Normalization quality (if your agent returns both raw + normalized)

If the agent outputs both extracted claim and a normalized form, score normalized claims against gold `claim_text`.

## Matching policy (how predicted claims align to gold claims)

Claim extraction is a **set matching** problem. Use a transparent, deterministic policy:

1. Compute similarity between each predicted claim \(p_i\) and each gold claim \(g_j\).
2. Produce a **one-to-one** alignment (e.g., Hungarian assignment) maximizing similarity.
3. Accept matches above a configurable threshold; the rest become:
   - **False positives (FP):** predicted claims with no match
   - **False negatives (FN):** gold claims with no match

Recommended similarity signals:

- **BERTScore F1** (semantic)
- **ROUGE-L F1** (lexical overlap)

Threshold strategy (starting point; tune on `train.jsonl`):

- Accept a match if `BERTScore_F1 >= 0.85` OR (`BERTScore_F1 >= 0.80` AND `ROUGE_L_F1 >= 0.30`)

Also track **near-misses**: `0.70 <= BERTScore_F1 < threshold` for error analysis.

## Metrics (report overall + by slice)

### Core extraction metrics

- **Precision / Recall / F1** (micro-averaged at claim level)
- **Average matched similarity** (mean BERTScore F1 over matched pairs)
- **Duplicates rate** (predicted claims that are near-identical to another predicted claim)
- **Compound-claim rate** (heuristic: multiple clauses/“and”/multiple numbers; optional)

### Retrieval coverage metrics (must be reported)

- **Fetch success rate**: `#fetched_success / #total_articles`
- **By source**: live vs wayback success counts
- **By split/domain/source**: success rates per group

### Slice reporting

Report all metrics:

- **Overall** per split (`test_indomain`, `test_oos`)
- **By domain** (`politics`, `health`, `science_env`, `business`)
- **By retrieval_source** (`live`, `wayback`)
- **By source outlet** (optional, if enough volume)

## Logging & artifacts (thesis-grade error analysis)

Write JSONL outputs so you can filter/group later without re-running the agent.

### 1) Retrieval logs

- `artifacts/retrieval/retrieval_results.jsonl`
  - `article_id`, `split`, `domain`, `source`, `url`
  - `retrieval_source`: `live|wayback`
  - `status`: `success|failed`
  - `failure_stage`: `live_fetch|live_extract|wayback_lookup|wayback_fetch|wayback_extract`
  - `reason`: short error string
  - `char_count`, `word_count`, `sha256` (hash of extracted text for reproducibility)

- `artifacts/retrieval/unfetchable_articles_log.json`
  - JSON array (or JSONL) of skipped articles with the same metadata + `reason`

### 2) Per-article evaluation records

- `artifacts/eval/per_article_results.jsonl`
  - `article_id`, `split`, `domain`, `retrieval_source`
  - `gold_claims`: list of `{claim_id, claim_text}`
  - `pred_claims`: list of `{pred_id, claim_text}`
  - `matches`: list of `{gold_claim_id, pred_id, bert_f1, rouge_l_f1}`
  - `false_positives`: list of `pred_id`
  - `missed_gold`: list of `claim_id`
  - `near_misses`: list of candidate pairs + scores

### 3) Aggregate reports

- `artifacts/reports/metrics_summary.json`
  - overall + per-slice metrics
- `artifacts/reports/error_buckets.json`
  - counts for error categories (entity errors, numeric errors, temporal errors, compound claims, duplicates)
- Optional: `artifacts/reports/summary.html` (quick browsing)

## Proposed code structure

Add a dedicated evaluation package (example layout):

```
eval/
  README.md
  pyproject.toml (or requirements.txt)
  src/
    newscope/
      io.py                 # read/write jsonl, schema validation
      fetch/
        live.py             # fetch live url (http client, retries, headers)
        wayback.py          # wayback lookup + snapshot fetch
        extract.py          # main-text extraction (trafilatura/readability)
        cache.py            # disk cache keyed by article_id + retrieval_source
      agent_runner.py       # adapter: call your Claim Understanding Agent
      metrics/
        similarity.py       # bertscore/rouge scoring
        matching.py         # assignment + thresholding
        aggregate.py        # slice metrics
      logging/
        retrieval_log.py
        eval_log.py
      cli.py                # `python -m eval ...` entrypoints
  artifacts/                # ignored by git; generated outputs
```

## Libraries (recommended)

- **HTTP + retries:** `httpx`, `tenacity`
- **Extraction:** `trafilatura` (or `readability-lxml`), `beautifulsoup4` (optional)
- **Similarity:** `bert-score`, `rouge-score`
- **Assignment:** `scipy` (Hungarian), or implement greedy matching first and upgrade later
- **Data:** `pydantic` (optional schema validation), `orjson` (optional)
- **Experiment tracking (optional):** `pandas`, `matplotlib`, `seaborn`

## Pipeline steps (end-to-end)

1. **Ingest dataset split(s)** (`test_indomain.jsonl`, `test_oos.jsonl`).
2. **Retrieve authentic text** using strict policy:
   - live fetch → extract
   - else wayback lookup → fetch snapshot → extract
   - else skip + append to `unfetchable_articles_log.json`
3. **Cache retrieved texts** (by `article_id` + `retrieval_source`) to make runs reproducible.
4. **Run agent** on the retrieved text to produce predicted claims.
5. **Score similarity** for all predicted–gold pairs.
6. **Match claims** (one-to-one assignment + thresholds).
7. **Write per-article results** JSONL and aggregate reports.
8. **Slice analysis** by domain + retrieval source; export summaries.

## Reproducibility & controls

- Fix random seeds for any stochastic components.
- Record model/version/prompt hash for the agent run.
- Store extracted text hashes (not necessarily full text) to allow reproducibility checks without redistributing copyrighted content.

