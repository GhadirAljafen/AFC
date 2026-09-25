from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.src.io import read_jsonl, write_jsonl
from eval.src.paths import get_paths
from eval.src.fetch.cache import CachePaths, read_text_if_exists


async def run_agent_on_split(split: str, limit: int | None = None) -> Path:
    """
    Run ClaimUnderstandingAgent on fetch-success articles only.

    Inputs:
      - eval/data/intermediate/newsscope_<split>_normalized.jsonl
      - eval/data/cache/{live,wayback}/... extracted texts (produced by fetch stage)

    Output:
      - eval/data/intermediate/predictions_<split>.jsonl
    """
    from factcheck_agent.llm_client import get_default_llm_client
    from factcheck_agent.agents.claim_understanding import ClaimUnderstandingAgent

    paths = get_paths()
    norm_path = paths.intermediate_dir / f"newsscope_{split}_normalized.jsonl"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalized split. Run ingest first: {norm_path}")

    fetched_index_path = paths.intermediate_dir / f"fetched_index_{split}.jsonl"
    if not fetched_index_path.exists():
        raise FileNotFoundError(f"Missing fetched index. Run fetch first: {fetched_index_path}")

    # Prefer a real LLM if configured; otherwise allow dummy so the pipeline can run end-to-end.
    llm = get_default_llm_client(use_dummy_if_missing_key=True)
    agent = ClaimUnderstandingAgent(llm=llm)

    cache = CachePaths(base_dir=paths.cache_dir)

    fetched_by_id: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(fetched_index_path):
        eid = str(row.get("example_id"))
        if eid:
            fetched_by_id[eid] = row

    out_rows: List[Dict[str, Any]] = []

    for i, ex in enumerate(read_jsonl(norm_path)):
        if limit is not None and i >= limit:
            break

        eid = str(ex.get("example_id"))
        fetched = fetched_by_id.get(eid)
        if not fetched:
            continue

        url = fetched.get("url") if isinstance(fetched.get("url"), str) else None
        retrieval_source = fetched.get("retrieval_source") if isinstance(fetched.get("retrieval_source"), str) else None
        if not url or retrieval_source not in ("live", "wayback"):
            continue

        text = read_text_if_exists(cache.text_path(retrieval_source, url))
        if not text:
            continue

        t0 = time.time()
        detected = await agent.detect_claims(text=text, min_importance=0.5, normalize_detected=True)
        elapsed_s = time.time() - t0

        pred_claims = []
        for dc in detected:
            pred_claims.append(
                {
                    "pred_id": dc.id,
                    "claim_text": dc.raw_text,
                    "normalized_text": dc.normalized_text,
                    "importance": dc.importance,
                }
            )

        out_rows.append(
            {
                "example_id": eid,
                "split": split,
                "url": url,
                "domain": ex.get("domain"),
                "source": ex.get("source"),
                "retrieval_source": retrieval_source,
                "runtime_s": elapsed_s,
                "pred_claims": pred_claims,
            }
        )

    out_path = paths.intermediate_dir / f"predictions_{split}.jsonl"
    write_jsonl(out_path, out_rows)
    return out_path

