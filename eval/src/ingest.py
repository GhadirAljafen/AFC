from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from eval.src.io import read_jsonl, write_jsonl
from eval.src.paths import get_paths


class GoldClaim(TypedDict):
    claim_id: str
    claim_text: str
    evidence_from_article: str


class NormalizedExample(TypedDict):
    example_id: str
    split: str
    article_id: Optional[str]
    url: Optional[str]
    domain: Optional[str]
    source: Optional[str]
    headline: Optional[str]
    gold_claims: List[GoldClaim]


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _coerce_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return str(x).strip() or None


def _build_example_id(article_id: Optional[str], headline: Optional[str], gold_claims: List[GoldClaim]) -> str:
    if article_id:
        return article_id
    seed_parts: List[str] = []
    if headline:
        seed_parts.append(headline)
    # include first few claims to stabilize even if headline missing
    for gc in gold_claims[:3]:
        seed_parts.append(gc.get("claim_id") or "")
        seed_parts.append(gc.get("claim_text") or "")
    seed = "\n".join(seed_parts).strip()
    if not seed:
        seed = "missing"
    return f"ns_{_stable_hash(seed)}"


def _normalize_row(split: str, row: Dict[str, Any]) -> NormalizedExample:
    annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
    claims_raw = annotation.get("claims") if isinstance(annotation.get("claims"), list) else []

    gold_claims: List[GoldClaim] = []
    for c in claims_raw:
        if not isinstance(c, dict):
            continue
        claim_text = _coerce_str(c.get("claim_text")) or ""
        evidence = _coerce_str(c.get("evidence_from_article")) or ""
        claim_id = _coerce_str(c.get("claim_id")) or f"gold_{_stable_hash(claim_text + '|' + evidence)}"
        if claim_text:
            gold_claims.append(
                {
                    "claim_id": claim_id,
                    "claim_text": claim_text,
                    "evidence_from_article": evidence,
                }
            )

    article_id = _coerce_str(row.get("article_id"))
    url = _coerce_str(row.get("url"))
    # Some NewsScope files keep these empty at top-level; fallback to annotation when present.
    domain = _coerce_str(row.get("domain")) or _coerce_str(annotation.get("domain"))
    source = _coerce_str(row.get("source")) or _coerce_str(annotation.get("source"))
    headline = _coerce_str(annotation.get("headline"))

    example_id = _build_example_id(article_id=article_id, headline=headline, gold_claims=gold_claims)

    return {
        "example_id": example_id,
        "split": split,
        "article_id": article_id,
        "url": url,
        "domain": domain,
        "source": source,
        "headline": headline,
        "gold_claims": gold_claims,
    }


def ingest_split(split: str, limit: int | None = None) -> Path:
    """
    Read the raw NewsScope JSONL and write a normalized JSONL suitable for downstream stages.

    Output: eval/data/intermediate/newsscope_<split>_normalized.jsonl
    """
    paths = get_paths()
    input_path = paths.dataset_dir / f"{split}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing dataset split file: {input_path}")

    out_path = paths.intermediate_dir / f"newsscope_{split}_normalized.jsonl"

    rows_out: List[NormalizedExample] = []
    for i, row in enumerate(read_jsonl(input_path)):
        if limit is not None and i >= limit:
            break
        rows_out.append(_normalize_row(split=split, row=row))

    write_jsonl(out_path, rows_out)  # type: ignore[arg-type]
    return out_path

