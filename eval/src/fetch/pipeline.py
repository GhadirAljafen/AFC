from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from eval.src.io import read_jsonl, write_jsonl
from eval.src.paths import get_paths
from eval.src.fetch.cache import CachePaths, read_text_if_exists, write_text
from eval.src.fetch.client import FetchConfig, Throttler, fetch_html
from eval.src.fetch.extract import extract_main_text
from eval.src.fetch.resolve import resolve_url_from_headline
from eval.src.fetch.wayback import lookup_closest_snapshot


@dataclass
class RetrievalResult:
    example_id: str
    split: str
    url: Optional[str]
    domain: Optional[str]
    source: Optional[str]
    retrieval_source: Optional[str]  # live|wayback|None
    status: str  # success|failed
    failure_stage: Optional[str]
    reason: Optional[str]
    extraction_method: Optional[str]
    char_count: Optional[int]
    url_resolution_method: Optional[str] = None


def _unfetchable_log_path() -> Path:
    paths = get_paths()
    return paths.reports_dir / "unfetchable_articles_log.json"


async def fetch_split(normalized_jsonl: Path, split: str, limit: int | None = None) -> None:
    """
    Fetch and extract text for each example in a normalized split JSONL.

    Produces:
    - cache HTML/text under eval/data/cache/{live,wayback}/...
    - retrieval_results.jsonl under eval/data/reports/
    - unfetchable_articles_log.json under eval/data/reports/
    """
    import httpx

    paths = get_paths()
    cache_paths = CachePaths(base_dir=paths.cache_dir)
    retrieval_rows: list[Dict[str, Any]] = []
    unfetchable: list[Dict[str, Any]] = []
    fetched_index: list[Dict[str, Any]] = []

    cfg = FetchConfig()
    throttler = Throttler(per_host_min_delay_s=cfg.per_host_min_delay_s)

    async with httpx.AsyncClient() as client:
        for i, ex in enumerate(read_jsonl(normalized_jsonl)):
            if limit is not None and i >= limit:
                break

            example_id = str(ex.get("example_id"))
            url = ex.get("url") if isinstance(ex.get("url"), str) else None
            domain = ex.get("domain") if isinstance(ex.get("domain"), str) else None
            source = ex.get("source") if isinstance(ex.get("source"), str) else None
            headline = ex.get("headline") if isinstance(ex.get("headline"), str) else None

            if not url:
                # URL is missing in some NewsScope files. Try non-LLM URL resolution via headline search.
                resolved = None
                if headline:
                    resolved = await resolve_url_from_headline(client=client, headline=headline)
                if resolved:
                    url = resolved.url
                    url_resolution_method = resolved.method
                else:
                    rr = RetrievalResult(
                        example_id=example_id,
                        split=split,
                        url=None,
                        domain=domain,
                        source=source,
                        retrieval_source=None,
                        status="failed",
                        failure_stage="url_resolve",
                        reason="empty_url_and_unresolved",
                        extraction_method=None,
                        char_count=None,
                        url_resolution_method=None,
                    )
                    retrieval_rows.append(asdict(rr))
                    unfetchable.append(asdict(rr))
                    continue
            else:
                url_resolution_method = None

            # Attempt 1: live
            cached_text = read_text_if_exists(cache_paths.text_path("live", url))
            if cached_text:
                rr = RetrievalResult(
                    example_id=example_id,
                    split=split,
                    url=url,
                    domain=domain,
                    source=source,
                    retrieval_source="live",
                    status="success",
                    failure_stage=None,
                    reason=None,
                    extraction_method="cache",
                    char_count=len(cached_text),
                    url_resolution_method=url_resolution_method,
                )
                retrieval_rows.append(
                    asdict(
                        rr
                    )
                )
                fetched_index.append({"example_id": example_id, "url": url, "retrieval_source": "live"})
                continue

            live_html: Optional[str] = None
            try:
                live_html = await fetch_html(client=client, url=url, cfg=cfg, throttler=throttler)
                write_text(cache_paths.html_path("live", url), live_html)
            except Exception as e:
                live_html = None
                live_err = str(e)
            else:
                live_err = None

            live_text = None
            extraction_method = None
            if live_html:
                extracted = extract_main_text(live_html, url=url)
                if extracted:
                    live_text = extracted.text
                    extraction_method = extracted.method
                    write_text(cache_paths.text_path("live", url), live_text)

            if live_text:
                rr = RetrievalResult(
                    example_id=example_id,
                    split=split,
                    url=url,
                    domain=domain,
                    source=source,
                    retrieval_source="live",
                    status="success",
                    failure_stage=None,
                    reason=None,
                    extraction_method=extraction_method,
                    char_count=len(live_text),
                    url_resolution_method=url_resolution_method,
                )
                retrieval_rows.append(
                    asdict(
                        rr
                    )
                )
                fetched_index.append({"example_id": example_id, "url": url, "retrieval_source": "live"})
                continue

            # Attempt 2: wayback
            cached_wb_text = read_text_if_exists(cache_paths.text_path("wayback", url))
            if cached_wb_text:
                rr = RetrievalResult(
                    example_id=example_id,
                    split=split,
                    url=url,
                    domain=domain,
                    source=source,
                    retrieval_source="wayback",
                    status="success",
                    failure_stage=None,
                    reason=None,
                    extraction_method="cache",
                    char_count=len(cached_wb_text),
                    url_resolution_method=url_resolution_method,
                )
                retrieval_rows.append(
                    asdict(
                        rr
                    )
                )
                fetched_index.append({"example_id": example_id, "url": url, "retrieval_source": "wayback"})
                continue

            snapshot = await lookup_closest_snapshot(client=client, url=url)
            if not snapshot:
                rr = RetrievalResult(
                    example_id=example_id,
                    split=split,
                    url=url,
                    domain=domain,
                    source=source,
                    retrieval_source=None,
                    status="failed",
                    failure_stage="wayback_lookup",
                    reason=live_err or "wayback_not_found",
                    extraction_method=None,
                    char_count=None,
                    url_resolution_method=url_resolution_method,
                )
                retrieval_rows.append(asdict(rr))
                unfetchable.append(asdict(rr))
                continue

            wb_html: Optional[str] = None
            try:
                wb_html = await fetch_html(client=client, url=snapshot.snapshot_url, cfg=cfg, throttler=throttler)
                write_text(cache_paths.html_path("wayback", url), wb_html)
            except Exception as e:
                wb_html = None
                wb_err = str(e)
            else:
                wb_err = None

            wb_text = None
            wb_method = None
            if wb_html:
                extracted = extract_main_text(wb_html, url=snapshot.snapshot_url)
                if extracted:
                    wb_text = extracted.text
                    wb_method = extracted.method
                    write_text(cache_paths.text_path("wayback", url), wb_text)

            if wb_text:
                rr = RetrievalResult(
                    example_id=example_id,
                    split=split,
                    url=url,
                    domain=domain,
                    source=source,
                    retrieval_source="wayback",
                    status="success",
                    failure_stage=None,
                    reason=None,
                    extraction_method=wb_method,
                    char_count=len(wb_text),
                    url_resolution_method=url_resolution_method,
                )
                retrieval_rows.append(
                    asdict(
                        rr
                    )
                )
                fetched_index.append({"example_id": example_id, "url": url, "retrieval_source": "wayback"})
                continue

            rr = RetrievalResult(
                example_id=example_id,
                split=split,
                url=url,
                domain=domain,
                source=source,
                retrieval_source=None,
                status="failed",
                failure_stage="wayback_extract" if wb_html else "wayback_fetch",
                reason=wb_err or "extraction_failed",
                extraction_method=None,
                char_count=None,
                url_resolution_method=url_resolution_method,
            )
            retrieval_rows.append(asdict(rr))
            unfetchable.append(asdict(rr))

    # write logs
    retrieval_out = paths.reports_dir / f"retrieval_results_{split}.jsonl"
    write_jsonl(retrieval_out, retrieval_rows)

    fetched_index_out = paths.intermediate_dir / f"fetched_index_{split}.jsonl"
    write_jsonl(fetched_index_out, fetched_index)

    # append/update unfetchable json (array)
    unf_path = _unfetchable_log_path()
    unf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(unf_path.read_text(encoding="utf-8")) if unf_path.exists() else []
        if not isinstance(existing, list):
            existing = []
    except Exception:
        existing = []
    existing.extend(unfetchable)
    unf_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

