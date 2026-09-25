"""
Stage 2: can the full-text enrichment path actually get text from real URLs?

§8.1 showed evidence DEPTH is the highest-value lever available — doubling
snippet length raised verdict accuracy 0.544 -> 0.662. But that used longer
snippets from the offline corpus. It did not test `ContentFetcher`, which is the
machinery that would deliver depth in production, and which has never been run
against real URLs.

Depth is only worth pursuing if the pages are reachable. This audits that
directly against AVeriTeC's gold source URLs, which are exactly the kind of
evidence the live pipeline would need to fetch.

Measured per URL:
  * fetch outcome (ok / http error / timeout / binary / extraction failure)
  * extracted character count vs the gold answer it should contain
  * whether the URL is a Wayback snapshot (31.9% of dev gold URLs are)
  * whether extraction used trafilatura or fell back to BeautifulSoup

Costs no LLM calls and no search quota — HTTP only. Uses modest concurrency and
the fetcher's own timeouts to stay polite to third-party sites.

Usage:
    python results/evidence_retrieval/fetchability_audit.py --limit 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from averitec import GoldAnswer, default_data_dir, evaluable, load_split  # noqa: E402

from factcheck_agent.content_fetch import ContentFetcher, _looks_binary, extract_main_text  # noqa: E402

try:
    import httpx
except ImportError:
    httpx = None


class AuditFetcher(ContentFetcher):
    """ContentFetcher that reports WHY a fetch failed instead of just None."""

    async def audit(self, url: str) -> Dict:
        if httpx is None:
            return {"outcome": "no_httpx"}

        headers = {"User-Agent": self.user_agent,
                   "Accept": "text/html,application/xhtml+xml"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                status = resp.status_code
                raw = resp.content[: self.max_bytes]
                content_type = resp.headers.get("content-type", "").lower()
        except asyncio.TimeoutError:
            return {"outcome": "timeout"}
        except Exception as e:
            name = type(e).__name__
            return {"outcome": "connect_error", "detail": name}

        if status != 200:
            return {"outcome": f"http_{status // 100}xx", "status": status}
        if _looks_binary(raw):
            return {"outcome": "binary", "detail": raw[:4].decode("latin-1", "replace")}
        if content_type and "html" not in content_type and "text" not in content_type:
            return {"outcome": "non_html", "detail": content_type.split(";")[0]}

        html = raw.decode(resp.encoding or "utf-8", errors="replace")
        text = extract_main_text(html, url=url)
        if not text:
            return {"outcome": "extraction_failed", "status": status}

        method = "trafilatura"
        try:
            import trafilatura
            if not trafilatura.extract(html, url=url, favor_precision=True):
                method = "bs4_fallback"
        except Exception:
            method = "bs4_fallback"

        return {"outcome": "ok", "chars": len(text), "method": method, "status": status}


async def main_async(args) -> int:
    claims = evaluable(load_split(os.path.join(args.data_dir, f"{args.split}.json")))

    targets: List[GoldAnswer] = []
    for claim in claims:
        for answer in claim.scorable_answers:
            targets.append(answer)
            if len(targets) >= args.limit:
                break
        if len(targets) >= args.limit:
            break

    print(f"{'=' * 84}\nFETCHABILITY AUDIT — can full-text enrichment reach real evidence?\n{'=' * 84}")
    print(f"URLs sampled: {len(targets)} gold source URLs from AVeriTeC {args.split}")
    wayback = sum(1 for t in targets if "web.archive.org" in (t.raw_source_url or ""))
    print(f"Wayback-wrapped originals: {wayback} ({100*wayback/len(targets):.1f}%)")
    print(f"Concurrency: {args.concurrency}, timeout: {args.timeout}s\n")

    fetcher = AuditFetcher(cache_dir=os.path.join(_HERE, ".cache_audit"),
                           timeout=args.timeout, concurrency=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(answer: GoldAnswer) -> Dict:
        async with sem:
            # Prefer the Wayback copy when the original is dead: the archive is
            # what the annotators actually cited.
            result = await fetcher.audit(answer.source_url)
            used = "original"
            if result["outcome"] != "ok" and answer.raw_source_url != answer.source_url:
                alt = await fetcher.audit(answer.raw_source_url)
                if alt["outcome"] == "ok":
                    result, used = alt, "wayback"
            return {**result, "domain": answer.domain, "gold_chars": len(answer.answer),
                    "url_used": used,
                    "was_wayback": "web.archive.org" in (answer.raw_source_url or "")}

    results = await asyncio.gather(*(one(t) for t in targets))

    outcomes = Counter(r["outcome"] for r in results)
    ok = [r for r in results if r["outcome"] == "ok"]

    print(f"{'=' * 84}\nOUTCOMES\n{'=' * 84}")
    for outcome, n in outcomes.most_common():
        print(f"  {outcome:22} {n:>4}  {100*n/len(results):>5.1f}%")

    print(f"\nFETCH SUCCESS RATE: {len(ok)}/{len(results)} = {100*len(ok)/len(results):.1f}%")

    if ok:
        chars = [r["chars"] for r in ok]
        gold = [r["gold_chars"] for r in ok]
        print(f"\n{'=' * 84}\nEXTRACTED TEXT (successful fetches)\n{'=' * 84}")
        print(f"  extracted chars: median={statistics.median(chars):,.0f} "
              f"mean={statistics.mean(chars):,.0f} max={max(chars):,}")
        print(f"  gold answer chars (what a snippet carries): median={statistics.median(gold):,.0f}")
        print(f"  depth multiple: {statistics.median(chars)/max(1,statistics.median(gold)):.0f}x")
        print(f"  extraction method: {dict(Counter(r['method'] for r in ok))}")
        print(f"  recovered via Wayback fallback: {sum(1 for r in ok if r['url_used']=='wayback')}")

    wb = [r for r in results if r["was_wayback"]]
    live = [r for r in results if not r["was_wayback"]]
    if wb and live:
        print(f"\n  success on Wayback-origin URLs: "
              f"{100*sum(1 for r in wb if r['outcome']=='ok')/len(wb):.1f}% (n={len(wb)})")
        print(f"  success on plain URLs:          "
              f"{100*sum(1 for r in live if r['outcome']=='ok')/len(live):.1f}% (n={len(live)})")

    failed_domains = Counter(r["domain"] for r in results if r["outcome"] != "ok")
    if failed_domains:
        print(f"\n  domains failing most: {failed_domains.most_common(6)}")

    out = os.path.join(_HERE, f"fetchability_{args.split}.json")
    with open(out, "w") as f:
        json.dump({
            "split": args.split, "urls_sampled": len(results),
            "success_rate": len(ok) / len(results) if results else 0,
            "outcomes": dict(outcomes),
            "extracted_chars_median": statistics.median([r["chars"] for r in ok]) if ok else 0,
            "gold_chars_median": statistics.median([r["gold_chars"] for r in ok]) if ok else 0,
            "extraction_methods": dict(Counter(r["method"] for r in ok)),
            "wayback_recoveries": sum(1 for r in ok if r["url_used"] == "wayback"),
        }, f, indent=2)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--data-dir", default=default_data_dir())
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=12.0)
    sys.exit(asyncio.run(main_async(p.parse_args())))
