"""
Export the search-query trail into a committed, auditable artifact.

Why this exists. Live search results are cached in `.cache/search/`, which is
gitignored — reasonably, since it is regenerable scratch. But for the live
open-web run (§11 of RESULTS.md) it is not regenerable: reproducing it means
spending Google CSE quota again (55 queries of a 100/day tier), and the live web
changes, so the same queries would not return the same results tomorrow.

Without an export, §11's findings — particularly the claim that 10% of round-1
results are NLP research papers — rest on numbers nobody can re-derive. This
writes every query and its returned results into a file that travels with the
repo, so the analysis is auditable without re-spending quota.

Also records, per query, whether each result came from a fact-checking site or a
research venue, since those two classifications are what §11.3 asserts.

Usage:
    python results/evidence_retrieval/export_query_trail.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from factcheck_agent.agents.retrieval import _domain_of, _is_fact_check_source  # noqa: E402

# Research venues and shared-task sites. Retrieving these means the system found
# papers *about* fact-checking rather than evidence about the claim — a distinct
# failure from citing a fact-checker's verdict. See RESULTS.md section 11.3.
RESEARCH_DOMAINS = frozenset({
    "arxiv.org", "aclanthology.org", "openreview.net", "semanticscholar.org",
    "researchgate.net", "fever.ai", "uio.no", "paperswithcode.com",
    "huggingface.co", "dl.acm.org", "ieeexplore.ieee.org",
})


def classify(url: str) -> str:
    domain = _domain_of(url)
    if _is_fact_check_source(url):
        return "fact_check"
    if domain in RESEARCH_DOMAINS:
        return "research_venue"
    return "other"


def main() -> int:
    cache_dir = os.path.join(_ROOT, ".cache", "search")
    files = sorted(glob.glob(os.path.join(cache_dir, "*.json")))
    if not files:
        print(f"No cached queries under {cache_dir}", file=sys.stderr)
        return 1

    queries, counts = [], Counter()
    for path in files:
        try:
            entry = json.load(open(path))
        except Exception as e:
            print(f"  skipping unreadable {os.path.basename(path)}: {e}")
            continue

        results = []
        for r in entry.get("results", []):
            kind = classify(r["url"])
            counts[kind] += 1
            results.append({
                "rank": r.get("rank"), "url": r["url"],
                "domain": _domain_of(r["url"]),
                "title": r.get("title", ""), "snippet": r.get("snippet", ""),
                "classification": kind,
            })
        queries.append({
            "query": entry.get("query", ""),
            "provider": entry.get("provider", ""),
            "k": entry.get("k"),
            "n_results": len(results),
            "results": results,
        })

    queries.sort(key=lambda q: q["query"])
    total = sum(counts.values())

    out = os.path.join(_HERE, "query_trail.json")
    with open(out, "w") as f:
        json.dump({
            "exported": str(date.today()),
            "purpose": ("Auditable record of every live search query and its results. "
                        "Exported because .cache/search/ is gitignored and the live run "
                        "cannot be reproduced without re-spending Google CSE quota, on a "
                        "web whose results change over time."),
            "provider_note": ("Google Custom Search JSON API — closed to new customers, "
                              "sunsets 2027-01-01."),
            "n_queries": len(queries),
            "n_results": total,
            "result_classification": dict(counts),
            "classification_note": ("'research_venue' means the result is a paper or "
                                    "shared-task site about fact-checking rather than "
                                    "evidence about the claim — see RESULTS.md 11.3."),
            "queries": queries,
        }, f, indent=2)

    print(f"exported {len(queries)} queries, {total} results -> {out}")
    print(f"  size: {os.path.getsize(out) / 1024:.0f} KB")
    print()
    print("result classification across all cached queries:")
    for kind, n in counts.most_common():
        print(f"  {kind:>15} {n:>4}  ({100 * n / total:.1f}%)")

    domains = Counter(r["domain"] for q in queries for r in q["results"])
    # Count fact-check results PER DOMAIN rather than flagging the whole domain.
    # Some publishers (e.g. reuters.com) host a fact-check desk under a path, so
    # a domain-level flag would mislabel their ordinary reporting.
    fc_by_domain = Counter(r["domain"] for q in queries for r in q["results"]
                           if r["classification"] == "fact_check")
    print("\nmost frequent domains:")
    for domain, n in domains.most_common(10):
        if domain in RESEARCH_DOMAINS:
            tag = "  <- research venue (paper about fact-checking, not evidence)"
        elif fc_by_domain[domain]:
            tag = f"  <- {fc_by_domain[domain]}/{n} results are fact-check pages"
        else:
            tag = ""
        print(f"  {n:>3}  {domain}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
