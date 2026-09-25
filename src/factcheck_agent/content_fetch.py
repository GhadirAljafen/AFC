"""
Full-text enrichment for retrieved evidence.

Search backends return ~180-character snippets, which is too thin to reason a
verdict from. This module fetches the actual page for the top-ranked results and
replaces the snippet with claim-relevant passages from the article body.

Design constraints, each learned from an observed failure:

* **Top-k only.** Fetching every result is slow and mostly fails — news domains
  commonly return 403/401 (nytimes, politico, ft, thehill, reuters were all
  observed doing so). Rank first, fetch a few.
* **Never lose a source.** Any failure keeps the original snippet, so enrichment
  can only improve on the previous behaviour.
* **Magic-byte check, not just Content-Type.** A PDF served with an HTML-ish
  content type previously ended up cached as a 119 KB "text" file of raw
  `%PDF-1.7` bytes and fed to the LLM.
* **Character cap + passage selection.** Whole articles are expensive and dilute
  the evidence; only the passages matching the claim are kept.

The trafilatura-then-BeautifulSoup extraction strategy follows the one proven in
`eval/src/fetch/extract.py`. It is reimplemented here rather than imported so
production code does not depend on the evaluation harness.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from typing import List, Optional, Sequence, Set
from urllib.parse import urlparse

from factcheck_agent.models import EvidenceSnippet

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BINARY_SIGNATURES = (b"%PDF", b"\x89PNG", b"GIF8", b"\xff\xd8\xff", b"PK\x03\x04")


def _tokens(text: str) -> Set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


def _looks_binary(raw: bytes) -> bool:
    """True if the payload is not HTML/text, judged by magic bytes."""
    head = raw[:8]
    return any(head.startswith(sig) for sig in _BINARY_SIGNATURES)


def extract_main_text(html: str, url: Optional[str] = None) -> Optional[str]:
    """Extract article body text, preferring trafilatura, falling back to bs4."""
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=False,
            favor_precision=True,
        )
        if extracted:
            cleaned = _clean(extracted)
            if cleaned:
                return cleaned
    except Exception as e:
        logger.debug("trafilatura extraction failed for %s: %s", url, e)

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
            tag.decompose()
        cleaned = _clean(soup.get_text(separator="\n"))
        return cleaned or None
    except Exception as e:
        logger.debug("bs4 extraction failed for %s: %s", url, e)
        return None


def _clean(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def select_passages(text: str, claim_text: str, max_chars: int) -> str:
    """Keep the paragraphs most relevant to the claim, up to `max_chars`.

    Feeding a whole article to the LLM is costly and dilutes the signal, so
    paragraphs are ranked by lexical overlap with the claim and re-assembled in
    their original document order.
    """
    paragraphs = [p for p in text.split("\n") if len(p) > 40]
    if not paragraphs:
        return text[:max_chars]

    claim_tokens = _tokens(claim_text)
    if not claim_tokens:
        return text[:max_chars]

    scored = [
        (len(claim_tokens & _tokens(p)) / len(claim_tokens), i, p)
        for i, p in enumerate(paragraphs)
    ]
    scored.sort(key=lambda t: t[0], reverse=True)

    chosen: List[tuple] = []
    budget = max_chars
    for score, index, para in scored:
        if score <= 0 and chosen:
            break                      # nothing left that mentions the claim
        if len(para) > budget:
            continue
        chosen.append((index, para))
        budget -= len(para) + 1
        if budget <= 0:
            break

    if not chosen:
        return text[:max_chars]

    chosen.sort(key=lambda t: t[0])    # restore document order for readability
    return "\n".join(p for _, p in chosen)[:max_chars]


class ContentFetcher:
    """Fetches and extracts article text for the top-ranked evidence."""

    #: Wayback's "latest snapshot" redirect. `2` means "closest to now".
    WAYBACK_TEMPLATE = "https://web.archive.org/web/2/{url}"

    def __init__(
        self,
        cache_dir: str = ".cache/pages",
        timeout: float = 15.0,
        max_bytes: int = 4_000_000,
        max_chars: int = 4000,
        user_agent: str = "AFC-FactCheckAgent/0.1 (+research)",
        concurrency: int = 4,
        wayback_fallback: bool = True,
    ) -> None:
        """
        Args:
            wayback_fallback: On a failed live fetch, retry via the Wayback
                Machine. Measured on 150 AVeriTeC gold URLs: live URLs succeed
                64.5% of the time, Wayback-origin URLs 80.7%, and the fallback
                recovered 16 of 106 total successes. Archives neither bot-block
                nor go dead, so this is usually the *more* reliable source.
        """
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.user_agent = user_agent
        self.concurrency = concurrency
        self.wayback_fallback = wayback_fallback
        self.fetched = 0
        self.failed = 0
        self.wayback_recoveries = 0

    # ---------- public ----------

    async def enrich(
        self,
        snippets: Sequence[EvidenceSnippet],
        claim_text: str,
        top_k: int = 3,
    ) -> List[EvidenceSnippet]:
        """Return snippets with the top-k entries' text replaced by article passages.

        Order is preserved. Entries that fail keep their original snippet.
        """
        snippets = list(snippets)
        if not snippets or top_k <= 0:
            return snippets

        targets = snippets[:top_k]
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(snippet: EvidenceSnippet) -> EvidenceSnippet:
            async with semaphore:
                url = (snippet.metadata or {}).get("url") or snippet.id
                text = await self._article_text(url)
            if not text:
                self.failed += 1
                return snippet
            self.fetched += 1
            passages = select_passages(text, claim_text, self.max_chars)
            if not passages or len(passages) <= len(snippet.text):
                return snippet          # extraction added nothing useful
            return snippet.model_copy(update={
                "text": passages,
                "metadata": {**(snippet.metadata or {}), "full_text": "true",
                             "full_text_chars": str(len(passages))},
            })

        enriched = await asyncio.gather(*(one(s) for s in targets), return_exceptions=True)

        out: List[EvidenceSnippet] = []
        for original, result in zip(targets, enriched):
            if isinstance(result, BaseException):
                logger.warning("Enrichment failed for %s: %s", original.id, result)
                self.failed += 1
                out.append(original)
            else:
                out.append(result)
        return out + snippets[top_k:]

    # ---------- internal ----------

    def _cache_path(self, url: str) -> str:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{key}.txt")

    async def _article_text(self, url: str) -> Optional[str]:
        path = self._cache_path(url)
        try:
            with open(path) as f:
                return f.read()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Unusable page cache entry %s: %s", path, e)

        html = await self._fetch(url)

        # ~30% of real evidence URLs are unreachable live (403 bot-blocks, dead
        # links). The archived copy is often still there, and measurably more
        # reliable than the original.
        if html is None and self.wayback_fallback and "web.archive.org" not in url:
            archived = await self._fetch(self.WAYBACK_TEMPLATE.format(url=url))
            if archived is not None:
                self.wayback_recoveries += 1
                logger.info("Recovered %s via the Wayback Machine", urlparse(url).netloc)
                html = archived

        if html is None:
            return None

        text = extract_main_text(html, url=url)
        if not text:
            return None

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        except Exception as e:
            logger.warning("Could not write page cache %s: %s", path, e)
        return text

    async def _fetch(self, url: str) -> Optional[str]:
        if httpx is None:
            logger.warning("httpx unavailable; cannot fetch %s", url)
            return None

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                raw = resp.content[: self.max_bytes]
        except Exception as e:
            # Paywalls and bot-blocks are routine here; the snippet fallback covers it.
            logger.info("Could not fetch %s (%s); keeping snippet", urlparse(url).netloc, e)
            return None

        if _looks_binary(raw):
            logger.info("Skipping non-HTML payload at %s (magic bytes)", url)
            return None

        content_type = resp.headers.get("content-type", "").lower()
        if content_type and "html" not in content_type and "text" not in content_type:
            logger.info("Skipping %s (content-type %s)", url, content_type)
            return None

        return raw.decode(resp.encoding or "utf-8", errors="replace")
