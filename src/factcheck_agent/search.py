"""
Search backend abstraction.

Why this layer exists
---------------------
Google's Custom Search JSON API is **closed to new customers and sunsets
2027-01-01**. Everything above this module therefore works in terms of
`SearchResult` and the `SearchClient` interface, never a specific provider, so
the backend can be swapped without touching retrieval logic.

It also provides the two things the rest of the system needs:

* `CachedSearchClient` — a disk cache keyed by (provider, query, k). The free
  CSE tier allows 100 queries/day, and a connected pipeline spends queries per
  *claim*, so caching is a requirement rather than an optimisation. It also
  makes evaluation runs repeatable.
* `StaticSearchClient` — an offline double that serves canned results, so the
  retrieval loop can be tested deterministically and without quota.

Note that domain filtering (e.g. excluding fact-checking sites) deliberately
does NOT live here: the cache stores raw provider output so that filtering
policy can change without invalidating cached queries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only in stripped envs
    httpx = None


logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    """Lowercase alphanumeric tokens longer than two characters."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


class SearchError(Exception):
    """Raised when a search backend fails."""


class SearchResult(BaseModel):
    """One result from a search backend, provider-agnostic."""

    url: str = Field(..., description="Result URL")
    title: str = Field("", description="Result title")
    snippet: str = Field("", description="Short preview text from the provider")
    rank: int = Field(..., ge=1, description="1-based position within its query")

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"[{self.rank}] {self.title or self.url}"


class SearchClient(ABC):
    """Provider-agnostic search interface."""

    #: Short identifier used in cache keys so providers don't collide.
    name: str = "search"

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Return up to `k` results for `query`, ranked from 1."""
        ...


class GoogleCSEClient(SearchClient):
    """Google Custom Search JSON API.

    DEPRECATED UPSTREAM: closed to new customers, sunsets 2027-01-01. Kept as the
    current implementation; replace with another `SearchClient` before then.
    """

    name = "google_cse"
    ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: Optional[str] = None, engine_id: Optional[str] = None,
                 timeout: float = 10.0) -> None:
        from factcheck_agent.config import get_config

        config = get_config()
        self.api_key = api_key or getattr(config, "GOOGLE_SEARCH_API_KEY", None)
        self.engine_id = engine_id or getattr(config, "GOOGLE_SEARCH_ENGINE_ID", None)
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.api_key and self.engine_id)

    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if not self.is_configured():
            raise SearchError(
                "Google Search is not configured. Set GOOGLE_SEARCH_API_KEY and "
                "GOOGLE_SEARCH_ENGINE_ID in your environment or .env file."
            )
        if httpx is None:
            raise SearchError(
                "httpx is required for Google Search. Install it with: pip install httpx"
            )

        params = {"key": self.api_key, "cx": self.engine_id, "q": query, "num": k}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(self.ENDPOINT, params=params)
            if resp.status_code != 200:
                raise SearchError(
                    f"Google Search API error: {resp.status_code} - {resp.text[:300]}"
                )
            items = resp.json().get("items", [])

        if not isinstance(items, list):
            return []

        results: List[SearchResult] = []
        for position, item in enumerate(items, start=1):
            url = item.get("link")
            if not url:
                continue
            results.append(SearchResult(
                url=url,
                title=item.get("title") or "",
                snippet=item.get("snippet") or "",
                rank=position,
            ))
        return results


class StaticSearchClient(SearchClient):
    """Offline double serving canned results.

    Used for deterministic, quota-free testing of the retrieval loop, and as the
    backing for corpus-driven evaluation (e.g. an AVeriTeC knowledge store).
    `queries_seen` records call order so tests can assert that later retrieval
    rounds actually issue *different* queries.
    """

    name = "static"

    def __init__(self, responses: Optional[Dict[str, Sequence[SearchResult]]] = None,
                 default: Optional[Sequence[SearchResult]] = None) -> None:
        self.responses = dict(responses or {})
        self.default = list(default or [])
        self.queries_seen: List[str] = []

    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        self.queries_seen.append(query)
        results = self.responses.get(query, self.default)
        return list(results)[:k]


class CorpusSearchClient(SearchClient):
    """Offline search over a fixed document set, ranked by IDF-weighted overlap.

    Unlike `StaticSearchClient` (exact-query lookup), this actually ranks against
    the query text, so different queries surface different documents. That is what
    makes it usable for evaluating the retrieval loop: a gap-driven second query
    can retrieve a document the first query missed.

    It measures **closed-corpus retrieval recall**, not open-web evidence
    coverage — the corpus is guaranteed to contain the answer, so results are
    easier than live search and must be reported as a distinct condition.
    """

    name = "corpus"

    def __init__(self, documents: Sequence[Dict[str, str]], snippet_chars: int = 200) -> None:
        """
        Args:
            documents: dicts with `url`, optional `title`, and `text`.
            snippet_chars: how much of the body to expose, mimicking a search
                snippet rather than handing over the whole document.
        """
        import math

        self.documents = list(documents)
        self.snippet_chars = snippet_chars
        self.queries_seen: List[str] = []

        self._tokens: List[set] = []
        doc_freq: Dict[str, int] = {}
        for doc in self.documents:
            toks = _tokenize(f"{doc.get('title', '')} {doc.get('text', '')}")
            self._tokens.append(toks)
            for tok in toks:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        n = max(1, len(self.documents))
        self._idf = {t: math.log(n / (1 + df)) + 1.0 for t, df in doc_freq.items()}

    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        self.queries_seen.append(query)
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scored = []
        for i, doc_tokens in enumerate(self._tokens):
            shared = q_tokens & doc_tokens
            if not shared:
                continue
            scored.append((sum(self._idf.get(t, 1.0) for t in shared), i))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        results = []
        for rank, (_score, i) in enumerate(scored[:k], start=1):
            doc = self.documents[i]
            body = doc.get("text", "")
            results.append(SearchResult(
                url=doc["url"],
                title=doc.get("title", ""),
                snippet=body[: self.snippet_chars],
                rank=rank,
            ))
        return results


class CachedSearchClient(SearchClient):
    """Disk-cache wrapper around any `SearchClient`.

    Keyed by (provider, k, query) so different providers and result widths never
    collide. Cache read/write failures are logged and ignored — a broken cache
    must degrade to a live call, never break retrieval.
    """

    def __init__(self, inner: SearchClient, cache_dir: str = ".cache/search") -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        self.name = f"cached:{inner.name}"
        self.hits = 0
        self.misses = 0

    def _key(self, query: str, k: int) -> str:
        raw = f"{self.inner.name}:{k}:{query}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _path(self, query: str, k: int) -> str:
        return os.path.join(self.cache_dir, f"{self._key(query, k)}.json")

    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        path = self._path(query, k)
        try:
            with open(path) as f:
                payload = json.load(f)
            self.hits += 1
            return [SearchResult(**r) for r in payload["results"]]
        except FileNotFoundError:
            pass
        except Exception as e:  # corrupt entry, wrong shape, unreadable ...
            logger.warning("Ignoring unusable search cache entry %s: %s", path, e)

        self.misses += 1
        results = await self.inner.search(query, k=k)

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {"provider": self.inner.name, "query": query, "k": k,
                     "results": [r.model_dump() for r in results]},
                    f,
                )
        except Exception as e:
            logger.warning("Could not write search cache entry %s: %s", path, e)

        return results


def get_default_search_client(cache_dir: Optional[str] = ".cache/search") -> SearchClient:
    """Build the configured search client, wrapped in a cache unless disabled."""
    client: SearchClient = GoogleCSEClient()
    if cache_dir:
        client = CachedSearchClient(client, cache_dir=cache_dir)
    return client
