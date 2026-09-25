"""
RetrievalAgent — turns a Claim into ranked EvidenceSnippets.

This agent:
- Generates search queries from a claim (LLM-based when available)
- Issues them through a provider-agnostic `SearchClient`
- Fuses results across queries with reciprocal-rank fusion
- Returns scored `EvidenceSnippet` objects

Two behaviours matter for the agentic loop:

* `gaps` — on later retrieval rounds the evidence evaluator says what is still
  missing, and query generation targets that instead of re-issuing the original
  queries. Without this the loop re-fetches identical URLs and cannot improve.
* `exclude_urls` — URLs already collected, so a later round spends its quota on
  genuinely new sources.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Sequence, Set
from urllib.parse import urlparse

from factcheck_agent.config import get_config
from factcheck_agent.llm_client import LLMClient
from factcheck_agent.models import Claim, EvidenceSnippet
from factcheck_agent.search import SearchClient, SearchError, SearchResult, get_default_search_client


logger = logging.getLogger(__name__)
config = get_config()

# Reciprocal-rank-fusion damping constant. 60 is the value from the original RRF
# paper and is standard in IR; it keeps top ranks from dominating completely.
RRF_K = 60

# Domains whose presence makes a verdict potentially circular: the system would be
# reading a fact-checker's conclusion rather than verifying the claim itself. This
# list is used for MEASUREMENT (leakage rate) regardless of whether filtering is on.
FACT_CHECK_DOMAINS = frozenset({
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "apnews.com/hub/ap-fact-check", "reuters.com/fact-check", "checkyourfact.com",
    "truthorfiction.com", "leadstories.com", "factcheck.afp.com", "africacheck.org",
    "washingtonpost.com/news/fact-checker", "usatoday.com/story/news/factcheck",
    "poynter.org", "boomlive.in", "altnews.in",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class RetrievalError(Exception):
    """Custom exception for retrieval-related errors."""


def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return netloc[4:] if netloc.startswith("www.") else netloc


def _is_fact_check_source(url: str) -> bool:
    """True when the URL looks like a fact-checking article.

    Matches on domain, and also on a few publishers whose fact-check desks live
    under a path prefix (so the domain alone would be too broad).
    """
    domain = _domain_of(url)
    lowered = url.lower()
    for entry in FACT_CHECK_DOMAINS:
        if "/" in entry:
            if entry in lowered.replace("www.", ""):
                return True
        elif domain == entry:
            return True
    return False


def _tokens(text: str) -> Set[str]:
    """Lowercase alphanumeric tokens, dropping very short ones."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


class RetrievalAgent:
    """Retrieval agent backed by a pluggable search client."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        search_client: Optional[SearchClient] = None,
        max_results_per_query: int = 5,
        max_queries: int = 3,
        exclude_domains: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Args:
            llm: Optional LLM client for query generation. Without it, a small
                 rule-based query set is used.
            search_client: Search backend. Defaults to the configured provider
                 wrapped in a disk cache.
            max_results_per_query: Results requested per query.
            max_queries: Maximum queries issued in the first retrieval round.
            exclude_domains: Domains to drop from results entirely. Leakage is
                 measured regardless; this controls whether it is also filtered.
        """
        self.llm = llm
        self.search = search_client or get_default_search_client()
        self.max_results_per_query = max_results_per_query
        self.max_queries = max_queries

        configured = exclude_domains
        if configured is None:
            raw = getattr(config, "EXCLUDE_DOMAINS", "") or ""
            configured = [d.strip().lower() for d in raw.split(",") if d.strip()]
        self.exclude_domains = {d[4:] if d.startswith("www.") else d for d in configured}

        # Counters for reporting; reset per agent instance, not per call.
        self.searches_issued = 0
        self.results_seen = 0
        self.fact_check_results_seen = 0
        #: Queries issued, grouped by retrieve_evidence() call. One entry per
        #: retrieval round, which makes round-by-round behaviour inspectable.
        self.query_rounds: List[List[str]] = []

    # ---------- Public API ----------

    @property
    def fact_check_leakage_rate(self) -> float:
        """Share of retrieved results that came from fact-checking sites."""
        return self.fact_check_results_seen / self.results_seen if self.results_seen else 0.0

    async def retrieve_evidence(
        self,
        claim: Claim,
        gaps: Optional[str] = None,
        exclude_urls: Optional[Set[str]] = None,
        max_queries: Optional[int] = None,
    ) -> List[EvidenceSnippet]:
        """
        Retrieve and rank evidence for a claim.

        Args:
            claim: The claim to find evidence for.
            gaps: What the evidence evaluator says is still missing. When given,
                  queries target the gap rather than repeating the first round.
            exclude_urls: URLs already collected in earlier rounds.
            max_queries: Override the query budget for this round. Gap-driven
                  rounds should pass a small number (1-2) to conserve quota.

        Returns:
            EvidenceSnippets with populated relevance scores, best first.
        """
        claim_text = claim.normalized_text or claim.raw_text
        exclude_urls = exclude_urls or set()

        queries = await self._generate_queries(claim_text, gaps=gaps)
        budget = max_queries if max_queries is not None else self.max_queries
        queries = queries[:budget]
        self.query_rounds.append(list(queries))

        # url -> accumulated RRF contribution, plus the best result seen for it
        fused: dict[str, float] = {}
        best: dict[str, SearchResult] = {}

        for query in queries:
            try:
                results = await self.search.search(query, k=self.max_results_per_query)
            except SearchError as e:
                logger.error("Search failed for %r: %s", query, e)
                continue
            self.searches_issued += 1

            for result in results:
                self.results_seen += 1
                if _is_fact_check_source(result.url):
                    self.fact_check_results_seen += 1

                if result.url in exclude_urls:
                    continue
                if _domain_of(result.url) in self.exclude_domains:
                    continue

                fused[result.url] = fused.get(result.url, 0.0) + 1.0 / (RRF_K + result.rank)
                prior = best.get(result.url)
                if prior is None or result.rank < prior.rank:
                    best[result.url] = result

        if not fused:
            return []

        claim_tokens = _tokens(claim_text)
        # Normalise RRF so a URL ranked #1 by every query scores ~1.0.
        max_rrf = (len(queries) or 1) / (RRF_K + 1)

        snippets: List[EvidenceSnippet] = []
        for url, rrf in fused.items():
            result = best[url]
            rrf_norm = min(1.0, rrf / max_rrf) if max_rrf else 0.0
            overlap = self._lexical_overlap(claim_tokens, f"{result.title} {result.snippet}")
            score = 0.6 * rrf_norm + 0.4 * overlap

            snippets.append(EvidenceSnippet(
                id=url,
                source="search",
                text=result.snippet,
                score=round(score, 4),
                metadata={
                    "title": result.title,
                    "url": url,
                    "domain": _domain_of(url),
                    "rank": str(result.rank),
                    "fact_check_source": str(_is_fact_check_source(url)),
                },
            ))

        snippets.sort(key=lambda s: s.score or 0.0, reverse=True)

        if self.fact_check_results_seen:
            logger.info(
                "Fact-check leakage: %d/%d results (%.1f%%) came from fact-checking sites",
                self.fact_check_results_seen, self.results_seen,
                100 * self.fact_check_leakage_rate,
            )
        return snippets

    # ---------- Internal helpers ----------

    @staticmethod
    def _lexical_overlap(claim_tokens: Set[str], text: str) -> float:
        """Fraction of the claim's tokens that appear in the candidate text."""
        if not claim_tokens:
            return 0.0
        return len(claim_tokens & _tokens(text)) / len(claim_tokens)

    async def _generate_queries(self, claim_text: str, gaps: Optional[str] = None) -> List[str]:
        """
        Generate search queries for a claim.

        On a gap-driven round the queries must differ from the first round, or the
        retrieval loop cannot surface anything new.
        """
        if self.llm is None:
            if gaps:
                return [f"{claim_text} {gaps}"[:300]]
            return [claim_text, f"{claim_text} false", f"{claim_text} debunked"]

        if gaps:
            system_prompt = (
                "You are a search query generator for a fact-checking system. A first "
                "round of searching has already happened and was judged insufficient.\n\n"
                "Write NEW search queries that target ONLY the missing information "
                "described below. Do not repeat the obvious queries about the claim "
                "itself — those have already been tried.\n\n"
                "Rules:\n"
                "- Focus narrowly on the stated gap\n"
                "- Use specific entities, dates, numbers and locations\n"
                "- Output one query per line. No extra text."
            )
            user_content = (
                f"Claim:\n{claim_text}\n\nMissing information:\n{gaps}\n\nNew search queries:"
            )
        else:
            system_prompt = (
                "You are a search query generator for a fact-checking system. "
                "Given a factual claim, create 3-5 search queries that would help find evidence.\n\n"
                "IMPORTANT: Include queries that look for:\n"
                "1. Direct information about the claim\n"
                "2. REFUTATIONS of the claim (e.g. 'claim X false', 'claim X debunked')\n"
                "3. Key entities, dates, numbers, and locations from the claim\n\n"
                "Rules:\n"
                "- Use key entities, dates, numbers, and locations\n"
                "- Include at least one query explicitly looking for refutations\n"
                "- Avoid words like 'prove' in the main query\n"
                "- Output one query per line. No extra text."
            )
            user_content = f"Claim:\n{claim_text}\n\nSearch queries (include refutation queries):"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            raw_queries = await self.llm.chat(messages)
        except Exception as e:
            logger.error("Query generation failed (%s); falling back to claim text", e)
            return [claim_text]

        queries = [q.strip() for q in raw_queries.splitlines() if q.strip()]

        # Only the first round is forced to include a refutation query; gap rounds
        # must stay focused on what is actually missing.
        if not gaps and queries:
            has_refutation = any(
                kw in q.lower()
                for q in queries
                for kw in ("false", "debunk", "incorrect", "misleading")
            )
            if not has_refutation:
                queries.append(f"{claim_text} false")

        return queries or [claim_text]
