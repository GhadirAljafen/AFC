"""
Configuration management for the fact-check agent.
Handles API keys, logging, and runtime settings.
"""

import os
from typing import Optional

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional


class Config:
    """Configuration object for the fact-checking agent."""

    # ---- LLM Providers ----
    # You may choose one or have fallback logic in LLMClient
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # ---- Search / Retrieval API (optional) ----
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    SERPAPI_KEY: Optional[str] = os.getenv("SERPAPI_KEY")
    CUSTOM_SEARCH_API_KEY: Optional[str] = os.getenv("SEARCH_API_KEY")  # legacy name
    GOOGLE_SEARCH_API_KEY: Optional[str] = os.getenv("GOOGLE_SEARCH_API_KEY")
    GOOGLE_SEARCH_ENGINE_ID: Optional[str] = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    # ---- General Settings ----
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_EVIDENCE_SOURCES: int = int(os.getenv("MAX_EVIDENCE_SOURCES", "5"))
    # Default 2, not 3: a genuinely iterating retrieval loop multiplies search
    # spend by the number of rounds, and the CSE free tier allows 100/day.
    MAX_RETRIEVAL_LOOPS: int = int(os.getenv("MAX_RETRIEVAL_LOOPS", "2"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

    # ---- Retrieval / quota controls ----
    # Search spend is the binding constraint on the connected pipeline: cost is
    # (claims per article) x (queries per claim) x (retrieval rounds).
    SEARCH_CACHE_DIR: str = os.getenv("SEARCH_CACHE_DIR", ".cache/search")
    MAX_QUERIES_PER_CLAIM: int = int(os.getenv("MAX_QUERIES_PER_CLAIM", "3"))
    MAX_GAP_QUERIES: int = int(os.getenv("MAX_GAP_QUERIES", "2"))
    MAX_CLAIMS_PER_ARTICLE: int = int(os.getenv("MAX_CLAIMS_PER_ARTICLE", "10"))
    # Hard ceiling per article; retrieval stops and says so rather than silently
    # truncating once this many searches have been issued.
    QUERY_BUDGET_PER_ARTICLE: int = int(os.getenv("QUERY_BUDGET_PER_ARTICLE", "60"))
    # How many top-ranked results get their full page fetched (vs snippet only).
    FULL_TEXT_TOP_K: int = int(os.getenv("FULL_TEXT_TOP_K", "3"))
    MAX_SOURCE_CHARS: int = int(os.getenv("MAX_SOURCE_CHARS", "4000"))

    # ---- Evidence integrity ----
    # Comma-separated domains dropped from results. Fact-check leakage is MEASURED
    # regardless of this setting; this only controls whether it is also filtered.
    EXCLUDE_DOMAINS: str = os.getenv("EXCLUDE_DOMAINS", "")

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration.
        
        Raises:
            RuntimeError: if no LLM key is provided.
        """
        if not cls.ANTHROPIC_API_KEY and not cls.OPENAI_API_KEY:
            raise RuntimeError(
                "No LLM API key provided! Set at least one: ANTHROPIC_API_KEY or OPENAI_API_KEY."
            )


def get_config() -> Config:
    """Return configuration singleton."""
    return Config()
