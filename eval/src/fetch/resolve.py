from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResolvedUrl:
    url: str
    method: str  # e.g. ddg_html


async def resolve_url_from_headline(client: "httpx.AsyncClient", headline: str) -> Optional[ResolvedUrl]:
    """
    Resolve a likely article URL using a non-LLM web search by headline.

    Tries several DuckDuckGo surfaces (HTML + lite) because markup and bot tolerance vary.
    When all fail, returns None (strict policy: skip article).
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return None

    q = headline.strip()
    if not q:
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml",
    }

    # Strategy 1: DDG HTML POST (often more stable than GET /html/)
    for method, url, kwargs in (
        (
            "post",
            "https://html.duckduckgo.com/html/",
            {"data": {"q": q, "b": ""}, "headers": headers, "follow_redirects": True, "timeout": 30.0},
        ),
        (
            "get",
            "https://duckduckgo.com/html/",
            {"params": {"q": q}, "headers": headers, "follow_redirects": True, "timeout": 30.0},
        ),
        (
            "get",
            "https://lite.duckduckgo.com/lite/",
            {"params": {"q": q}, "headers": headers, "follow_redirects": True, "timeout": 30.0},
        ),
    ):
        try:
            if method == "post":
                r = await client.post(url, **kwargs)
            else:
                r = await client.get(url, **kwargs)
            r.raise_for_status()
        except Exception:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        href = _first_result_href(soup)
        if href:
            resolved = _normalize_ddg_href(href)
            if resolved.startswith("http"):
                return ResolvedUrl(url=resolved, method=f"ddg:{method}:{url.split('/')[2]}")

    return None


def _first_result_href(soup: "BeautifulSoup") -> Optional[str]:
    """Extract first organic result link from DDG HTML or lite HTML."""
    selectors = [
        "a.result__a",
        ".result__title a",
        ".result a.result__a",
        "a[data-testid='result-title-a']",
        "a.link-result",
        "a.result-link",
    ]
    for sel in selectors:
        a = soup.select_one(sel)
        if a and a.get("href"):
            href = str(a.get("href")).strip()
            if href and not href.startswith("#"):
                return href
    # Fallback: first external-looking href in main results
    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        if "duckduckgo.com" in href and "uddg=" in href:
            return href
        if href.startswith("http") and "duckduckgo.com" not in href:
            return href
    return None


def _normalize_ddg_href(href: str) -> str:
    """
    DDG sometimes returns redirector links like:
      //duckduckgo.com/l/?uddg=<encoded_target>&rut=...

    This function extracts the true target URL when possible.
    """
    from urllib.parse import parse_qs, unquote, urlparse

    h = href.strip()
    if h.startswith("//"):
        h = "https:" + h

    try:
        u = urlparse(h)
    except Exception:
        return h

    if "duckduckgo.com" in (u.netloc or "") and u.path.startswith("/l/"):
        qs = parse_qs(u.query)
        uddg = qs.get("uddg", [None])[0]
        if isinstance(uddg, str) and uddg:
            return unquote(uddg)

    return h

