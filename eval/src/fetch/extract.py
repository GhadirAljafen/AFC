from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ExtractResult:
    text: str
    method: str


def extract_main_text(html: str, url: str | None = None) -> Optional[ExtractResult]:
    """
    Extract main article text from HTML.

    Uses `trafilatura` when available; returns None if extraction fails.
    """
    try:
        import trafilatura  # type: ignore
    except Exception:
        trafilatura = None  # type: ignore

    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if extracted:
                text = _clean_text(extracted)
                if text:
                    return ExtractResult(text=text, method="trafilatura")
        except Exception:
            pass

    # Lightweight fallback (very naive): strip tags if BeautifulSoup is installed
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        # remove scripts/styles
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        text = soup.get_text(separator="\n")
        text = _clean_text(text)
        if text:
            return ExtractResult(text=text, method="bs4_get_text")
    except Exception:
        return None

    return None


def _clean_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    cleaned = "\n".join(lines)
    return cleaned.strip()

