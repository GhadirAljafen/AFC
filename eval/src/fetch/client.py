from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FetchConfig:
    concurrency: int = 12
    timeout_s: float = 25.0
    user_agent: str = "AFC-NewsScopeEval/0.1 (+academic evaluation)"
    max_html_bytes: int = 8_000_000
    per_host_min_delay_s: float = 0.4


class Throttler:
    def __init__(self, per_host_min_delay_s: float) -> None:
        self.per_host_min_delay_s = per_host_min_delay_s
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_time: Dict[str, float] = {}

    async def wait(self, host: str) -> None:
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            last = self._last_time.get(host, 0.0)
            delay = self.per_host_min_delay_s - (now - last)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_time[host] = time.monotonic()


async def fetch_html(client: "httpx.AsyncClient", url: str, cfg: FetchConfig, throttler: Throttler) -> str:
    import httpx
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.netloc or "unknown"
    await throttler.wait(host)

    headers = {"User-Agent": cfg.user_agent, "Accept": "text/html,application/xhtml+xml"}
    r = await client.get(url, headers=headers, timeout=cfg.timeout_s, follow_redirects=True)
    r.raise_for_status()
    content = r.content
    if len(content) > cfg.max_html_bytes:
        content = content[: cfg.max_html_bytes]
    # best-effort decode
    try:
        return content.decode(r.encoding or "utf-8", errors="replace")
    except Exception:
        return content.decode("utf-8", errors="replace")

