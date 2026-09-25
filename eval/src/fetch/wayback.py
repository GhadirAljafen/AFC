from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WaybackSnapshot:
    snapshot_url: str
    timestamp: str


async def lookup_closest_snapshot(client: "httpx.AsyncClient", url: str) -> Optional[WaybackSnapshot]:
    """
    Query Wayback Machine for closest available snapshot of the given URL.

    API: https://archive.org/wayback/available?url=<url>
    """
    import httpx

    api_url = "https://archive.org/wayback/available"
    try:
        r = await client.get(api_url, params={"url": url}, follow_redirects=True)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    snapshots = data.get("archived_snapshots") if isinstance(data, dict) else None
    closest = snapshots.get("closest") if isinstance(snapshots, dict) else None
    if not isinstance(closest, dict):
        return None
    if not closest.get("available"):
        return None
    snapshot_url = closest.get("url")
    timestamp = closest.get("timestamp")
    if not snapshot_url or not timestamp:
        return None
    return WaybackSnapshot(snapshot_url=str(snapshot_url), timestamp=str(timestamp))

