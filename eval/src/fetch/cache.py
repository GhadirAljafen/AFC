from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def url_cache_key(url: str) -> str:
    return _sha(url)[:24]


@dataclass(frozen=True)
class CachePaths:
    base_dir: Path

    def html_path(self, retrieval_source: str, url: str) -> Path:
        return self.base_dir / retrieval_source / f"{url_cache_key(url)}.html"

    def text_path(self, retrieval_source: str, url: str) -> Path:
        return self.base_dir / retrieval_source / f"{url_cache_key(url)}.txt"

    def meta_path(self, retrieval_source: str, url: str) -> Path:
        return self.base_dir / retrieval_source / f"{url_cache_key(url)}.json"


def read_text_if_exists(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

