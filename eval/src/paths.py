from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalPaths:
    repo_root: Path

    @property
    def dataset_dir(self) -> Path:
        return self.repo_root / "docs" / "dataset" / "NewsScope"

    @property
    def eval_dir(self) -> Path:
        return self.repo_root / "eval"

    @property
    def data_dir(self) -> Path:
        return self.eval_dir / "data"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def intermediate_dir(self) -> Path:
        return self.data_dir / "intermediate"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


def get_repo_root() -> Path:
    # This file lives at <repo>/eval/src/paths.py
    return Path(__file__).resolve().parents[2]


def get_paths() -> EvalPaths:
    return EvalPaths(repo_root=get_repo_root())

