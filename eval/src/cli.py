from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from eval.src.paths import get_paths


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="newsscope-eval",
        description="NewsScope Claim Understanding evaluation pipeline",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ["fetch", "run-agent", "score", "all"]:
        sp = sub.add_parser(name)
        sp.add_argument(
            "--split",
            required=True,
            choices=["test_indomain", "test_oos", "train"],
            help="Dataset split to run on",
        )
        sp.add_argument("--limit", type=int, default=None, help="Optional max articles")

    return p


async def _run_async(args: argparse.Namespace) -> int:
    # Lazy imports to keep startup fast and avoid importing optional deps unless needed.
    from eval.src.ingest import ingest_split
    from eval.src.fetch.pipeline import fetch_split

    paths = get_paths()
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    paths.intermediate_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = ingest_split(split=args.split, limit=args.limit)

    if args.cmd in ("fetch", "all"):
        await fetch_split(normalized_jsonl=normalized_path, split=args.split, limit=args.limit)

    if args.cmd in ("run-agent", "all"):
        from eval.src.agent_runner import run_agent_on_split

        await run_agent_on_split(split=args.split, limit=args.limit)

    if args.cmd in ("score", "all"):
        from eval.src.score import score_split

        score_split(split=args.split, limit=args.limit)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

