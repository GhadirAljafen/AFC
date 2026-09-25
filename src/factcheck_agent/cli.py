"""Command-line interface for the fact-check agent.

Two modes:
  * single claim   — `factcheck "some claim"`
  * whole document — `factcheck --file article.txt`

Document mode runs the connected pipeline: detect check-worthy claims, then verify
each one. It reports per-claim verdicts and does NOT roll them into a single
document verdict — an article containing one true and one false claim is not a
conflict to resolve.
"""

import argparse
import asyncio
import sys
import uuid
from typing import Optional

from factcheck_agent.models import ArticleFactCheckResult, Claim, FactCheckResult
from factcheck_agent.pipeline import FactCheckingPipeline
from factcheck_agent.llm_client import get_default_llm_client


VERDICT_ICON = {"SUPPORTS": "✅", "REFUTES": "❌", "NOT_ENOUGH_INFO": "⚠️"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agentic automated fact-checking system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  factcheck "Saudi Arabia is the largest oil producer in the world."
  factcheck "The Earth is flat." --source "social_media"
  factcheck --file article.txt
  factcheck --file article.txt --max-claims 5 --min-importance 0.6
        """,
    )
    parser.add_argument("claim", nargs="?", help="A single claim to fact-check")
    parser.add_argument("--source", help="Source of the claim (optional)")
    parser.add_argument("--file", help="Fact-check every check-worthy claim in a text file")
    parser.add_argument("--text", help="Fact-check every check-worthy claim in this text")
    parser.add_argument(
        "--min-importance", type=float, default=0.5,
        help="Minimum check-worthiness for a claim to be verified (document mode)",
    )
    parser.add_argument(
        "--max-claims", type=int, default=None,
        help="Cap on claims verified per document. Also limits search spend.",
    )
    return parser


def _print_single(result: FactCheckResult) -> None:
    print("=" * 80)
    print("✅ FACT-CHECK RESULT")
    print("=" * 80)
    print(f"\n📊 Verdict: {result.verdict.label}")
    print(f"🎯 Confidence: {result.verdict.confidence:.0%}")

    if result.verdict.reasoning:
        print(f"\n🧠 Reasoning: {result.verdict.reasoning}")

    print(f"\n💭 Explanation:\n{result.explanation}\n")

    if result.evidence:
        print(f"Evidence ({len(result.evidence)} snippet(s)):")
        print("-" * 80)
        for i, snippet in enumerate(result.evidence, 1):
            preview = snippet.text[:80] + ("..." if len(snippet.text) > 80 else "")
            score_str = f" (score: {snippet.score:.2f})" if snippet.score is not None else ""
            print(f"{i}. [{snippet.source}]{score_str}")
            print(f"   {preview}\n")
    else:
        print("No evidence found.")
    print("=" * 80)


def _print_article(article: ArticleFactCheckResult) -> None:
    print("=" * 80)
    print("✅ DOCUMENT FACT-CHECK RESULT")
    print("=" * 80)
    print(f"\nClaims detected: {article.detected_claim_count}")
    print(f"Claims checked:  {article.checked_claim_count}")
    if article.failures:
        print(f"Claims failed:   {len(article.failures)}")
    print(f"Searches spent:  {article.searches_issued}")
    if article.budget_exhausted:
        print("⚠️  Per-article search budget was exhausted; some claims were skipped.")

    counts = article.counts_by_label()
    if counts:
        print("\nVerdicts: " + ", ".join(
            f"{VERDICT_ICON.get(label, '•')} {label}={n}" for label, n in sorted(counts.items())
        ))

    for i, result in enumerate(article.results, 1):
        icon = VERDICT_ICON.get(result.verdict.label, "•")
        print("\n" + "-" * 80)
        print(f"{icon} [{i}] {result.claim.raw_text}")
        print(f"    Verdict: {result.verdict.label} ({result.verdict.confidence:.0%}) "
              f"| {len(result.evidence)} evidence snippet(s)")
        print(f"    {result.explanation[:300]}")

    for failure in article.failures:
        print("\n" + "-" * 80)
        print(f"❗ Could not check: {failure.claim_text[:70]}")
        print(f"    {failure.error}")

    print("\n" + "=" * 80)


def main(args: Optional[list] = None) -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = _build_parser()
    parsed_args = parser.parse_args(args)

    document_text: Optional[str] = None
    if parsed_args.file:
        try:
            with open(parsed_args.file) as f:
                document_text = f.read().strip()
        except OSError as e:
            print(f"❌ Could not read {parsed_args.file}: {e}", file=sys.stderr)
            return 1
    elif parsed_args.text:
        document_text = parsed_args.text.strip()

    if not document_text and not parsed_args.claim:
        parser.print_help()
        return 1

    try:
        llm = get_default_llm_client(use_dummy_if_missing_key=False)
    except RuntimeError:
        print("❌ Error: Real LLM API key required for fact-checking.", file=sys.stderr)
        print("\nPlease set one of the following environment variables:", file=sys.stderr)
        print("  OPENAI_API_KEY=your_key_here", file=sys.stderr)
        print("  ANTHROPIC_API_KEY=your_key_here", file=sys.stderr)
        print("\nOr add them to your .env file.", file=sys.stderr)
        return 1

    from factcheck_agent.config import get_config
    config = get_config()
    if not (config.GOOGLE_SEARCH_API_KEY and config.GOOGLE_SEARCH_ENGINE_ID):
        print("⚠️  Warning: Search API not configured.", file=sys.stderr)
        print("   Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID for web search.",
              file=sys.stderr)
        print("   Continuing with limited retrieval capabilities...\n", file=sys.stderr)

    try:
        pipeline = FactCheckingPipeline(llm=llm)
    except RuntimeError as e:
        print(f"❌ Error initializing pipeline: {e}", file=sys.stderr)
        return 1

    try:
        if document_text:
            print(f"🔍 Fact-checking document ({len(document_text)} characters)\n")
            print("⏳ Detecting claims, then verifying each one. This may take a while.\n")
            article = asyncio.run(pipeline.process_text(
                document_text,
                min_importance=parsed_args.min_importance,
                max_claims=parsed_args.max_claims,
            ))
            _print_article(article)
        else:
            claim = Claim(
                id=str(uuid.uuid4()),
                raw_text=parsed_args.claim,
                metadata={"source": parsed_args.source} if parsed_args.source else None,
            )
            print(f"🔍 Fact-checking claim: {parsed_args.claim}\n")
            print("⏳ Processing... This may take a moment.\n")
            _print_single(asyncio.run(pipeline.process(claim)))
    except Exception as e:
        print(f"❌ Error during fact-checking: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
