"""Benchmark script to compare performance and accuracy of custom vs LangChain implementations."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

from factcheck_agent.models import Claim
from factcheck_agent.agents.claim_understanding import ClaimUnderstandingAgent
from factcheck_agent.llm_client import get_default_llm_client

# Check if LangChain version is available
try:
    from factcheck_agent.agents.claim_understanding_langchain import (
        ClaimUnderstandingAgentLangChain,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ClaimUnderstandingAgentLangChain = None


class BenchmarkResults:
    """Container for benchmark results."""

    def __init__(self):
        self.normalization_times_custom: List[float] = []
        self.normalization_times_langchain: List[float] = []
        self.detection_times_custom: List[float] = []
        self.detection_times_langchain: List[float] = []
        self.normalization_matches: int = 0
        self.normalization_total: int = 0
        self.detection_counts_custom: List[int] = []
        self.detection_counts_langchain: List[int] = []

    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            "normalization": {
                "custom_avg_time": (
                    sum(self.normalization_times_custom) / len(self.normalization_times_custom)
                    if self.normalization_times_custom
                    else 0
                ),
                "langchain_avg_time": (
                    sum(self.normalization_times_langchain) / len(self.normalization_times_langchain)
                    if self.normalization_times_langchain
                    else 0
                ),
                "matches": self.normalization_matches,
                "total": self.normalization_total,
                "match_rate": (
                    self.normalization_matches / self.normalization_total
                    if self.normalization_total > 0
                    else 0
                ),
            },
            "detection": {
                "custom_avg_time": (
                    sum(self.detection_times_custom) / len(self.detection_times_custom)
                    if self.detection_times_custom
                    else 0
                ),
                "langchain_avg_time": (
                    sum(self.detection_times_langchain) / len(self.detection_times_langchain)
                    if self.detection_times_langchain
                    else 0
                ),
                "custom_avg_claims": (
                    sum(self.detection_counts_custom) / len(self.detection_counts_custom)
                    if self.detection_counts_custom
                    else 0
                ),
                "langchain_avg_claims": (
                    sum(self.detection_counts_langchain) / len(self.detection_counts_langchain)
                    if self.detection_counts_langchain
                    else 0
                ),
            },
        }


async def benchmark_normalization(
    test_claims: List[str],
    custom_agent: ClaimUnderstandingAgent,
    langchain_agent: ClaimUnderstandingAgentLangChain,
    results: BenchmarkResults,
):
    """Benchmark normalization performance."""
    print("📊 Benchmarking normalization...")

    for claim_text in test_claims:
        # Custom
        claim = Claim(id="bench-custom", raw_text=claim_text)
        start = time.time()
        custom_result = await custom_agent.normalize_claim_llm(claim)
        custom_time = time.time() - start
        results.normalization_times_custom.append(custom_time)

        # LangChain
        claim2 = Claim(id="bench-langchain", raw_text=claim_text)
        start = time.time()
        langchain_result = await langchain_agent.normalize_claim_llm(claim2)
        langchain_time = time.time() - start
        results.normalization_times_langchain.append(langchain_time)

        # Check match
        if (
            custom_result.normalized_text.lower().strip()
            == langchain_result.normalized_text.lower().strip()
        ):
            results.normalization_matches += 1
        results.normalization_total += 1

    print(f"   ✅ Completed {len(test_claims)} normalizations")


async def benchmark_detection(
    test_texts: List[str],
    custom_agent: ClaimUnderstandingAgent,
    langchain_agent: ClaimUnderstandingAgentLangChain,
    results: BenchmarkResults,
    min_importance: float = 0.5,
):
    """Benchmark detection performance."""
    print("📊 Benchmarking detection...")

    for text in test_texts:
        # Custom
        start = time.time()
        custom_claims = await custom_agent.detect_claims(text, min_importance=min_importance)
        custom_time = time.time() - start
        results.detection_times_custom.append(custom_time)
        results.detection_counts_custom.append(len(custom_claims))

        # LangChain
        start = time.time()
        langchain_claims = await langchain_agent.detect_claims(
            text, min_importance=min_importance
        )
        langchain_time = time.time() - start
        results.detection_times_langchain.append(langchain_time)
        results.detection_counts_langchain.append(len(langchain_claims))

    print(f"   ✅ Completed {len(test_texts)} detections")


async def main():
    """Run benchmark."""
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain packages not installed.")
        print("   Install with: pip install langchain langchain-core langchain-openai langchain-anthropic")
        return

    # Initialize agents
    try:
        llm = get_default_llm_client(use_dummy_if_missing_key=False)
        custom_agent = ClaimUnderstandingAgent(llm=llm)
        langchain_agent = ClaimUnderstandingAgentLangChain(provider="openai")
        print("✅ Agents initialized\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Test data
    test_claims = [
        "I heard that Saudi Arabia is the world's largest oil producer, you know?",
        "The Earth is flat, according to some people.",
        "In 2020, COVID-19 pandemic started.",
        "He said the economy is growing rapidly.",
        "Some experts believe that climate change is accelerating.",
    ]

    test_texts = [
        """
        Saudi Arabia is the world's largest oil producer. The country has been 
        working to diversify its economy. In 2023, the government announced new 
        renewable energy projects.
        """,
        """
        The capital city is Riyadh. The population exceeds 35 million people.
        The country has a rich cultural heritage dating back thousands of years.
        """,
    ]

    results = BenchmarkResults()

    # Run benchmarks
    await benchmark_normalization(test_claims, custom_agent, langchain_agent, results)
    await benchmark_detection(test_texts, custom_agent, langchain_agent, results)

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    results_dict = results.to_dict()

    print("\n📊 Normalization:")
    print(f"   Custom avg time:    {results_dict['normalization']['custom_avg_time']:.3f}s")
    print(f"   LangChain avg time: {results_dict['normalization']['langchain_avg_time']:.3f}s")
    print(f"   Match rate:         {results_dict['normalization']['match_rate']:.1%}")

    print("\n📊 Detection:")
    print(f"   Custom avg time:    {results_dict['detection']['custom_avg_time']:.3f}s")
    print(f"   LangChain avg time: {results_dict['detection']['langchain_avg_time']:.3f}s")
    print(f"   Custom avg claims:  {results_dict['detection']['custom_avg_claims']:.1f}")
    print(f"   LangChain avg claims: {results_dict['detection']['langchain_avg_claims']:.1f}")

    # Save results
    output_file = Path("benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

