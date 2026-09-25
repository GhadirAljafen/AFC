"""Comparison tests between custom and LangChain implementations of ClaimUnderstandingAgent.

Focus: Detecting check-worthy claims from long text documents.
"""

import asyncio
import time
from typing import List, Tuple, Dict, Set

from factcheck_agent.models import Claim, DetectedClaim
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


# Test articles with various check-worthy claims
TEST_ARTICLES = [
    {
        "title": "Energy and Economy Article",
        "text": """
        Saudi Arabia is the world's largest oil producer, with daily production exceeding 
        10 million barrels. The country has been working to diversify its economy through 
        Vision 2030, a strategic framework launched in 2016. In 2023, the government 
        announced new renewable energy projects worth $50 billion. The capital city is 
        Riyadh, which has a population of over 7 million people. The total population of 
        Saudi Arabia exceeds 35 million people. The country's GDP grew by 8.7% in 2022, 
        according to official statistics.
        """,
        "expected_claims": [
            "Saudi Arabia is the world's largest oil producer",
            "Saudi Arabia produces over 10 million barrels of oil daily",
            "Vision 2030 was launched in 2016",
            "In 2023, the government announced renewable energy projects worth $50 billion",
            "Riyadh has a population of over 7 million people",
            "Saudi Arabia's total population exceeds 35 million people",
            "Saudi Arabia's GDP grew by 8.7% in 2022",
        ],
    },
    {
        "title": "Climate Change Article",
        "text": """
        Climate scientists have warned that global temperatures could rise by 2.5 degrees 
        Celsius by 2050 if current trends continue. The Intergovernmental Panel on Climate 
        Change (IPCC) released a report in 2023 stating that carbon emissions must be 
        reduced by 45% by 2030 to limit warming. Some experts believe that renewable 
        energy sources could provide 80% of global electricity by 2050. However, critics 
        argue that the transition will cost trillions of dollars.
        """,
        "expected_claims": [
            "Global temperatures could rise by 2.5 degrees Celsius by 2050",
            "The IPCC released a report in 2023",
            "Carbon emissions must be reduced by 45% by 2030",
            "Renewable energy could provide 80% of global electricity by 2050",
        ],
    },
    {
        "title": "Technology and Innovation Article",
        "text": """
        Artificial intelligence companies raised over $50 billion in funding in 2023. 
        OpenAI's ChatGPT reached 100 million users within two months of its launch. 
        According to a study published in Nature, quantum computers could break current 
        encryption methods within the next decade. The global semiconductor market is 
        expected to reach $1 trillion by 2030.
        """,
        "expected_claims": [
            "AI companies raised over $50 billion in funding in 2023",
            "ChatGPT reached 100 million users within two months",
            "Quantum computers could break current encryption methods within the next decade",
            "The global semiconductor market is expected to reach $1 trillion by 2030",
        ],
    },
]


async def compare_claim_detection_detailed(
    article: Dict[str, str],
    custom_agent: ClaimUnderstandingAgent,
    langchain_agent: ClaimUnderstandingAgentLangChain,
    min_importance: float = 0.5,
) -> Dict:
    """
    Detailed comparison of claim detection from long text.
    
    Returns:
        Dictionary with comparison metrics and results
    """
    print("\n" + "=" * 80)
    print(f"ARTICLE: {article['title']}")
    print("=" * 80)
    
    text = article["text"].strip()
    print(f"\n📄 Text length: {len(text)} characters")
    print(f"📝 Text preview: {text[:150]}...\n")

    # Custom implementation
    print("🔧 Custom Implementation:")
    print("-" * 80)
    start_time = time.time()
    custom_claims = await custom_agent.detect_claims(
        text, min_importance=min_importance, normalize_detected=True
    )
    custom_time = time.time() - start_time
    
    print(f"⏱️  Time: {custom_time:.2f}s")
    print(f"📊 Detected {len(custom_claims)} check-worthy claims:\n")
    for i, claim in enumerate(custom_claims, 1):
        importance_bar = "█" * int(claim.importance * 20) if claim.importance else ""
        print(
            f"   {i}. [{claim.importance:.2f}] {importance_bar}\n"
            f"      \"{claim.raw_text}\""
        )
        if claim.normalized_text and claim.normalized_text != claim.raw_text:
            print(f"      Normalized: \"{claim.normalized_text}\"")

    # LangChain implementation
    print(f"\n🔗 LangChain Implementation:")
    print("-" * 80)
    start_time = time.time()
    langchain_claims = await langchain_agent.detect_claims(
        text, min_importance=min_importance, normalize_detected=True
    )
    langchain_time = time.time() - start_time
    
    print(f"⏱️  Time: {langchain_time:.2f}s")
    print(f"📊 Detected {len(langchain_claims)} check-worthy claims:\n")
    for i, claim in enumerate(langchain_claims, 1):
        importance_bar = "█" * int(claim.importance * 20) if claim.importance else ""
        print(
            f"   {i}. [{claim.importance:.2f}] {importance_bar}\n"
            f"      \"{claim.raw_text}\""
        )
        if claim.normalized_text and claim.normalized_text != claim.raw_text:
            print(f"      Normalized: \"{claim.normalized_text}\"")

    # Detailed comparison metrics
    print(f"\n📈 DETAILED COMPARISON:")
    print("-" * 80)
    
    # Extract claim texts (normalized for comparison)
    custom_texts = {
        c.normalized_text.lower().strip() if c.normalized_text else c.raw_text.lower().strip()
        for c in custom_claims
    }
    langchain_texts = {
        c.normalized_text.lower().strip() if c.normalized_text else c.raw_text.lower().strip()
        for c in langchain_claims
    }
    
    # Find overlaps
    common_claims = custom_texts & langchain_texts
    only_custom = custom_texts - langchain_texts
    only_langchain = langchain_texts - custom_texts
    
    print(f"✅ Common claims: {len(common_claims)}")
    if common_claims:
        print("   Both implementations detected:")
        for claim_text in sorted(common_claims)[:5]:  # Show first 5
            print(f"      • {claim_text[:80]}...")
    
    print(f"\n🔧 Only in Custom: {len(only_custom)}")
    if only_custom:
        for claim_text in sorted(only_custom)[:3]:  # Show first 3
            print(f"      • {claim_text[:80]}...")
    
    print(f"\n🔗 Only in LangChain: {len(only_langchain)}")
    if only_langchain:
        for claim_text in sorted(only_langchain)[:3]:  # Show first 3
            print(f"      • {claim_text[:80]}...")
    
    # Importance score comparison for common claims
    if common_claims:
        print(f"\n📊 Importance Score Comparison (for common claims):")
        custom_by_text = {
            (c.normalized_text.lower().strip() if c.normalized_text else c.raw_text.lower().strip()): c.importance
            for c in custom_claims
        }
        langchain_by_text = {
            (c.normalized_text.lower().strip() if c.normalized_text else c.raw_text.lower().strip()): c.importance
            for c in langchain_claims
        }
        
        score_diffs = []
        for claim_text in common_claims:
            custom_score = custom_by_text.get(claim_text, 0)
            langchain_score = langchain_by_text.get(claim_text, 0)
            diff = abs(custom_score - langchain_score)
            score_diffs.append(diff)
        
        avg_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0
        print(f"   Average importance score difference: {avg_diff:.3f}")
        print(f"   Max difference: {max(score_diffs):.3f}" if score_diffs else "")
    
    # Performance metrics
    print(f"\n⚡ Performance:")
    print(f"   Custom:    {custom_time:.2f}s ({len(custom_claims)/custom_time:.1f} claims/sec)" if custom_time > 0 else f"   Custom:    {custom_time:.2f}s")
    print(f"   LangChain: {langchain_time:.2f}s ({len(langchain_claims)/langchain_time:.1f} claims/sec)" if langchain_time > 0 else f"   LangChain: {langchain_time:.2f}s")
    speedup = custom_time / langchain_time if langchain_time > 0 else 0
    if speedup > 1:
        print(f"   LangChain is {speedup:.2f}x faster")
    elif speedup < 1 and speedup > 0:
        print(f"   Custom is {1/speedup:.2f}x faster")
    
    # Coverage metrics (if expected claims provided)
    if "expected_claims" in article:
        expected_set = {c.lower().strip() for c in article["expected_claims"]}
        custom_coverage = len(custom_texts & expected_set) / len(expected_set) if expected_set else 0
        langchain_coverage = len(langchain_texts & expected_set) / len(expected_set) if expected_set else 0
        
        print(f"\n🎯 Coverage (vs expected claims):")
        print(f"   Custom:    {custom_coverage:.1%} ({len(custom_texts & expected_set)}/{len(expected_set)})")
        print(f"   LangChain: {langchain_coverage:.1%} ({len(langchain_texts & expected_set)}/{len(expected_set)})")
    
    return {
        "custom_claims": custom_claims,
        "langchain_claims": langchain_claims,
        "custom_time": custom_time,
        "langchain_time": langchain_time,
        "common_count": len(common_claims),
        "only_custom_count": len(only_custom),
        "only_langchain_count": len(only_langchain),
    }


async def compare_with_different_thresholds(
    article: Dict[str, str],
    custom_agent: ClaimUnderstandingAgent,
    langchain_agent: ClaimUnderstandingAgentLangChain,
) -> None:
    """Compare detection with different importance thresholds."""
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    text = article["text"].strip()
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    print(f"\n📊 Claims detected at different importance thresholds:\n")
    print(f"{'Threshold':<12} {'Custom':<10} {'LangChain':<12} {'Common':<10}")
    print("-" * 50)
    
    for threshold in thresholds:
        custom_claims = await custom_agent.detect_claims(text, min_importance=threshold)
        langchain_claims = await langchain_agent.detect_claims(text, min_importance=threshold)
        
        custom_texts = {c.raw_text.lower().strip() for c in custom_claims}
        langchain_texts = {c.raw_text.lower().strip() for c in langchain_claims}
        common = len(custom_texts & langchain_texts)
        
        print(f"{threshold:<12.1f} {len(custom_claims):<10} {len(langchain_claims):<12} {common:<10}")


async def run_comprehensive_comparison():
    """Run comprehensive comparison focusing on claim detection from long text."""
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain packages not installed. Skipping comparison.")
        print("   Install with: pip install langchain langchain-core langchain-openai langchain-anthropic")
        return

    # Initialize agents
    try:
        llm = get_default_llm_client(use_dummy_if_missing_key=False)
        custom_agent = ClaimUnderstandingAgent(llm=llm)
        langchain_agent = ClaimUnderstandingAgentLangChain(provider="openai")
        print("✅ Both agents initialized successfully")
        print("=" * 80)
        print("CHECK-WORTHY CLAIM DETECTION COMPARISON")
        print("=" * 80)
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        return

    # Compare detection for each article
    all_results = []
    for article in TEST_ARTICLES:
        result = await compare_claim_detection_detailed(
            article, custom_agent, langchain_agent, min_importance=0.5
        )
        all_results.append(result)
        
        # Threshold sensitivity for first article
        if article == TEST_ARTICLES[0]:
            await compare_with_different_thresholds(
                article, custom_agent, langchain_agent
            )

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    total_custom = sum(len(r["custom_claims"]) for r in all_results)
    total_langchain = sum(len(r["langchain_claims"]) for r in all_results)
    total_common = sum(r["common_count"] for r in all_results)
    total_custom_time = sum(r["custom_time"] for r in all_results)
    total_langchain_time = sum(r["langchain_time"] for r in all_results)
    
    print(f"\n📊 Across all {len(TEST_ARTICLES)} articles:")
    print(f"   Custom total claims:    {total_custom}")
    print(f"   LangChain total claims: {total_langchain}")
    print(f"   Common claims:          {total_common}")
    if max(total_custom, total_langchain) > 0:
        print(f"   Agreement rate:        {total_common / max(total_custom, total_langchain):.1%}")
    
    print(f"\n⚡ Total time:")
    print(f"   Custom:    {total_custom_time:.2f}s")
    print(f"   LangChain: {total_langchain_time:.2f}s")
    
    avg_custom = sum(len(r["custom_claims"]) for r in all_results) / len(all_results)
    avg_langchain = sum(len(r["langchain_claims"]) for r in all_results) / len(all_results)
    print(f"\n📈 Average claims per article:")
    print(f"   Custom:    {avg_custom:.1f}")
    print(f"   LangChain: {avg_langchain:.1f}")


async def main():
    """Main entry point."""
    await run_comprehensive_comparison()


if __name__ == "__main__":
    asyncio.run(main())
