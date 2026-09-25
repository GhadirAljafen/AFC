#!/usr/bin/env python3
"""Test script for claim detection from long text."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factcheck_agent.llm_client import get_default_llm_client
from factcheck_agent.agents.claim_understanding import ClaimUnderstandingAgent


async def test_claim_detection():
    """Test detecting claims from a long text article."""
    print("=" * 80)
    print("🧪 Testing Claim Detection from Long Text")
    print("=" * 80)
    
    # Get real LLM
    try:
        llm = get_default_llm_client(use_dummy_if_missing_key=False)
        print(f"✅ Using LLM: {type(llm).__name__}\n")
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("\nPlease set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return False
    
    # Create agent
    agent = ClaimUnderstandingAgent(llm=llm)
    
    # Sample long text (news article style)
    sample_text = """
    Saudi Arabia announced plans to invest $100 billion in renewable energy by 2030, 
    according to a statement from the Ministry of Energy. The country, which is currently 
    the world's largest oil producer, aims to diversify its energy sources and reduce 
    carbon emissions. Climate experts have praised the initiative, noting that it represents 
    a significant shift in the country's energy policy. The investment will focus on solar 
    and wind power projects across the kingdom. Some analysts believe this move could 
    transform the global energy market. The announcement came during a climate summit 
    in Riyadh, where officials also revealed plans to achieve net-zero emissions by 2060.
    """
    
    print("📄 Input Text:")
    print("-" * 80)
    print(sample_text.strip())
    print("\n" + "=" * 80)
    print("🔍 Detecting check-worthy claims...\n")
    
    try:
        detected_claims = await agent.detect_claims(sample_text, min_importance=0.5)
        
        print(f"✅ Detected {len(detected_claims)} check-worthy claim(s):\n")
        
        for i, claim in enumerate(detected_claims, 1):
            print(f"[{i}] {claim.raw_text}")
            print(f"    Importance: {claim.importance:.0%}")
            if claim.sentence_index is not None:
                print(f"    Sentence index: {claim.sentence_index}")
            if claim.start_char is not None and claim.end_char is not None:
                print(f"    Position: chars {claim.start_char}-{claim.end_char}")
            print()
        
        if not detected_claims:
            print("⚠️  No claims detected above the importance threshold (0.5)")
            print("   Try lowering min_importance or check the input text.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during claim detection: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_with_real_article():
    """Test with a longer, more realistic article."""
    print("\n" + "=" * 80)
    print("📰 Testing with Longer Article Text")
    print("=" * 80)
    
    try:
        llm = get_default_llm_client(use_dummy_if_missing_key=False)
    except RuntimeError:
        print("⚠️  Skipping - no API key")
        return
    
    agent = ClaimUnderstandingAgent(llm=llm)
    
    article = """
    In a groundbreaking announcement today, Saudi Arabia revealed plans to invest 
    $100 billion in renewable energy infrastructure by 2030. The initiative, dubbed 
    "Vision 2030 Energy Transformation," aims to position the kingdom as a global leader 
    in clean energy while maintaining its status as the world's largest oil producer.
    
    Energy Minister Prince Abdulaziz bin Salman stated that the investment will create 
    over 200,000 jobs and reduce the country's carbon emissions by 30% within the next 
    decade. The plan includes construction of massive solar farms in the Empty Quarter 
    and wind farms along the Red Sea coast.
    
    International climate experts have praised the move. Dr. Sarah Chen from the 
    International Energy Agency called it "a paradigm shift" that could influence 
    other oil-producing nations. However, some analysts remain skeptical, pointing 
    out that Saudi Arabia still plans to increase oil production capacity to 13 million 
    barrels per day by 2027.
    
    The announcement comes as the kingdom prepares to host the 2030 World Expo, which 
    officials say will be powered entirely by renewable energy. Critics argue that the 
    timeline is overly ambitious, but supporters point to the country's successful 
    completion of the NEOM smart city project as evidence of its capability.
    """
    
    print("\n🔍 Analyzing article for check-worthy claims...\n")
    
    claims = await agent.detect_claims(article, min_importance=0.6)
    
    print(f"✅ Found {len(claims)} high-importance claims:\n")
    for i, claim in enumerate(claims, 1):
        print(f"{i}. {claim.raw_text}")
        print(f"   📊 Importance: {claim.importance:.0%}\n")


async def main():
    """Run all tests."""
    success = await test_claim_detection()
    
    if success:
        await test_with_real_article()
    
    print("=" * 80)
    if success:
        print("✅ Claim detection tests completed!")
    else:
        print("❌ Tests failed - check API keys")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

