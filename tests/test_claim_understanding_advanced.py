"""Tests for advanced claim understanding features."""

import asyncio

from factcheck_agent.llm_client import get_default_llm_client
from factcheck_agent.models import Claim, DetectedClaim
from factcheck_agent.agents.claim_understanding import ClaimUnderstandingAgent


def test_basic_normalization():
    """Basic normalization strips/collapses whitespace and PRESERVES case."""
    agent = ClaimUnderstandingAgent(llm=None)
    claim = Claim(id="test-1", raw_text="  This is a   TEST claim  ")
    normalized = agent.normalize_claim(claim)

    assert normalized.normalized_text == "This is a TEST claim"
    assert normalized.id == "test-1"


def test_detected_claim_model():
    """Test DetectedClaim model."""
    detected = DetectedClaim(
        id="dc1",
        raw_text="Saudi Arabia is the largest oil producer",
        start_char=10,
        end_char=50,
        sentence_index=2,
        importance=0.85,
    )
    
    assert detected.id == "dc1"
    assert detected.importance == 0.85
    assert detected.sentence_index == 2
    assert "importance: 0.85" in str(detected)


async def test_llm_normalization():
    """Test LLM-based normalization (requires real API key)."""
    try:
        llm = get_default_llm_client(use_dummy_if_missing_key=False)
    except RuntimeError:
        print("⚠️  Skipping LLM normalization test - no API key")
        return
    
    agent = ClaimUnderstandingAgent(llm=llm)
    claim = Claim(id="test-2", raw_text="I heard that like Saudi Arabia became the biggest oil producer in the world??")
    
    normalized = await agent.normalize_claim_llm(claim)
    
    assert normalized.normalized_text is not None
    assert len(normalized.normalized_text) > 0
    assert normalized.normalized_text != claim.raw_text
    print(f"✅ LLM normalization: '{claim.raw_text}' -> '{normalized.normalized_text}'")


async def test_claim_detection():
    """Test claim detection from long text (requires real API key)."""
    try:
        llm = get_default_llm_client(use_dummy_if_missing_key=False)
    except RuntimeError:
        print("⚠️  Skipping claim detection test - no API key")
        return
    
    agent = ClaimUnderstandingAgent(llm=llm)
    
    text = """
    Saudi Arabia announced plans to invest $100 billion in renewable energy by 2030.
    The country is also the world's largest oil producer, according to recent reports.
    Climate change is a serious concern for many nations.
    """
    
    detected_claims = await agent.detect_claims(text, min_importance=0.5)
    
    assert isinstance(detected_claims, list)
    print(f"✅ Detected {len(detected_claims)} claim(s)")
    
    for claim in detected_claims:
        assert isinstance(claim, DetectedClaim)
        assert claim.importance is not None
        assert claim.importance >= 0.5
        print(f"   - {claim.raw_text[:60]}... (importance: {claim.importance:.2f})")


async def main():
    """Run all tests."""
    print("🧪 Testing Claim Understanding Agent\n")
    
    test_basic_normalization()
    print("✅ Basic normalization test passed")
    
    test_detected_claim_model()
    print("✅ DetectedClaim model test passed")
    
    await test_llm_normalization()
    await test_claim_detection()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())

