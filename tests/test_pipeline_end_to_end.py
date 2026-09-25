"""End-to-end tests for the fact-checking pipeline."""

import asyncio
import os

from factcheck_agent.llm_client import get_default_llm_client
from factcheck_agent.models import Claim
from factcheck_agent.pipeline import FactCheckingPipeline
from factcheck_agent.agents.retrieval import RetrievalAgent


async def test_pipeline_with_google_search():
    """Test the full pipeline with Google Search retrieval."""
    # Check if Google Search API keys are configured
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key or not engine_id:
        print("⚠️  Skipping Google Search test - API keys not configured")
        return
    
    # Check if httpx is installed
    try:
        import httpx
    except ImportError:
        print("⚠️  Skipping Google Search test - httpx not installed")
        print("   Install with: pip install httpx")
        return
    
    # Initialize pipeline
    llm = get_default_llm_client(use_dummy_if_missing_key=True)
    retrieval_agent = RetrievalAgent(llm=llm)
    pipeline = FactCheckingPipeline(llm=llm, retrieval_agent=retrieval_agent)
    
    # Test claim
    claim = Claim(
        id="test-1",
        raw_text="Saudi Arabia is the largest oil producer in the world.",
    )
    
    print("🔍 Running full fact-checking pipeline...")
    print(f"Claim: {claim.raw_text}\n")
    
    try:
        result = await pipeline.process(claim)
        
        print("=" * 80)
        print("FACT-CHECK RESULT")
        print("=" * 80)
        print(f"\n✅ Verdict: {result.verdict.label}")
        print(f"📊 Confidence: {result.verdict.confidence:.0%}")
        print(f"\n💭 Explanation:\n{result.explanation}\n")
        
        if result.verdict.reasoning:
            print(f"🧠 Reasoning: {result.verdict.reasoning}\n")
        
        print(f"📚 Evidence used: {len(result.evidence)} snippet(s)")
        for i, e in enumerate(result.evidence[:5], 1):  # Show first 5
            print(f"\n  [{i}] {e.metadata.get('title', 'No title')}")
            print(f"      Source: {e.source}")
            if e.metadata.get('url'):
                print(f"      URL: {e.metadata.get('url')}")
        
        print("\n" + "=" * 80)
        
        # Basic assertions
        assert result.verdict.label in ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        assert 0.0 <= result.verdict.confidence <= 1.0
        assert len(result.explanation) > 0
        assert isinstance(result.evidence, list)
        
        print("\n✅ All assertions passed!")
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        raise


async def test_pipeline_without_retrieval():
    """Test pipeline components without actual retrieval (for testing logic)."""
    print("\n" + "=" * 80)
    print("Testing pipeline components (without retrieval)")
    print("=" * 80)
    
    llm = get_default_llm_client(use_dummy_if_missing_key=True)
    
    # Create a simple mock retrieval agent that returns empty results
    from factcheck_agent.models import EvidenceSnippet
    
    class SimpleMockRetrieval:
        async def retrieve_evidence(self, claim):
            return []  # Return empty for testing
    
    retrieval_agent = SimpleMockRetrieval()
    
    pipeline = FactCheckingPipeline(llm=llm, retrieval_agent=retrieval_agent)
    
    claim = Claim(
        id="test-2",
        raw_text="The Earth is flat.",
    )
    
    print(f"\nClaim: {claim.raw_text}")
    result = await pipeline.process(claim)
    
    print(f"\n✅ Verdict: {result.verdict.label}")
    print(f"📊 Confidence: {result.verdict.confidence:.0%}")
    print(f"💭 Explanation: {result.explanation[:200]}...")
    
    assert result.verdict.label in ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    print("\n✅ Component test passed!")


async def main():
    """Run all pipeline tests."""
    print("🧪 Running end-to-end pipeline tests\n")
    
    # Test 1: Full pipeline with Google Search (if available)
    await test_pipeline_with_google_search()
    
    # Test 2: Pipeline components without retrieval
    await test_pipeline_without_retrieval()
    
    print("\n" + "=" * 80)
    print("✅ All pipeline tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

