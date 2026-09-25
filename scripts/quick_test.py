#!/usr/bin/env python3
"""Quick test script to verify the fact-checking system works end-to-end."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factcheck_agent.llm_client import get_default_llm_client
from factcheck_agent.pipeline import FactCheckingPipeline
from factcheck_agent.models import Claim


async def test_fact_check(claim_text: str):
    """Test the fact-checking pipeline with a real claim."""
    print("=" * 80)
    print("🧪 Testing Fact-Checking System")
    print("=" * 80)
    print(f"\n📝 Claim: {claim_text}\n")
    
    try:
        # Get real LLM (no dummies)
        print("🔧 Initializing LLM...")
        llm = get_default_llm_client(use_dummy_if_missing_key=False)
        print(f"   ✅ Using: {type(llm).__name__}\n")
        
        # Create pipeline
        print("🔧 Initializing pipeline...")
        pipeline = FactCheckingPipeline(llm=llm)
        print("   ✅ Pipeline ready\n")
        
        # Create claim
        claim = Claim(id="test-1", raw_text=claim_text)
        
        # Process
        print("⏳ Processing claim... This may take 30-60 seconds...\n")
        result = await pipeline.process(claim)
        
        # Display results
        print("=" * 80)
        print("✅ RESULTS")
        print("=" * 80)
        print(f"\n📊 Verdict: {result.verdict.label}")
        print(f"🎯 Confidence: {result.verdict.confidence:.0%}")
        
        if result.verdict.reasoning:
            print(f"\n🧠 Reasoning:\n{result.verdict.reasoning}")
        
        print(f"\n💭 Explanation:\n{result.explanation}\n")
        
        if result.evidence:
            print(f"📚 Evidence ({len(result.evidence)} snippet(s)):")
            print("-" * 80)
            for i, e in enumerate(result.evidence[:5], 1):  # Show first 5
                title = e.metadata.get('title', 'No title') if e.metadata else 'No title'
                url = e.metadata.get('url', '') if e.metadata else ''
                print(f"\n[{i}] {title}")
                if url:
                    print(f"    URL: {url}")
                print(f"    Source: {e.source}")
                preview = e.text[:150] + "..." if len(e.text) > 150 else e.text
                print(f"    {preview}")
        else:
            print("📚 No evidence retrieved")
        
        print("\n" + "=" * 80)
        print("✅ Test completed successfully!")
        print("=" * 80)
        
        return True
        
    except RuntimeError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease check your API keys. See setup_guide.md for help.")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    # Default test claim
    test_claim = "Saudi Arabia is the largest oil producer in the world."
    
    if len(sys.argv) > 1:
        test_claim = " ".join(sys.argv[1:])
    
    success = asyncio.run(test_fact_check(test_claim))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

