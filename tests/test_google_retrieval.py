# tests/test_google_retrieval.py

import asyncio
import os

from factcheck_agent.models import Claim
from factcheck_agent.agents.retrieval import RetrievalAgent


async def main() -> None:
    # Check if Google Search API keys are configured
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    if not api_key or not engine_id:
        print("⚠️  Google Search API keys are not configured.")
        print("\nTo use Google Search, you need to:")
        print("1. Get a Google Custom Search API key from: https://console.cloud.google.com/")
        print("2. Create a Custom Search Engine at: https://programmablesearchengine.google.com/")
        print("3. Set environment variables:")
        print("   export GOOGLE_SEARCH_API_KEY=your_api_key")
        print("   export GOOGLE_SEARCH_ENGINE_ID=your_engine_id")
        print("\nOr add them to your .env file:")
        print("   GOOGLE_SEARCH_API_KEY=your_api_key")
        print("   GOOGLE_SEARCH_ENGINE_ID=your_engine_id")
        print("\nFor now, skipping the test.")
        return

    # Try RetrievalAgent without LLM (uses claim text directly as query)
    try:
        retrieval_agent = RetrievalAgent(llm=None)
    except Exception as e:
        print(f"Error initializing RetrievalAgent: {e}")
        return

    claim = Claim(
        id="c1",
        raw_text="Saudi Arabia is the largest oil producer in the world.",
    )

    # This function is async, so we need to use await
    try:
        print("🔍 Searching for evidence...")
        snippets = await retrieval_agent.retrieve_evidence(claim)
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        return

    print(f"\n✅ Got {len(snippets)} evidence snippets:")
    for i, e in enumerate(snippets, start=1):
        print(f"\n[{i}] {e.metadata.get('title', 'No title')}")
        print(f"    URL: {e.metadata.get('url', 'No URL')}")
        print(f"    Snippet: {e.text[:150]}..." if len(e.text) > 150 else f"    Snippet: {e.text}")


if __name__ == "__main__":
    asyncio.run(main())
