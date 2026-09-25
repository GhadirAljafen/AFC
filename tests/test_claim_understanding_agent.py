# scripts/test_claim_understanding_agent.py

import asyncio

from factcheck_agent.llm_client import get_default_llm_client
from factcheck_agent.models import Claim
from factcheck_agent.agents.claim_understanding import ClaimUnderstandingAgent


async def main() -> None:
    # 1) Get an LLM client (real or dummy depending on your config)
    llm = get_default_llm_client()

    # 2) Create the agent
    agent = ClaimUnderstandingAgent(llm=llm)

    # 3) Create a messy raw claim
    raw_claim = "I heard that like last year Saudi Arabia became the biggest oil producer in the world??"
    claim = Claim(id="test-1", raw_text=raw_claim)

    # 4) Call the agent
    normalized_claim = await agent.normalize_claim(claim)

    # 5) Inspect the result
    print("RAW:       ", claim.raw_text)
    print("NORMALIZED:", normalized_claim.normalized_text)


if __name__ == "__main__":
    asyncio.run(main())
