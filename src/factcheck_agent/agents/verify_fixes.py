import asyncio
from factcheck_agent.agents.claim_understanding import (
    ClaimUnderstandingAgent, ClaimDetectionError,
)

class BoomLLM:
    async def complete(self, *a, **k):
        raise RuntimeError("simulated API failure")

class GarbageLLM:
    async def complete(self, *a, **k):
        return "this is not json at all"

class FakeLLM:
    async def complete(self, prompt, **k):
        if "JSON array" in prompt:
            return '[{"claim_text": "Saudi Arabia is the largest oil producer", "sentence_index": 0, "importance": 0.9}]'
        return "Saudi Arabia is the largest oil producer"

async def main():
    print("--- Fix 1: error vs empty ---")
    agent = ClaimUnderstandingAgent(llm=BoomLLM())
    try:
        await agent.detect_claims("Some text here.", on_error="raise")
        print("FAIL: should have raised")
    except ClaimDetectionError:
        print("PASS: raises ClaimDetectionError on LLM failure")

    out = await agent.detect_claims("Some text here.", on_error="empty")
    print("PASS: empty fallback" if out == [] else f"FAIL: {out}")

    agent2 = ClaimUnderstandingAgent(llm=GarbageLLM())
    try:
        await agent2.detect_claims("Some text here.", on_error="raise")
        print("FAIL: bad JSON should have raised")
    except ClaimDetectionError:
        print("PASS: raises on unparseable JSON")

    out = await agent.detect_claims("", on_error="raise")
    print("PASS: empty input -> []" if out == [] else f"FAIL: {out}")

    print("--- Fix 3: sentence tokenizer ---")
    agent3 = ClaimUnderstandingAgent()
    sents = agent3._split_sentences("Dr. Smith met the U.S. team. Inflation hit 3.5% in Jan. 2026.")
    print(f"{'PASS' if len(sents) == 2 else 'FAIL'}: got {len(sents)} sentences")
    for i, s in enumerate(sents):
        print(f"  [{i}] {s}")

    print("--- Fix 2: offsets None, sentence_index kept ---")
    agent4 = ClaimUnderstandingAgent(llm=FakeLLM())
    claims = await agent4.detect_claims("Saudi Arabia produces the most oil.")
    c = claims[0]
    print("PASS" if c.start_char is None and c.end_char is None else "FAIL",
          "offsets:", c.start_char, c.end_char)
    print("  sentence_index:", c.sentence_index, "| normalized:", c.normalized_text)

asyncio.run(main())