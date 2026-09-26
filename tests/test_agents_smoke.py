"""
Smoke tests: every agent can be constructed and called, and honours its contract.

REWRITTEN. The previous version was written against a pre-LLM skeleton and had
rotted badly — and worse, it reported success while testing nothing: there was no
`__main__` block, so running the file executed zero assertions and exited 0. When
the functions were invoked directly, 5 of 6 failed:

  * `RetrievalAgent(corpus=...)` — that keyword no longer exists
  * `EvidenceEvaluationAgent.evaluate` / `ReasoningAndVerdictAgent.decide` treated
    as synchronous when both are `async`
  * `ReasoningAndVerdictAgent()` / `ExplanationAgent()` called with no `llm`,
    which is required
  * a normalization assertion expecting the lowercasing that was deliberately
    removed so entity casing survives

A test that passes without running is worse than no test, because it makes the
suite look green. This version actually runs, uses the current APIs, and asserts
contracts rather than the placeholder return values of long-replaced stubs.

Deeper behavioural coverage lives in `test_stage2_behaviour.py` (43 checks) and
`test_stage3_verdict.py`. This file only establishes that each agent is wired up.

Run:  python tests/test_agents_smoke.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factcheck_agent.agents.claim_understanding import ClaimUnderstandingAgent
from factcheck_agent.agents.evidence_evaluation import EvidenceEvaluationAgent
from factcheck_agent.agents.evidence_selection import EvidenceSelectionAgent
from factcheck_agent.agents.explanation import ExplanationAgent
from factcheck_agent.agents.reasoning_and_verdict import ReasoningAndVerdictAgent
from factcheck_agent.agents.retrieval import RetrievalAgent
from factcheck_agent.models import Claim, EvidenceSnippet
from factcheck_agent.search import SearchResult, StaticSearchClient


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(label)


CLAIM = Claim(id="smoke-1", raw_text="  Saudi Arabia is the   largest oil producer  ")
EVIDENCE = [
    EvidenceSnippet(id=f"https://e{i}.com", source="search",
                    text=f"evidence body number {i}", score=0.9 - i * 0.1)
    for i in range(4)
]


class StubLLM:
    """Answers each agent with something its parser accepts."""

    async def complete(self, prompt, **kwargs):
        if "JSON array" in prompt:                      # claim detection
            return '[{"claim_text": "A claim", "sentence_index": 0, "importance": 0.9}]'
        if "evidence is sufficient" in prompt:          # sufficiency
            return '{"sufficient": true, "confidence": 0.8, "notes": "n", "gaps": ""}'
        if "SUPPORTS" in prompt and "JSON object" in prompt:   # verdict
            return '{"label": "REFUTES", "confidence": 0.85, "reasoning": "r"}'
        return "A short generated text."

    async def chat(self, messages, **kwargs):           # query generation
        return "query one\nquery two"


def test_claim_understanding_preserves_case():
    agent = ClaimUnderstandingAgent(llm=None)
    out = agent.normalize_claim(CLAIM)
    check("normalization collapses whitespace",
          out.normalized_text == "Saudi Arabia is the largest oil producer", out.normalized_text)
    check("normalization PRESERVES case (lowercasing was removed on purpose)",
          "Saudi Arabia" in out.normalized_text)


async def test_retrieval_returns_scored_snippets():
    search = StaticSearchClient(default=[
        SearchResult(url="https://a.com", title="A", snippet="oil producer output", rank=1),
        SearchResult(url="https://b.com", title="B", snippet="unrelated", rank=2),
    ])
    agent = RetrievalAgent(llm=None, search_client=search)
    out = await agent.retrieve_evidence(Claim(id="c", raw_text="oil producer"))
    check("retrieval returns EvidenceSnippets", out and all(
        isinstance(e, EvidenceSnippet) for e in out), f"{len(out)}")
    check("retrieval populates a relevance score (it used to be None)",
          all(e.score is not None for e in out))


def test_selection_returns_at_most_k():
    agent = EvidenceSelectionAgent()
    out = agent.select(CLAIM, EVIDENCE, k=2)
    check("selection respects k", len(out) == 2, f"{len(out)}")
    check("selection returns the given snippets", all(e in EVIDENCE for e in out))


async def test_evaluation_is_async_and_reports_gaps():
    agent = EvidenceEvaluationAgent(llm=StubLLM())
    result = await agent.evaluate(CLAIM, EVIDENCE)
    check("evaluation is awaited and yields a result object",
          hasattr(result, "sufficient") and isinstance(result.sufficient, bool))
    check("evaluation exposes a gaps field for the retrieval loop",
          hasattr(result, "gaps"))
    empty = await agent.evaluate(CLAIM, [])
    check("no evidence -> not sufficient", empty.sufficient is False)


async def test_verdict_requires_llm_and_is_async():
    try:
        ReasoningAndVerdictAgent()          # type: ignore[call-arg]
        check("verdict agent requires an llm", False, "constructed without one")
    except TypeError:
        check("verdict agent requires an llm", True)

    agent = ReasoningAndVerdictAgent(llm=StubLLM())
    verdict = await agent.decide(CLAIM, EVIDENCE)
    check("verdict label is one of the three allowed values",
          verdict.label in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"), verdict.label)
    check("confidence is in range", 0.0 <= verdict.confidence <= 1.0)
    empty = await agent.decide(CLAIM, [])
    check("no evidence -> NOT_ENOUGH_INFO", empty.label == "NOT_ENOUGH_INFO")


async def test_explanation_requires_llm_and_returns_text():
    try:
        ExplanationAgent()                  # type: ignore[call-arg]
        check("explanation agent requires an llm", False, "constructed without one")
    except TypeError:
        check("explanation agent requires an llm", True)

    agent = ExplanationAgent(llm=StubLLM())
    verdict = await ReasoningAndVerdictAgent(llm=StubLLM()).decide(CLAIM, EVIDENCE)
    text = await agent.generate(CLAIM, verdict, EVIDENCE)
    check("explanation returns non-empty text", isinstance(text, str) and text.strip())
    no_ev = await agent.generate(CLAIM, verdict, [])
    check("explanation handles empty evidence without raising", isinstance(no_ev, str))


async def main():
    print("--- agent smoke tests ---")
    test_claim_understanding_preserves_case()
    await test_retrieval_returns_scored_snippets()
    test_selection_returns_at_most_k()
    await test_evaluation_is_async_and_reports_gaps()
    await test_verdict_requires_llm_and_is_async()
    await test_explanation_requires_llm_and_returns_text()
    print("\nAll agent smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
