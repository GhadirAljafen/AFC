"""Smoke tests for agent skeletons."""

from factcheck_agent.agents import (
    ClaimUnderstandingAgent,
    EvidenceEvaluationAgent,
    EvidenceSelectionAgent,
    ExplanationAgent,
    ReasoningAndVerdictAgent,
    RetrievalAgent,
)
from factcheck_agent.llm_client import LLMClient
from factcheck_agent.models import Claim, EvidenceSnippet


class DummyLLM(LLMClient):
    async def complete(self, prompt: str, **kwargs):
        return f"completed: {prompt}"

    async def chat(self, messages, **kwargs):
        return "chat response"


def test_claim_understanding_normalizes():
    llm = DummyLLM()
    agent = ClaimUnderstandingAgent(llm=llm)
    claim = Claim(id="c1", raw_text="  Mixed CASE Claim  ")
    normalized = agent.normalize_claim(claim)
    assert normalized.normalized_text == "mixed case claim"


def test_retrieval_returns_subset():
    corpus = [
        EvidenceSnippet(id="e1", source="wiki", text="a", score=0.9),
        EvidenceSnippet(id="e2", source="news", text="b", score=0.8),
    ]
    agent = RetrievalAgent(corpus=corpus)
    claim = Claim(id="c1", raw_text="x")
    results = agent.retrieve_evidence(claim, k=1)
    assert len(results) == 1
    assert results[0].id == "e1"


def test_selection_returns_first_k():
    candidates = [
        EvidenceSnippet(id="e1", source="wiki", text="a"),
        EvidenceSnippet(id="e2", source="news", text="b"),
    ]
    agent = EvidenceSelectionAgent()
    claim = Claim(id="c1", raw_text="x")
    selected = agent.select(claim, candidates, k=1)
    assert len(selected) == 1
    assert selected[0].id == "e1"


def test_evaluation_placeholder():
    agent = EvidenceEvaluationAgent()
    claim = Claim(id="c1", raw_text="x")
    evidence = [EvidenceSnippet(id="e1", source="wiki", text="a")]
    result = agent.evaluate(claim, evidence)
    assert result.sufficient is True
    assert result.confidence == 0.5
    assert result.selected_evidence == evidence


def test_reasoning_and_verdict_placeholder():
    agent = ReasoningAndVerdictAgent()
    claim = Claim(id="c1", raw_text="x")
    evidence = [EvidenceSnippet(id="e1", source="wiki", text="a")]
    verdict = agent.decide(claim, evidence)
    assert verdict.label == "NOT_ENOUGH_INFO"
    assert verdict.used_evidence_ids == ["e1"]


def test_explanation_placeholder():
    agent = ExplanationAgent()
    claim = Claim(id="c1", raw_text="x")
    evidence = [EvidenceSnippet(id="e1", source="wiki", text="a")]
    from factcheck_agent.models import FactCheckVerdict

    verdict = FactCheckVerdict(
        label="NOT_ENOUGH_INFO",
        confidence=0.2,
        used_evidence_ids=["e1"],
    )
    explanation = agent.generate(claim, verdict, evidence)
    assert "NOT_ENOUGH_INFO" in explanation
    assert "1 evidence snippet" in explanation

