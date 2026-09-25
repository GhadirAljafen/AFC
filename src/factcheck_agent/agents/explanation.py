"""Explanation agent using LLM to generate natural-language explanations."""

from factcheck_agent.llm_client import LLMClient
from factcheck_agent.models import Claim, EvidenceSnippet, FactCheckVerdict


class ExplanationAgent:
    """Generate explanations for fact-check verdicts."""

    def __init__(self, llm: LLMClient):
        """Initialize the explanation agent with an LLM client."""
        self.llm = llm

    async def generate(
        self, claim: Claim, verdict: FactCheckVerdict, evidence: list[EvidenceSnippet]
    ) -> str:
        """Generate a natural-language explanation using LLM."""
        if not evidence:
            return (
                f"The claim '{claim.normalized_text or claim.raw_text}' could not be verified "
                "because no relevant evidence was found."
            )

        # Build evidence context
        evidence_texts = "\n".join(
            [f"- [{e.source}] {e.text}" for e in evidence[:10]]
        )

        prompt = f"""You are a fact-checking assistant. Write a clear, concise explanation for why the verdict was reached.

Claim: {claim.normalized_text or claim.raw_text}

Verdict: {verdict.label}
Confidence: {verdict.confidence:.0%}
Reasoning: {verdict.reasoning or "Not provided"}

Evidence used:
{evidence_texts}

Write a 2-4 sentence explanation that:
1. States the verdict clearly
2. Summarizes the key evidence that led to this conclusion
3. Explains the reasoning in plain language

Be objective and factual. Do not include markdown formatting."""

        try:
            explanation = await self.llm.complete(prompt, temperature=0.3, max_tokens=300)
            return explanation.strip()
        except Exception as e:
            # Fallback explanation
            return (
                f"Verdict: {verdict.label} (confidence: {verdict.confidence:.0%}). "
                f"Based on {len(evidence)} evidence snippet(s). "
                f"{verdict.reasoning or 'Analysis completed.'}"
            )

