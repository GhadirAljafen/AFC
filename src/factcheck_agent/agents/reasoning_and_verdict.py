"""Reasoning and verdict agent using LLM for fact-checking decisions."""

from factcheck_agent.llm_client import LLMClient
from factcheck_agent.models import Claim, EvidenceSnippet, FactCheckVerdict


class ReasoningAndVerdictAgent:
    """Decide on a verdict given a claim and supporting evidence."""

    def __init__(self, llm: LLMClient):
        """Initialize the reasoning agent with an LLM client."""
        self.llm = llm

    async def decide(self, claim: Claim, evidence: list[EvidenceSnippet]) -> FactCheckVerdict:
        """Determine verdict using LLM-based reasoning."""
        if not evidence:
            return FactCheckVerdict(
                label="NOT_ENOUGH_INFO",
                confidence=0.0,
                used_evidence_ids=[],
                reasoning="No evidence available to evaluate the claim.",
            )

        # Build evidence context
        evidence_texts = "\n".join(
            [f"[{e.source}] {e.text}" for e in evidence[:10]]  # Limit to top 10
        )
        used_ids = [e.id for e in evidence[:10]]

        # Create prompt for LLM with explicit refutation guidance
        prompt = f"""You are a fact-checking assistant. Your job is to determine if the evidence SUPPORTS, REFUTES, or provides NOT_ENOUGH_INFO about the claim.

Claim: {claim.normalized_text or claim.raw_text}

Evidence:
{evidence_texts}

CRITICAL INSTRUCTIONS:
- If ANY evidence clearly shows the claim is FALSE or INCORRECT, you MUST return "REFUTES"
- Only return "SUPPORTS" if evidence clearly confirms the claim is true
- Return "NOT_ENOUGH_INFO" only if evidence is truly insufficient, ambiguous, or doesn't directly address the claim
- Be decisive: if evidence contradicts the claim, return REFUTES even if confidence is moderate

Decision criteria:
- "SUPPORTS": Evidence clearly confirms the claim is true
- "REFUTES": Evidence clearly shows the claim is false, incorrect, or misleading
- "NOT_ENOUGH_INFO": Evidence is insufficient, ambiguous, or doesn't directly address the claim

Respond with ONLY a JSON object:
{{
    "label": "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO",
    "confidence": 0.0-1.0,
    "reasoning": "Explain your decision, especially if refuting. Cite specific evidence."
}}

Be objective and decisive. Base your decision strictly on the evidence provided."""

        try:
            response = await self.llm.complete(prompt, temperature=0.2, max_tokens=512)
            
            # Try to parse JSON response
            import json
            import re
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                verdict_data = json.loads(json_match.group())
            else:
                # Fallback: try parsing entire response
                verdict_data = json.loads(response.strip())
            
            label = verdict_data.get("label", "NOT_ENOUGH_INFO")
            confidence = float(verdict_data.get("confidence", 0.5))
            reasoning = verdict_data.get("reasoning", "LLM-based reasoning")
            
            # Validate label
            if label not in ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]:
                label = "NOT_ENOUGH_INFO"
            
            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))
            
            # Adjust confidence thresholds based on verdict type (Priority 2.2)
            if label == "SUPPORTS":
                # For supports, require at least moderate confidence
                if confidence < 0.5:
                    label = "NOT_ENOUGH_INFO"
                    confidence = 0.4
            elif label == "REFUTES":
                # For refutations, we want reasonable confidence (but not too strict)
                # If confidence is very low (< 0.4), might be uncertain
                if confidence < 0.4:
                    # Keep REFUTES but note uncertainty in reasoning
                    reasoning = f"Low confidence refutation: {reasoning}"
            # NOT_ENOUGH_INFO is fine as-is
            
            return FactCheckVerdict(
                label=label,
                confidence=confidence,
                used_evidence_ids=used_ids,
                reasoning=reasoning,
            )
        except Exception as e:
            # Fallback to simple heuristic if LLM fails
            return FactCheckVerdict(
                label="NOT_ENOUGH_INFO",
                confidence=0.3,
                used_evidence_ids=used_ids,
                reasoning=f"Error during LLM reasoning: {str(e)}. Using fallback.",
            )

