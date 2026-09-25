"""Evidence evaluation agent — judges sufficiency and names what is missing."""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from factcheck_agent.llm_client import LLMClient
from factcheck_agent.models import Claim, EvidenceSnippet


logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block in `text`, or None.

    Replaces a ``\\{[^}]+\\}`` regex that could not match an object containing a
    nested brace and stopped at the first ``}``. This runs a brace counter that
    is aware of strings and escapes. It matters because the parsed result now
    controls whether the retrieval loop continues.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_evaluation_json(response: str) -> Optional[dict]:
    """Parse the evaluator's JSON reply. Returns None on failure (never raises)."""
    for candidate in (_extract_json_object(response), response.strip()):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


class EvidenceEvaluationResult(BaseModel):
    """Outcome of evaluating evidence sufficiency."""

    sufficient: bool = Field(..., description="Whether evidence sufficiently addresses the claim")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in sufficiency assessment")
    selected_evidence: List[EvidenceSnippet] = Field(
        ..., description="Evidence considered sufficient or most relevant"
    )
    notes: Optional[str] = Field(None, description="Additional notes about evaluation")
    gaps: Optional[str] = Field(
        None,
        description=(
            "What specific information is still missing. Drives the next retrieval "
            "round's queries; without it the loop repeats identical searches."
        ),
    )
    parse_failed: bool = Field(
        False, description="True when the LLM reply could not be parsed and a heuristic was used"
    )


class EvidenceEvaluationAgent:
    """Evaluate whether collected evidence is sufficient, and name the gaps."""

    def __init__(self, llm: Optional[LLMClient] = None, max_selected: int = 10):
        """Initialize the evaluation agent.

        Args:
            llm: Optional LLM client.
            max_selected: Cap on evidence returned as `selected_evidence`, which
                the pipeline uses as the final evidence set. This used to be
                hardcoded to 10, independently of `MAX_EVIDENCE_SOURCES` — so
                raising the retrieval cap above 10 silently had no effect on what
                reached the verdict. The pipeline now sizes this from its own
                settings so the two caps cannot disagree.
        """
        self.llm = llm
        self.max_selected = max_selected
        #: Count of replies that failed to parse, for reporting.
        self.parse_failures = 0

    async def evaluate(
        self, claim: Claim, evidence: List[EvidenceSnippet]
    ) -> EvidenceEvaluationResult:
        """Assess evidence sufficiency using the LLM if available, else heuristics."""
        if not evidence:
            return EvidenceEvaluationResult(
                sufficient=False,
                confidence=0.0,
                selected_evidence=[],
                notes="No evidence available.",
                gaps="No evidence was retrieved at all.",
            )

        if self.llm:
            try:
                return await self._evaluate_with_llm(claim, evidence)
            except Exception as e:
                logger.warning("Evidence evaluation LLM call failed (%s); using heuristic", e)

        return self._heuristic_result(evidence)

    # ---------- Internal helpers ----------

    async def _evaluate_with_llm(
        self, claim: Claim, evidence: List[EvidenceSnippet]
    ) -> EvidenceEvaluationResult:
        evidence_texts = "\n".join(
            f"[{e.id}] [{e.source}] {e.text[:200]}" for e in evidence[:15]
        )

        prompt = f"""You are evaluating whether the collected evidence is sufficient to fact-check a claim.

Claim: {claim.normalized_text or claim.raw_text}

Evidence collected ({len(evidence)} snippets):
{evidence_texts}

Respond with ONLY a JSON object:
{{
    "sufficient": true or false,
    "confidence": 0.0-1.0,
    "notes": "Brief explanation",
    "gaps": "If insufficient, state SPECIFICALLY what information is still missing (dates, figures, named sources, an official statement, etc). If sufficient, use an empty string."
}}

Is there enough relevant evidence to make a reliable fact-checking decision?"""

        response = await self.llm.complete(prompt, temperature=0.2, max_tokens=300)
        data = _parse_evaluation_json(response)

        if data is None:
            self.parse_failures += 1
            logger.error(
                "sufficiency_parse_failed: could not parse evaluator JSON. "
                "Raw response (truncated):\n%s",
                response[:500],
            )
            result = self._heuristic_result(evidence)
            result.parse_failed = True
            result.notes = "LLM response parsing failed; heuristic used"
            return result

        sufficient = bool(data.get("sufficient", len(evidence) >= 3))
        try:
            confidence = float(data.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))

        gaps = (data.get("gaps") or "").strip() or None
        if sufficient:
            gaps = None  # nothing to chase on the next round

        return EvidenceEvaluationResult(
            sufficient=sufficient,
            confidence=confidence,
            selected_evidence=self._top_evidence(evidence),
            notes=data.get("notes", "LLM-based evaluation"),
            gaps=gaps,
        )

    def _top_evidence(self, evidence: List[EvidenceSnippet],
                      k: Optional[int] = None) -> List[EvidenceSnippet]:
        """Highest-scoring evidence first. Scores are now populated by retrieval,
        so this is a real ordering rather than the no-op it used to be."""
        limit = k if k is not None else self.max_selected
        return sorted(evidence, key=lambda e: e.score if e.score is not None else 0.0,
                      reverse=True)[:limit]

    def _heuristic_result(self, evidence: List[EvidenceSnippet]) -> EvidenceEvaluationResult:
        sufficient = len(evidence) >= 3
        return EvidenceEvaluationResult(
            sufficient=sufficient,
            confidence=min(0.7, 0.3 + len(evidence) * 0.1),
            selected_evidence=self._top_evidence(evidence),
            notes=f"Heuristic evaluation: {len(evidence)} evidence snippet(s) collected.",
            gaps=None if sufficient else "Only a few sources were found; more corroboration needed.",
        )
