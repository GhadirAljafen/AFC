"""Reasoning and verdict agent using LLM for fact-checking decisions.

INSTRUMENTATION NOTE (Stage 3, Phase A)
---------------------------------------
This module is instrumented but **behaviourally unchanged**: the prompt, the JSON
parsing, the label validation and the confidence-threshold rules all produce
exactly the same verdicts as before. What is new is observability.

Why it was needed. A single blanket `except Exception` previously wrapped the LLM
call, the parsing and the threshold logic, so an API timeout, a malformed JSON
reply and genuine epistemic uncertainty were **indistinguishable in the output** —
all three emitted `NOT_ENOUGH_INFO`. An evaluation run could not report its own
parse-failure rate, and measured over-abstention could not be separated from
silent infrastructure failure.

The threshold rules are also now counted, because saved evaluation data shows the
stage emits ~0.91 mean confidence while being ~0.55 accurate. Those rules fire
below 0.5 / 0.4, so they may almost never trigger — which would rule them out as
the cause of the SUPPORTS deficit. Counting settles it.

`decision_log` records the model's RAW label and confidence alongside the final
ones, because a demoted SUPPORTS has its confidence overwritten with 0.4,
destroying the number a calibration analysis needs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from factcheck_agent.llm_client import LLMClient
from factcheck_agent.models import Claim, EvidenceSnippet, FactCheckVerdict


logger = logging.getLogger(__name__)

VALID_LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
PROMPT_VARIANTS = ("original", "symmetric")

#: How the verdict is reached.
#:   "direct"    — one call: evidence straight to a label (the original behaviour)
#:   "synthesis" — two calls: first derive what the evidence establishes about the
#:                 claim, then judge using that synthesis alongside the evidence.
#:
#: Motivation is measured, not speculative. Supplying the fact-checker's own
#: justification — which is exactly this inferential bridge — raised accuracy by
#: +0.132 (p=0.0039) and roughly halved abstention, even on justifications that
#: never named the verdict. See `results/verdict/FINDINGS.md` §S.2-S.4. The QA
#: evidence carries facts but not the link from facts to verdict; synthesis tries
#: to derive that link from the evidence alone, with no gold data at inference.
REASONING_MODES = ("direct", "synthesis")

# Thresholds from IMPROVEMENT_PLAN Priority 2.2, preserved verbatim.
SUPPORTS_MIN_CONFIDENCE = 0.5
REFUTES_LOW_CONFIDENCE = 0.4
DEMOTED_CONFIDENCE = 0.4


class ReasoningAndVerdictAgent:
    """Decide on a verdict given a claim and supporting evidence."""

    def __init__(
        self,
        llm: LLMClient,
        record_decisions: bool = False,
        prompt_variant: str = "original",
        reasoning_mode: str = "direct",
    ):
        """Initialize the reasoning agent with an LLM client.

        Args:
            llm: LLM client.
            record_decisions: Append a per-decision diagnostic dict to
                `decision_log`. Off by default so production runs do not grow a
                list; evaluation harnesses turn it on.
            prompt_variant: `"original"` (default) or `"symmetric"`. Kept
                switchable so the two can be compared paired on identical claims
                and evidence. The default stays `"original"` until measurement
                justifies changing it — see `results/verdict/FINDINGS.md`.
        """
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"prompt_variant must be one of {PROMPT_VARIANTS}, got {prompt_variant!r}"
            )
        if reasoning_mode not in REASONING_MODES:
            raise ValueError(
                f"reasoning_mode must be one of {REASONING_MODES}, got {reasoning_mode!r}"
            )
        self.llm = llm
        self.record_decisions = record_decisions
        self.prompt_variant = prompt_variant
        self.reasoning_mode = reasoning_mode
        #: Synthesis calls made, and how many failed (cost accounting).
        self.synthesis_calls = 0
        self.synthesis_failures = 0

        # Observability counters. None of these affect verdicts.
        self.decisions = 0
        self.empty_evidence = 0
        self.llm_errors = 0
        self.parse_failures = 0
        self.label_coerced = 0
        self.missing_confidence = 0
        self.supports_demoted = 0
        self.refutes_low_confidence = 0
        self.decision_log: List[Dict[str, Any]] = []

    def stats(self) -> Dict[str, Any]:
        """Counter snapshot, for evaluation reporting."""
        total = max(1, self.decisions)
        return {
            "decisions": self.decisions,
            "empty_evidence": self.empty_evidence,
            "llm_errors": self.llm_errors,
            "parse_failures": self.parse_failures,
            "label_coerced": self.label_coerced,
            "missing_confidence": self.missing_confidence,
            "supports_demoted": self.supports_demoted,
            "refutes_low_confidence": self.refutes_low_confidence,
            "supports_demoted_rate": self.supports_demoted / total,
            "parse_failure_rate": self.parse_failures / total,
            "synthesis_calls": self.synthesis_calls,
            "synthesis_failures": self.synthesis_failures,
        }

    async def decide(self, claim: Claim, evidence: list[EvidenceSnippet]) -> FactCheckVerdict:
        """Determine verdict using LLM-based reasoning."""
        self.decisions += 1

        if not evidence:
            self.empty_evidence += 1
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

        # --- optional stage 0: derive what the evidence establishes -----------
        synthesis: Optional[str] = None
        if self.reasoning_mode == "synthesis":
            synthesis = await self._synthesize(claim, evidence_texts)

        prompt = self._build_prompt(claim, evidence_texts, synthesis=synthesis)

        # --- stage 1: the LLM call -------------------------------------------
        try:
            response = await self.llm.complete(prompt, temperature=0.2, max_tokens=512)
        except Exception as e:
            self.llm_errors += 1
            logger.error("verdict_llm_failed: %s: %s", type(e).__name__, e)
            return self._error_verdict(used_ids, e)

        # --- stage 2: parsing and field extraction ---------------------------
        try:
            verdict_data = self._parse_response(response)
            raw_label = verdict_data.get("label", "NOT_ENOUGH_INFO")
            if "confidence" not in verdict_data:
                self.missing_confidence += 1
            raw_confidence = float(verdict_data.get("confidence", 0.5))
            reasoning = verdict_data.get("reasoning", "LLM-based reasoning")
        except Exception as e:
            self.parse_failures += 1
            logger.error(
                "verdict_parse_failed: %s: %s | raw response (truncated):\n%s",
                type(e).__name__, e, response[:500],
            )
            return self._error_verdict(used_ids, e)

        # --- stage 3: validation and thresholds (unchanged semantics) --------
        label = raw_label
        if label not in VALID_LABELS:
            self.label_coerced += 1
            logger.warning("verdict_label_coerced: %r -> NOT_ENOUGH_INFO", raw_label)
            label = "NOT_ENOUGH_INFO"

        confidence = max(0.0, min(1.0, raw_confidence))

        rule_fired: Optional[str] = None
        if label == "SUPPORTS":
            if confidence < SUPPORTS_MIN_CONFIDENCE:
                self.supports_demoted += 1
                rule_fired = "supports_demoted"
                logger.info(
                    "verdict_supports_demoted: confidence %.3f < %.2f -> NOT_ENOUGH_INFO",
                    confidence, SUPPORTS_MIN_CONFIDENCE,
                )
                label = "NOT_ENOUGH_INFO"
                confidence = DEMOTED_CONFIDENCE
        elif label == "REFUTES":
            if confidence < REFUTES_LOW_CONFIDENCE:
                self.refutes_low_confidence += 1
                rule_fired = "refutes_low_confidence"
                reasoning = f"Low confidence refutation: {reasoning}"
        # NOT_ENOUGH_INFO is fine as-is

        if self.record_decisions:
            self.decision_log.append({
                "claim_id": claim.id,
                "raw_label": raw_label,
                "raw_confidence": raw_confidence,
                "final_label": label,
                "final_confidence": confidence,
                "rule_fired": rule_fired,
                "n_evidence": len(evidence),
                "synthesis_chars": len(synthesis) if synthesis else 0,
            })

        return FactCheckVerdict(
            label=label,
            confidence=confidence,
            used_evidence_ids=used_ids,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------

    async def _synthesize(self, claim: Claim, evidence_texts: str) -> Optional[str]:
        """Derive, from the evidence alone, what it establishes about the claim.

        Deliberately does NOT ask for a verdict. If it did, this would just be the
        verdict call twice. The point is to produce the inferential bridge that the
        measured experiment showed the QA evidence lacks — what the evidence
        establishes, and what it leaves open — and then let the verdict step judge
        on that.

        Kept neutral on purpose: it asks equally for what supports and what
        contradicts, so this does not smuggle in the bias that Phase B1 removed.

        Returns None on failure, in which case the verdict proceeds exactly as in
        `direct` mode — synthesis can only add, never break.
        """
        prompt = f"""You are analysing evidence for a fact-checking system. Do NOT give a verdict.

Claim: {claim.normalized_text or claim.raw_text}

Evidence:
{evidence_texts}

Write a short analysis covering, in plain prose:
1. What the evidence establishes that is relevant to the claim.
2. Which specific parts of the claim the evidence confirms, and which it contradicts.
3. What the claim asserts that the evidence does not address at all.

Rules:
- Reason only from the evidence above. Do not use outside knowledge.
- Give equal attention to confirming and contradicting details.
- State connections the evidence implies, even where it does not say them outright.
- Do NOT state a verdict or use the words SUPPORTS, REFUTES or NOT_ENOUGH_INFO.

Analysis:"""
        self.synthesis_calls += 1
        try:
            result = await self.llm.complete(prompt, temperature=0.2, max_tokens=400)
            return (result or "").strip() or None
        except Exception as e:
            self.synthesis_failures += 1
            logger.warning("verdict_synthesis_failed (falling back to direct): %s", e)
            return None

    def _build_prompt(self, claim: Claim, evidence_texts: str,
                      synthesis: Optional[str] = None) -> str:
        """Dispatch to the configured prompt variant, optionally with a synthesis."""
        if self.prompt_variant == "symmetric":
            prompt = self._prompt_symmetric(claim, evidence_texts)
        else:
            prompt = self._prompt_original(claim, evidence_texts)

        if synthesis:
            # Inserted before the JSON schema so the schema stays the final
            # instruction, which keeps output-format compliance unchanged.
            marker = "Respond with ONLY a JSON object:"
            block = (
                "Analysis of the evidence (derived from the evidence above, "
                "not from any external source):\n"
                f"{synthesis}\n\n"
            )
            if marker in prompt:
                prompt = prompt.replace(marker, block + marker, 1)
            else:  # pragma: no cover - both variants contain the marker
                prompt = f"{prompt}\n\n{block}"
        return prompt

    def _prompt_symmetric(self, claim: Claim, evidence_texts: str) -> str:
        """Phase B1: equal evidential bars for SUPPORTS and REFUTES.

        Three changes, each targeting a measured finding in
        `results/verdict/FINDINGS.md`:

        1. **Symmetric bars.** The original made REFUTES reachable on "ANY"
           evidence and a "MUST", while gating SUPPORTS behind "Only" and
           "clearly confirms". All 14 missed gold-SUPPORTS claims were the model
           declining to confirm, so the bars are now stated once, identically,
           with an explicit instruction not to demand more to confirm than to
           refute.
        2. **Removed the one-sided nudge.** "Be decisive: ... return REFUTES even
           if confidence is moderate" had no SUPPORTS counterpart.
        3. **Narrowed NOT_ENOUGH_INFO.** 33 of 37 errors (89%) were unwarranted
           abstention on gold evidence, so abstention is scoped to evidence that
           does not bear on the claim — explicitly not to evidence that is merely
           indirect or requires inference.

        Deliberately NOT done: adding any pro-SUPPORTS nudge. That would swap one
        bias for another rather than removing it.
        """
        return f"""You are a fact-checking assistant. Decide whether the evidence SUPPORTS the claim, REFUTES the claim, or is insufficient to judge it (NOT_ENOUGH_INFO).

Claim: {claim.normalized_text or claim.raw_text}

Evidence:
{evidence_texts}

Apply the SAME standard in both directions:
- "SUPPORTS": taken together, the evidence shows the claim is true
- "REFUTES": taken together, the evidence shows the claim is false, incorrect or misleading
- "NOT_ENOUGH_INFO": the evidence does not bear on the claim, or is genuinely balanced between true and false

Important:
- Do not require stronger evidence to confirm a claim than to refute it. Weigh confirming and contradicting evidence by the same standard.
- Evidence that needs a reasonable inference still counts. Reserve NOT_ENOUGH_INFO for evidence that genuinely does not address the claim — not for evidence that is indirect, partial, or merely implies the answer.
- If the evidence points one way on balance, say so. Use NOT_ENOUGH_INFO when it points nowhere, not when judging requires thought.

Respond with ONLY a JSON object:
{{
    "label": "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO",
    "confidence": 0.0-1.0,
    "reasoning": "Explain your decision, citing the specific evidence you relied on."
}}

Base your decision strictly on the evidence provided."""

    def _prompt_original(self, claim: Claim, evidence_texts: str) -> str:
        """The pre-existing prompt, kept verbatim as the measurement baseline.

        Note the asymmetry this variant exists to isolate: REFUTES is reachable
        on "ANY" evidence and is a "MUST", while SUPPORTS requires evidence that
        "clearly confirms" and is gated by "Only".
        """
        return f"""You are a fact-checking assistant. Your job is to determine if the evidence SUPPORTS, REFUTES, or provides NOT_ENOUGH_INFO about the claim.

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

    @staticmethod
    def _parse_response(response: str) -> Dict[str, Any]:
        """Parse the verdict JSON.

        DELIBERATELY UNCHANGED for Phase A, including its known defects: the
        `[^}]+` character class cannot match a nested object and stops at the
        first `}`, so a `}` inside `reasoning` truncates the match. Replacing it
        would fix verdicts that currently fail to parse, which is a behaviour
        change and therefore belongs in Phase B, measured. The counters above
        first establish how often it actually fails.
        """
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(response.strip())

    @staticmethod
    def _error_verdict(used_ids: List[str], error: Exception) -> FactCheckVerdict:
        """The pre-existing fallback verdict, byte-identical to before.

        Note for Phase B: this is indistinguishable in the output from a genuine
        "insufficient evidence" judgement, and 0.3 is a magic constant unrelated
        to the model's own confidence scale.
        """
        return FactCheckVerdict(
            label="NOT_ENOUGH_INFO",
            confidence=0.3,
            used_evidence_ids=used_ids,
            reasoning=f"Error during LLM reasoning: {str(error)}. Using fallback.",
        )
