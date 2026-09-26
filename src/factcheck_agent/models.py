"""Pydantic models for the fact-check agent."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Claim(BaseModel):
    """A user-submitted claim to be fact-checked."""

    id: str = Field(..., description="Unique identifier for the claim")
    raw_text: str = Field(..., description="Original claim text as provided")
    normalized_text: Optional[str] = Field(
        None, description="Normalized/cleaned version of the claim text"
    )
    metadata: Optional[Dict[str, str]] = Field(
        None, description="Arbitrary metadata such as source or timestamps"
    )

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return self.normalized_text or self.raw_text

    @model_validator(mode="after")
    def ensure_normalized_default(self) -> "Claim":
        """Fallback to raw_text when normalized_text is absent."""
        if not self.normalized_text:
            self.normalized_text = self.raw_text
        return self


class DetectedClaim(BaseModel):
    """A claim detected from a longer text document."""

    id: str = Field(..., description="Unique identifier for the detected claim")
    raw_text: str = Field(..., description="The extracted claim text")
    normalized_text: Optional[str] = Field(
        None, description="Normalized version of the detected claim text"
    )
    start_char: Optional[int] = Field(
        None, description="Character position where the claim starts in the source text"
    )
    end_char: Optional[int] = Field(
        None, description="Character position where the claim ends in the source text"
    )
    sentence_index: Optional[int] = Field(
        None, description="Index of the sentence containing this claim (0-based)"
    )
    importance: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="How check-worthy this claim is, from 0.0 (low) to 1.0 (high)",
    )
    metadata: Optional[Dict[str, str]] = Field(
        None, description="Additional metadata about the detected claim"
    )

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        importance_str = f" (importance: {self.importance:.2f})" if self.importance else ""
        return f"{self.raw_text}{importance_str}"


class EvidenceSnippet(BaseModel):
    """A snippet of evidence supporting or refuting a claim."""

    id: str = Field(..., description="Unique identifier for the evidence snippet")
    source: str = Field(..., description='Source identifier (e.g., "wiki", "news_corpus")')
    text: str = Field(..., description="Content of the evidence snippet")
    score: Optional[float] = Field(
        None, description="Optional relevance score from retrieval/ranking"
    )
    metadata: Optional[Dict[str, str]] = Field(
        None, description="Additional metadata such as URL, publication date"
    )

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        preview = self.text[:80] + ("..." if len(self.text) > 80 else "")
        return f"[{self.source}] {preview}"


class FactCheckVerdict(BaseModel):
    """Verdict for a fact-checking decision."""

    label: Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"] = Field(
        ..., description="Classification of the claim relative to evidence"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in verdict")
    used_evidence_ids: List[str] = Field(
        ...,
        description=(
            "IDs of the evidence snippets PRESENTED to the model for this verdict. "
            "NOT a citation list: the model is not shown these IDs and does not "
            "report which snippets it relied on, so this cannot be read as "
            "attribution. Real citation grounding requires putting IDs in the "
            "prompt and asking for them back, which changes verdict behaviour and "
            "so needs its own measured comparison. Tracked as a Stage 4 "
            "prerequisite - see results/verdict/FINDINGS.md."
        ),
    )
    reasoning: Optional[str] = Field(None, description="Rationale for the verdict")

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.label} ({self.confidence:.2f})"


class FactCheckResult(BaseModel):
    """Full fact-checking result containing verdict and evidence."""

    claim: Claim = Field(..., description="The claim being fact-checked")
    verdict: FactCheckVerdict = Field(..., description="Verdict derived from evidence")
    explanation: str = Field(..., description="Natural-language explanation of the verdict")
    evidence: List[EvidenceSnippet] = Field(
        ..., description="Evidence snippets used to reach the verdict"
    )

    def summary(self) -> dict:
        """Return a serializable summary of the fact-check result."""
        return {
            "claim_id": self.claim.id,
            "verdict": self.verdict.label,
            "confidence": self.verdict.confidence,
            "evidence_ids": self.verdict.used_evidence_ids,
            "explanation": self.explanation,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.claim}: {self.verdict.label}"


class ClaimCheckFailure(BaseModel):
    """A claim that was detected but could not be fact-checked."""

    claim_id: str = Field(..., description="Identifier of the claim that failed")
    claim_text: str = Field(..., description="Text of the claim that failed")
    error: str = Field(..., description="Why the check failed")


class ArticleFactCheckResult(BaseModel):
    """Result of running the full pipeline over a document.

    Deliberately has NO article-level verdict. An article containing one true and
    one false claim is not a conflict to be resolved into a single label — the
    per-claim verdicts *are* the answer. Rolling them up would discard the
    system's actual output.
    """

    results: List[FactCheckResult] = Field(
        default_factory=list, description="One result per successfully checked claim"
    )
    failures: List[ClaimCheckFailure] = Field(
        default_factory=list, description="Claims detected but not checkable"
    )
    detected_claim_count: int = Field(
        0, description="Claims detected before any max_claims cap was applied"
    )
    checked_claim_count: int = Field(0, description="Claims actually submitted for checking")
    searches_issued: int = Field(0, description="Search queries spent on this document")
    budget_exhausted: bool = Field(
        False, description="True if the per-article query budget stopped the run early"
    )

    def counts_by_label(self) -> Dict[str, int]:
        """Verdict label -> number of claims, for reporting."""
        counts: Dict[str, int] = {}
        for result in self.results:
            counts[result.verdict.label] = counts.get(result.verdict.label, 0) + 1
        return counts

    def summary(self) -> dict:
        """Return a serializable summary of the document-level run."""
        return {
            "detected_claims": self.detected_claim_count,
            "checked_claims": self.checked_claim_count,
            "failed_claims": len(self.failures),
            "verdicts": self.counts_by_label(),
            "searches_issued": self.searches_issued,
            "budget_exhausted": self.budget_exhausted,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts_by_label().items()))
        return f"{self.checked_claim_count} claim(s) checked [{counts or 'none'}]"

