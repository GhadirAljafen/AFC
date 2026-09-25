"""Unit tests for factcheck_agent models."""

from factcheck_agent.models import Claim, EvidenceSnippet, FactCheckResult, FactCheckVerdict


def test_claim_defaults():
    claim = Claim(id="c1", raw_text="The earth orbits the sun.")
    assert claim.id == "c1"
    assert claim.normalized_text == "The earth orbits the sun."
    assert claim.metadata is None


def test_evidence_snippet_str():
    snippet = EvidenceSnippet(
        id="e1",
        source="wiki",
        text="The Earth orbits the Sun once every 365.25 days.",
        score=0.9,
    )
    rendered = str(snippet)
    assert "[wiki]" in rendered
    assert "Earth orbits the Sun" in rendered


def test_factcheck_verdict_and_result():
    claim = Claim(id="c1", raw_text="The earth orbits the sun.")
    evidence = EvidenceSnippet(
        id="e1",
        source="wiki",
        text="The Earth orbits the Sun once every 365.25 days.",
        score=0.9,
    )
    verdict = FactCheckVerdict(
        label="SUPPORTS",
        confidence=0.88,
        used_evidence_ids=[evidence.id],
        reasoning="Astronomical consensus supports the claim.",
    )
    result = FactCheckResult(
        claim=claim,
        verdict=verdict,
        explanation="The claim is supported by astronomical evidence.",
        evidence=[evidence],
    )

    assert result.verdict.label == "SUPPORTS"
    assert result.verdict.confidence == 0.88
    assert result.evidence[0].id == "e1"
    summary = result.summary()
    assert summary["verdict"] == "SUPPORTS"
    assert summary["claim_id"] == "c1"

