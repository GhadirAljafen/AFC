"""Evidence selection agent that prioritizes contradictory evidence."""

from typing import List

from factcheck_agent.models import Claim, EvidenceSnippet


# Keywords suggesting a snippet refutes rather than supports.
REFUTATION_KEYWORDS = (
    "false", "incorrect", "debunked", "misleading", "untrue",
    "not true", "disproven", "refuted", "contradicts", "disputes",
    "wrong", "inaccurate", "myth", "hoax", "fake",
)

# Maximum additive boost for refuting evidence, on the retrieval score's 0-1 scale.
#
# This was previously `+0.3 + 0.1 * matches` applied to a base that was ALWAYS the
# constant 0.5, because retrieval never set a score. Now that retrieval produces
# real relevance scores spanning 0-1, an unbounded boost would swamp them: a
# snippet matching 15 keywords would have scored +1.8. The boost is therefore
# capped and saturating, so refuting evidence is still promoted but cannot
# override a large relevance gap.
REFUTATION_BOOST = 0.25

# Keyword matches at which the boost reaches its maximum.
REFUTATION_SATURATION = 3


class EvidenceSelectionAgent:
    """Select a subset of evidence snippets, prioritizing contradictory evidence."""

    def select(
        self, claim: Claim, candidates: List[EvidenceSnippet], k: int = 5
    ) -> List[EvidenceSnippet]:
        """
        Select evidence, prioritizing contradictory snippets.

        Ranks by the retrieval relevance score, then applies a bounded boost to
        snippets containing refutation language so refuting evidence is not
        crowded out by supporting evidence.

        Args:
            claim: The claim being fact-checked
            candidates: Candidate evidence snippets
            k: Number of snippets to select

        Returns:
            Selected snippets, best first.
        """
        if not candidates:
            return []

        scored = []
        for snippet in candidates:
            base = snippet.score if snippet.score is not None else 0.5

            text_lower = snippet.text.lower()
            matches = sum(1 for keyword in REFUTATION_KEYWORDS if keyword in text_lower)
            boost = REFUTATION_BOOST * min(1.0, matches / REFUTATION_SATURATION)

            scored.append((min(1.0, base + boost), snippet))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [snippet for _, snippet in scored[:k]]
