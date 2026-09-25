"""Main fact-checking pipeline.

Two distinct loops live here, and they are orthogonal:

* **Loop A — article orchestration** (`process_text`): across *claims*. Detects
  check-worthy claims in a document and verifies each one. Supports the claim
  "the system is wired end-to-end".
* **Loop B — retrieval refinement** (`process`): across *retrieval rounds for a
  single claim*. Feeds the evidence evaluator's stated gaps back into query
  generation so a later round searches for something new. Supports the claim
  "the retrieval loop is real".

Loop B previously could not work: it called `retrieve_evidence(claim)` with an
identical argument every round, so retrieval's URL dedup guaranteed round 2 added
nothing — after spending an LLM call and up to 15 searches whose results were all
discarded.

Search spend is the binding constraint on the connected system, since cost is
(claims per article) x (queries per claim) x (retrieval rounds). Both loops are
therefore budgeted.
"""

from typing import List, Optional, Set

from factcheck_agent.config import get_config
from factcheck_agent.content_fetch import ContentFetcher
from factcheck_agent.models import (
    ArticleFactCheckResult,
    Claim,
    ClaimCheckFailure,
    EvidenceSnippet,
    FactCheckResult,
)
from factcheck_agent.llm_client import LLMClient
from factcheck_agent.agents.claim_understanding import ClaimUnderstandingAgent
from factcheck_agent.agents.retrieval import RetrievalAgent
from factcheck_agent.agents.evidence_selection import EvidenceSelectionAgent
from factcheck_agent.agents.evidence_evaluation import EvidenceEvaluationAgent
from factcheck_agent.agents.reasoning_and_verdict import ReasoningAndVerdictAgent
from factcheck_agent.agents.explanation import ExplanationAgent


class QueryBudget:
    """Tracks search spend across a document so one article cannot drain the quota."""

    def __init__(self, limit: Optional[int] = None):
        self.limit = limit
        self.spent = 0

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.spent >= self.limit

    def remaining(self) -> Optional[int]:
        return None if self.limit is None else max(0, self.limit - self.spent)


class FactCheckingPipeline:
    """Main pipeline for fact-checking claims using an agentic retrieval-evaluation loop."""

    def __init__(
        self,
        llm: LLMClient,
        retrieval_agent: Optional[RetrievalAgent] = None,
        max_iterations: Optional[int] = None,
        initial_retrieval_k: Optional[int] = None,
        use_llm_normalization: bool = True,
        content_fetcher: Optional[ContentFetcher] = None,
        full_text_top_k: Optional[int] = None,
    ):
        """Initialize the fact-checking pipeline with agents.

        Args:
            llm: LLM client for agents that need it.
            retrieval_agent: Optional retrieval agent. Defaults to one built from config.
            max_iterations: Retrieval-evaluation rounds. Defaults to MAX_RETRIEVAL_LOOPS.
            initial_retrieval_k: Evidence snippets kept per round. Defaults to
                MAX_EVIDENCE_SOURCES.
            use_llm_normalization: Use the LLM normalizer rather than the rule-based
                one. The LLM path decontextualizes and standardizes entities, which
                materially improves the search queries generated downstream.
        """
        config = get_config()

        self.llm = llm
        self.max_iterations = (
            max_iterations if max_iterations is not None else config.MAX_RETRIEVAL_LOOPS
        )
        self.initial_retrieval_k = (
            initial_retrieval_k if initial_retrieval_k is not None else config.MAX_EVIDENCE_SOURCES
        )
        self.use_llm_normalization = use_llm_normalization
        self.max_gap_queries = config.MAX_GAP_QUERIES
        self.max_claims_per_article = config.MAX_CLAIMS_PER_ARTICLE
        self.query_budget_per_article = config.QUERY_BUDGET_PER_ARTICLE
        self.full_text_top_k = (
            full_text_top_k if full_text_top_k is not None else config.FULL_TEXT_TOP_K
        )
        # Set FULL_TEXT_TOP_K=0 to stay on search snippets only.
        self.content_fetcher = content_fetcher or (
            ContentFetcher(max_chars=config.MAX_SOURCE_CHARS) if self.full_text_top_k else None
        )

        self.claim_understanding = ClaimUnderstandingAgent(llm=llm)
        self.retrieval = retrieval_agent or RetrievalAgent(llm=llm)
        self.evidence_selection = EvidenceSelectionAgent()
        # Size the evaluator's cap from our own settings, so raising
        # MAX_EVIDENCE_SOURCES above 10 actually reaches the verdict stage.
        self.evidence_evaluation = EvidenceEvaluationAgent(
            llm=llm,
            max_selected=max(10, self.initial_retrieval_k * max(1, self.max_iterations)),
        )
        self.reasoning_and_verdict = ReasoningAndVerdictAgent(llm=llm)
        self.explanation = ExplanationAgent(llm=llm)

    # ------------------------------------------------------------------
    # Loop A — article orchestration (across claims)
    # ------------------------------------------------------------------

    async def process_text(
        self,
        text: str,
        min_importance: float = 0.5,
        max_claims: Optional[int] = None,
    ) -> ArticleFactCheckResult:
        """Detect check-worthy claims in a document and fact-check each one.

        Per-claim failures are isolated: one bad claim does not abort the document.

        Args:
            text: The document to process.
            min_importance: Minimum check-worthiness for a claim to be verified.
            max_claims: Cap on claims verified, highest importance first. Defaults
                to MAX_CLAIMS_PER_ARTICLE. This is a quota control as much as a
                scope control.

        Returns:
            ArticleFactCheckResult — per-claim results, with no rolled-up verdict.
        """
        cap = max_claims if max_claims is not None else self.max_claims_per_article
        budget = QueryBudget(self.query_budget_per_article)

        detected = await self.claim_understanding.detect_claims(
            text, min_importance=min_importance, normalize_detected=False
        )

        article = ArticleFactCheckResult(detected_claim_count=len(detected))
        selected = detected[:cap] if cap else detected
        article.checked_claim_count = len(selected)

        for detected_claim in selected:
            if budget.exhausted:
                article.budget_exhausted = True
                article.failures.append(ClaimCheckFailure(
                    claim_id=detected_claim.id,
                    claim_text=detected_claim.raw_text,
                    error=f"Per-article search budget ({budget.limit}) exhausted before this claim.",
                ))
                continue

            claim = Claim(
                id=detected_claim.id,
                raw_text=detected_claim.raw_text,
                metadata={
                    k: v for k, v in {
                        **(detected_claim.metadata or {}),
                        "sentence_index": (
                            str(detected_claim.sentence_index)
                            if detected_claim.sentence_index is not None else None
                        ),
                        "importance": (
                            str(detected_claim.importance)
                            if detected_claim.importance is not None else None
                        ),
                    }.items() if v is not None
                },
            )

            try:
                article.results.append(await self.process(claim, budget=budget))
            except Exception as e:
                article.failures.append(ClaimCheckFailure(
                    claim_id=claim.id, claim_text=claim.raw_text, error=str(e)
                ))

        article.searches_issued = budget.spent
        article.budget_exhausted = article.budget_exhausted or budget.exhausted
        return article

    # ------------------------------------------------------------------
    # Loop B — retrieval refinement (across rounds, for one claim)
    # ------------------------------------------------------------------

    async def process(
        self, claim: Claim, budget: Optional[QueryBudget] = None
    ) -> FactCheckResult:
        """Fact-check a single claim.

        Stages:
          1. Normalize the claim
          2. Retrieval-evaluation loop — each round targets the gaps the evaluator
             identified and excludes URLs already collected, so later rounds search
             for genuinely new evidence
          3. Verdict
          4. Explanation

        Args:
            claim: The claim to fact-check.
            budget: Optional shared search budget (used when called from process_text).

        Returns:
            FactCheckResult with verdict and explanation.
        """
        normalized_claim = await self._normalize(claim)
        all_evidence, evaluation_result, _rounds = await self.collect_evidence(
            normalized_claim, budget=budget
        )

        final_evidence = (
            evaluation_result.selected_evidence if evaluation_result else all_evidence
        )

        # Replace snippets with article passages for the evidence that will actually
        # be reasoned over. Done once, after the loop, rather than per round — and
        # only for the top-k, since most news domains block automated fetching.
        if self.content_fetcher and self.full_text_top_k and final_evidence:
            final_evidence = await self.content_fetcher.enrich(
                final_evidence,
                normalized_claim.normalized_text or normalized_claim.raw_text,
                top_k=self.full_text_top_k,
            )

        verdict = await self.reasoning_and_verdict.decide(normalized_claim, final_evidence)
        explanation = await self.explanation.generate(
            normalized_claim, verdict, final_evidence
        )

        return FactCheckResult(
            claim=normalized_claim,
            verdict=verdict,
            explanation=explanation,
            evidence=final_evidence,
        )

    async def collect_evidence(self, normalized_claim: Claim, budget: Optional[QueryBudget] = None):
        """Run the retrieval-evaluation loop for an already-normalized claim.

        Split out from `process` so Stage 2 evaluation can measure the real loop
        without paying for verdict and explanation generation.

        Returns:
            (evidence, last_evaluation_result, new_evidence_count_per_round)
        """
        all_evidence: List[EvidenceSnippet] = []
        seen_urls: Set[str] = set()
        evaluation_result = None
        gaps: Optional[str] = None
        per_round: List[int] = []

        for iteration in range(self.max_iterations):
            if budget is not None and budget.exhausted:
                break

            # Later rounds are gap-driven and get a smaller query allowance, because
            # search spend multiplies across rounds.
            round_queries = None if iteration == 0 else self.max_gap_queries
            if budget is not None and budget.remaining() is not None:
                allowance = budget.remaining()
                round_queries = min(round_queries or self.retrieval.max_queries, allowance)
                if round_queries <= 0:
                    break

            before = self.retrieval.searches_issued
            candidates = await self.retrieval.retrieve_evidence(
                normalized_claim,
                gaps=gaps,
                exclude_urls=seen_urls,
                max_queries=round_queries,
            )
            if budget is not None:
                budget.spent += self.retrieval.searches_issued - before

            selected = self.evidence_selection.select(
                normalized_claim, candidates, k=self.initial_retrieval_k
            )

            new_evidence = [e for e in selected if e.id not in seen_urls]
            all_evidence.extend(new_evidence)
            seen_urls.update(e.id for e in new_evidence)
            per_round.append(len(new_evidence))

            evaluation_result = await self.evidence_evaluation.evaluate(
                normalized_claim, all_evidence
            )

            if evaluation_result.sufficient:
                break

            # Stop condition: a round that surfaced no new sources cannot be improved
            # on by repeating it, and without a stated gap the next round would just
            # re-issue the same queries.
            if not new_evidence or not evaluation_result.gaps:
                break

            gaps = evaluation_result.gaps

        return all_evidence, evaluation_result, per_round

    # ------------------------------------------------------------------

    async def _normalize(self, claim: Claim) -> Claim:
        """Normalize a claim, preferring the LLM path for better query generation."""
        if self.use_llm_normalization and self.llm is not None:
            try:
                return await self.claim_understanding.normalize_claim_llm(claim)
            except Exception:
                pass
        return self.claim_understanding.normalize_claim(claim)
