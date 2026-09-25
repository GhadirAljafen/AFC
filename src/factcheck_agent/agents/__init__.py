"""Agent implementations for the fact-check agent."""

from factcheck_agent.agents.claim_understanding import ClaimUnderstandingAgent
from factcheck_agent.agents.evidence_evaluation import (
    EvidenceEvaluationAgent,
    EvidenceEvaluationResult,
)
from factcheck_agent.agents.evidence_selection import EvidenceSelectionAgent
from factcheck_agent.agents.explanation import ExplanationAgent
from factcheck_agent.agents.reasoning_and_verdict import ReasoningAndVerdictAgent
from factcheck_agent.agents.retrieval import RetrievalAgent

# LangChain version (optional, requires langchain packages)
try:
    from factcheck_agent.agents.claim_understanding_langchain import (
        ClaimUnderstandingAgentLangChain,
    )

    __all__ = [
        "ClaimUnderstandingAgent",
        "ClaimUnderstandingAgentLangChain",
        "EvidenceEvaluationAgent",
        "EvidenceEvaluationResult",
        "EvidenceSelectionAgent",
        "ExplanationAgent",
        "ReasoningAndVerdictAgent",
        "RetrievalAgent",
    ]
except ImportError:
    __all__ = [
        "ClaimUnderstandingAgent",
        "EvidenceEvaluationAgent",
        "EvidenceEvaluationResult",
        "EvidenceSelectionAgent",
        "ExplanationAgent",
        "ReasoningAndVerdictAgent",
        "RetrievalAgent",
    ]
