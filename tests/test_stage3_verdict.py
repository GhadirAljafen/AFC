"""
Stage 3 Phase A: prove the verdict-stage instrumentation is behaviourally inert.

Phase A added counters, structured logging and a decision log to
`ReasoningAndVerdictAgent`. The whole point is that verdicts are UNCHANGED — the
prompt, the JSON parsing (including its known defects), the label validation and
the confidence thresholds all behave exactly as before. These tests pin that
down, and check each counter fires on the path it claims to count.

Runs offline with scripted LLM doubles. No API calls, no network.

Run:  python tests/test_stage3_verdict.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factcheck_agent.agents.reasoning_and_verdict import ReasoningAndVerdictAgent
from factcheck_agent.models import Claim, EvidenceSnippet


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(label)


CLAIM = Claim(id="c1", raw_text="Saudi Arabia is the largest oil producer")
EVIDENCE = [
    EvidenceSnippet(id=f"https://e{i}.com", source="search", text=f"evidence body {i}", score=0.5)
    for i in range(3)
]


class Reply:
    """LLM double returning a fixed response (or raising)."""

    def __init__(self, response=None, raises=None):
        self.response, self.raises = response, raises

    async def complete(self, prompt, **kwargs):
        if self.raises:
            raise self.raises
        return self.response


# ------------------------------------------------------------------ inertness
async def test_normal_path_unchanged():
    llm = Reply('{"label": "REFUTES", "confidence": 0.86, "reasoning": "because X"}')
    agent = ReasoningAndVerdictAgent(llm=llm)
    v = await agent.decide(CLAIM, EVIDENCE)
    check("normal path returns the model's label", v.label == "REFUTES")
    check("normal path preserves the model's confidence", v.confidence == 0.86)
    check("normal path preserves reasoning", v.reasoning == "because X")
    check("used_evidence_ids is still the first 10 shown",
          v.used_evidence_ids == [e.id for e in EVIDENCE], f"{v.used_evidence_ids}")
    check("nothing miscounted",
          agent.low_confidence_seen == 0 and agent.parse_failures == 0 and agent.llm_errors == 0)


async def test_empty_evidence_unchanged():
    agent = ReasoningAndVerdictAgent(llm=Reply("unused"))
    v = await agent.decide(CLAIM, [])
    check("empty evidence -> NOT_ENOUGH_INFO at 0.0",
          v.label == "NOT_ENOUGH_INFO" and v.confidence == 0.0)
    check("empty evidence short-circuits before the LLM", agent.empty_evidence == 1)


async def test_llm_error_fallback_identical():
    err = RuntimeError("API timeout")
    agent = ReasoningAndVerdictAgent(llm=Reply(raises=err))
    v = await agent.decide(CLAIM, EVIDENCE)
    check("LLM error -> NOT_ENOUGH_INFO at 0.3 (unchanged)",
          v.label == "NOT_ENOUGH_INFO" and v.confidence == 0.3)
    check("LLM error fallback message is byte-identical",
          v.reasoning == "Error during LLM reasoning: API timeout. Using fallback.",
          repr(v.reasoning))
    check("LLM error is counted as an LLM error, not a parse failure",
          agent.llm_errors == 1 and agent.parse_failures == 0)


async def test_parse_failure_counted_separately():
    agent = ReasoningAndVerdictAgent(llm=Reply("this is not json at all"))
    v = await agent.decide(CLAIM, EVIDENCE)
    check("parse failure -> NOT_ENOUGH_INFO at 0.3 (unchanged)",
          v.label == "NOT_ENOUGH_INFO" and v.confidence == 0.3)
    check("parse failure is counted as a parse failure, not an LLM error",
          agent.parse_failures == 1 and agent.llm_errors == 0)
    check("the two failure causes are now distinguishable",
          agent.stats()["parse_failure_rate"] == 1.0)


async def test_known_regex_defect_preserved():
    """Phase A must NOT fix the parser — that is a Phase B behaviour change.

    The `[^}]+` class cannot match a nested object, so this reply still fails to
    parse and still falls back. Pinning it prevents an accidental silent fix.
    """
    nested = '{"label": "SUPPORTS", "confidence": 0.9, "meta": {"a": 1}, "reasoning": "r"}'
    agent = ReasoningAndVerdictAgent(llm=Reply(nested))
    v = await agent.decide(CLAIM, EVIDENCE)
    check("nested-brace reply still fails to parse (defect preserved)",
          agent.parse_failures == 1 and v.label == "NOT_ENOUGH_INFO",
          f"label={v.label} parse_failures={agent.parse_failures}")


# ------------------------------------------------------------------- counters
async def test_low_confidence_supports_is_no_longer_demoted():
    """The demotion rule was removed after measuring 0/150 firings.

    A low-confidence SUPPORTS must now survive as the model gave it, rather than
    being rewritten to NOT_ENOUGH_INFO with a magic 0.4 confidence.
    """
    llm = Reply('{"label": "SUPPORTS", "confidence": 0.31, "reasoning": "weak"}')
    agent = ReasoningAndVerdictAgent(llm=llm, record_decisions=True)
    v = await agent.decide(CLAIM, EVIDENCE)
    check("low-confidence SUPPORTS is PRESERVED, not demoted", v.label == "SUPPORTS")
    check("its confidence is not overwritten", v.confidence == 0.31)
    check("reasoning is untouched", v.reasoning == "weak")
    check("the unusual confidence is logged as observed", agent.low_confidence_seen == 1)
    entry = agent.decision_log[0]
    check("no rule rewrote the answer", entry["rule_fired"] is None, f"{entry}")
    check("decision log still records raw label/confidence",
          entry["raw_label"] == "SUPPORTS" and entry["raw_confidence"] == 0.31)


async def test_high_confidence_supports_survives():
    llm = Reply('{"label": "SUPPORTS", "confidence": 0.91, "reasoning": "strong"}')
    agent = ReasoningAndVerdictAgent(llm=llm)
    v = await agent.decide(CLAIM, EVIDENCE)
    check("SUPPORTS is passed through unchanged",
          v.label == "SUPPORTS" and v.confidence == 0.91)
    check("nothing flagged at normal confidence", agent.low_confidence_seen == 0)


async def test_low_confidence_refutes_no_longer_annotated():
    """The REFUTES side of the asymmetry is gone too.

    It used to prepend "Low confidence refutation:" to the reasoning, which then
    flowed into the explanation prompt. Both sides are now symmetric: neither
    label is rewritten or annotated.
    """
    llm = Reply('{"label": "REFUTES", "confidence": 0.22, "reasoning": "hmm"}')
    agent = ReasoningAndVerdictAgent(llm=llm)
    v = await agent.decide(CLAIM, EVIDENCE)
    check("low-confidence REFUTES is preserved", v.label == "REFUTES")
    check("reasoning is no longer prefixed", v.reasoning == "hmm", v.reasoning)
    check("the unusual confidence is logged", agent.low_confidence_seen == 1)


async def test_label_coercion_counted():
    llm = Reply('{"label": "CONFLICTING", "confidence": 0.8, "reasoning": "r"}')
    agent = ReasoningAndVerdictAgent(llm=llm)
    v = await agent.decide(CLAIM, EVIDENCE)
    check("off-menu label still coerced to NOT_ENOUGH_INFO", v.label == "NOT_ENOUGH_INFO")
    check("coercion is counted", agent.label_coerced == 1)


async def test_missing_confidence_counted():
    llm = Reply('{"label": "REFUTES", "reasoning": "no confidence field"}')
    agent = ReasoningAndVerdictAgent(llm=llm)
    v = await agent.decide(CLAIM, EVIDENCE)
    check("missing confidence still defaults to 0.5", v.confidence == 0.5)
    check("the silent default is counted", agent.missing_confidence == 1)


async def test_decision_log_off_by_default():
    llm = Reply('{"label": "REFUTES", "confidence": 0.8, "reasoning": "r"}')
    agent = ReasoningAndVerdictAgent(llm=llm)
    await agent.decide(CLAIM, EVIDENCE)
    check("decision log is off unless asked for", agent.decision_log == [])


async def main():
    print("--- inertness: verdicts unchanged ---")
    await test_normal_path_unchanged()
    await test_empty_evidence_unchanged()
    await test_llm_error_fallback_identical()
    await test_parse_failure_counted_separately()
    await test_known_regex_defect_preserved()

    print("\n--- thresholds removed; observability retained ---")
    await test_low_confidence_supports_is_no_longer_demoted()
    await test_high_confidence_supports_survives()
    await test_low_confidence_refutes_no_longer_annotated()
    await test_label_coercion_counted()
    await test_missing_confidence_counted()
    await test_decision_log_off_by_default()

    print("\nAll Stage 3 Phase A checks passed (instrumentation is inert).")


if __name__ == "__main__":
    asyncio.run(main())
