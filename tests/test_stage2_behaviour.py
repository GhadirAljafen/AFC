"""
Behavioural tests for Stage 2 (retrieval) and the two pipeline loops.

Runs fully offline against a StaticSearchClient and scripted LLM doubles, so it
costs no search quota and no API calls. Covers verification steps 1-4 and 6-8
from the Stage 2 plan.

Run:  python tests/test_stage2_behaviour.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factcheck_agent.agents.evidence_selection import EvidenceSelectionAgent
from factcheck_agent.agents.retrieval import RetrievalAgent
from factcheck_agent.models import Claim, EvidenceSnippet
from factcheck_agent.pipeline import FactCheckingPipeline, QueryBudget
from factcheck_agent.search import SearchResult, StaticSearchClient

ROUND1_QUERY = "claim about oil production"
ROUND2_QUERY = "official OPEC production figures 2024"


def R(url, rank, title="", snippet="body text"):
    return SearchResult(url=url, title=title, snippet=snippet, rank=rank)


class ScriptedLLM:
    """LLM double that dispatches on prompt content.

    `sufficiency` is a list of (sufficient, gaps) consumed one per evaluation call,
    which is how a multi-round retrieval loop is driven deterministically.
    """

    def __init__(self, sufficiency=((True, ""),)):
        self.sufficiency = list(sufficiency)
        self.eval_calls = 0
        self.gap_prompts = []

    async def complete(self, prompt, **kwargs):
        if "Normalize the following claim" in prompt:
            return "Saudi Arabia is the world's largest oil producer"
        if "evidence is sufficient" in prompt:
            i = min(self.eval_calls, len(self.sufficiency) - 1)
            sufficient, gaps = self.sufficiency[i]
            self.eval_calls += 1
            return (
                '{"sufficient": %s, "confidence": 0.8, "notes": "n", "gaps": "%s"}'
                % ("true" if sufficient else "false", gaps)
            )
        if "SUPPORTS, REFUTES" in prompt:
            return '{"label": "SUPPORTS", "confidence": 0.8, "reasoning": "r"}'
        return "An explanation."

    async def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        if "already happened" in system:          # gap-driven round
            self.gap_prompts.append(messages[1]["content"])
            return ROUND2_QUERY
        return ROUND1_QUERY


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(label)


# ---------------------------------------------------------------- step 1, 3, 8
async def test_loop_iterates_with_new_queries():
    search = StaticSearchClient({
        ROUND1_QUERY: [R("https://a.com/1", 1), R("https://b.com/2", 2)],
        ROUND2_QUERY: [R("https://c.com/3", 1), R("https://d.com/4", 2)],
    })
    llm = ScriptedLLM(sufficiency=[(False, "need official production figures"), (True, "")])
    pipeline = FactCheckingPipeline(
        llm=llm,
        retrieval_agent=RetrievalAgent(llm=llm, search_client=search),
        max_iterations=3,
    )

    result = await pipeline.process(Claim(id="c1", raw_text="Saudi Arabia produces the most oil"))

    check("Loop B issues a DIFFERENT query on round 2",
          search.queries_seen[0] != search.queries_seen[-1],
          f"{search.queries_seen}")
    check("Loop B accumulates NEW urls from round 2",
          any("c.com" in e.id or "d.com" in e.id for e in result.evidence),
          f"{[e.id for e in result.evidence]}")
    check("gap text was passed into round-2 query generation",
          any("official production figures" in p for p in llm.gap_prompts))


async def test_no_second_round_when_sufficient():
    search = StaticSearchClient({ROUND1_QUERY: [R("https://a.com/1", 1)]})
    llm = ScriptedLLM(sufficiency=[(True, "")])
    pipeline = FactCheckingPipeline(
        llm=llm,
        retrieval_agent=RetrievalAgent(llm=llm, search_client=search),
        max_iterations=3,
    )
    await pipeline.process(Claim(id="c1", raw_text="x"))
    # Round 1 legitimately issues more than one search (the auto-appended
    # refutation query), so count ROUNDS via gap-driven generation, not searches.
    check("no second round when round 1 suffices",
          llm.gap_prompts == [], f"gap rounds={len(llm.gap_prompts)}")


async def test_stops_when_no_gaps_given():
    """Insufficient but no gaps -> the next round would repeat itself, so stop."""
    search = StaticSearchClient({ROUND1_QUERY: [R("https://a.com/1", 1)]})
    llm = ScriptedLLM(sufficiency=[(False, ""), (False, "")])
    pipeline = FactCheckingPipeline(
        llm=llm,
        retrieval_agent=RetrievalAgent(llm=llm, search_client=search),
        max_iterations=3,
    )
    await pipeline.process(Claim(id="c1", raw_text="x"))
    check("stop condition fires when evaluator names no gap",
          llm.gap_prompts == [], f"gap rounds={len(llm.gap_prompts)}")


async def test_max_iterations_honoured():
    search = StaticSearchClient(default=[R("https://a.com/1", 1)])
    llm = ScriptedLLM(sufficiency=[(False, "more"), (False, "more"), (False, "more")])
    pipeline = FactCheckingPipeline(
        llm=llm,
        retrieval_agent=RetrievalAgent(llm=llm, search_client=search),
        max_iterations=1,
    )
    await pipeline.process(Claim(id="c1", raw_text="x"))
    check("max_iterations=1 limits the loop to one round",
          llm.gap_prompts == [], f"gap rounds={len(llm.gap_prompts)}")


# ---------------------------------------------------------------------- step 4
async def test_scores_and_rrf():
    # shared.com is returned by BOTH queries; solo.com only by the first, at rank 1.
    search = StaticSearchClient({
        ROUND1_QUERY: [R("https://solo.com/x", 1), R("https://shared.com/y", 2)],
        "extra": [R("https://shared.com/y", 1)],
    })

    class TwoQueryLLM(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            return f"{ROUND1_QUERY}\nextra"

    llm = TwoQueryLLM()
    agent = RetrievalAgent(llm=llm, search_client=search)
    snippets = await agent.retrieve_evidence(Claim(id="c", raw_text="oil production"))

    check("every snippet has a populated score",
          all(s.score is not None for s in snippets))
    shared = next(s for s in snippets if "shared.com" in s.id)
    solo = next(s for s in snippets if "solo.com" in s.id)
    check("RRF boosts a URL returned by multiple queries above a single-query rank-1",
          shared.score > solo.score, f"shared={shared.score} solo={solo.score}")
    check("ranking is by score, best first",
          snippets == sorted(snippets, key=lambda s: s.score, reverse=True))


def test_refutation_boost_is_bounded():
    agent = EvidenceSelectionAgent()
    many_keywords = "false incorrect debunked misleading untrue hoax fake wrong myth"
    low_relevance = EvidenceSnippet(id="a", source="s", text=many_keywords, score=0.10)
    high_relevance = EvidenceSnippet(id="b", source="s", text="neutral reporting", score=0.90)

    ordered = agent.select(Claim(id="c", raw_text="x"), [low_relevance, high_relevance], k=2)
    check("bounded boost cannot let 9 keywords swamp a large relevance gap",
          ordered[0].id == "b", f"winner={ordered[0].id}")

    mid_a = EvidenceSnippet(id="ref", source="s", text="this is false", score=0.50)
    mid_b = EvidenceSnippet(id="neu", source="s", text="neutral", score=0.55)
    ordered = agent.select(Claim(id="c", raw_text="x"), [mid_a, mid_b], k=2)
    check("refuting evidence still wins when relevance is comparable",
          ordered[0].id == "ref", f"winner={ordered[0].id}")


# ---------------------------------------------------------------------- step 7
async def test_leakage_measured_and_filtered():
    results = [R("https://www.snopes.com/fact-check/z", 1), R("https://bbc.com/news/y", 2)]

    measured = RetrievalAgent(llm=None, search_client=StaticSearchClient(default=results))
    got = await measured.retrieve_evidence(Claim(id="c", raw_text="x"))
    check("fact-check leakage is measured",
          measured.fact_check_results_seen > 0 and measured.fact_check_leakage_rate > 0,
          f"rate={measured.fact_check_leakage_rate:.2f}")
    check("leakage is NOT filtered unless configured",
          any("snopes" in s.id for s in got))

    filtered = RetrievalAgent(llm=None, search_client=StaticSearchClient(default=results),
                              exclude_domains=["snopes.com"])
    got = await filtered.retrieve_evidence(Claim(id="c", raw_text="x"))
    check("EXCLUDE_DOMAINS removes the source",
          not any("snopes" in s.id for s in got) and len(got) == 1)
    check("leakage is still measured even while filtering",
          filtered.fact_check_results_seen > 0)


# ---------------------------------------------------------------------- step 6
async def test_process_text_loop_a():
    class ArticleLLM(ScriptedLLM):
        async def complete(self, prompt, **kwargs):
            if "JSON array" in prompt:
                return (
                    '[{"claim_text": "Claim one", "sentence_index": 0, "importance": 0.9},'
                    ' {"claim_text": "Claim two", "sentence_index": 1, "importance": 0.8},'
                    ' {"claim_text": "Claim three", "sentence_index": 2, "importance": 0.7}]'
                )
            return await super().complete(prompt, **kwargs)

    llm = ArticleLLM()
    search = StaticSearchClient(default=[R("https://a.com/1", 1)])
    pipeline = FactCheckingPipeline(
        llm=llm,
        retrieval_agent=RetrievalAgent(llm=llm, search_client=search),
        max_iterations=1,
    )

    article = await pipeline.process_text("Some article. More text. Even more.", min_importance=0.5)
    check("one result per detected claim",
          len(article.results) == 3, f"{len(article.results)} results")
    check("no article-level verdict field exists",
          not hasattr(article, "verdict"))
    check("verdict counts are reported",
          article.counts_by_label().get("SUPPORTS") == 3, f"{article.counts_by_label()}")

    capped = await pipeline.process_text("Some article. More text. Even more.", max_claims=2)
    check("max_claims caps the work",
          capped.checked_claim_count == 2 and capped.detected_claim_count == 3,
          f"checked={capped.checked_claim_count} detected={capped.detected_claim_count}")

    pipeline.query_budget_per_article = 1
    budgeted = await pipeline.process_text("Some article. More text. Even more.")
    check("per-article query budget stops the run and says so",
          budgeted.budget_exhausted and len(budgeted.failures) > 0,
          f"checked={len(budgeted.results)} failures={len(budgeted.failures)}")


async def test_failing_claim_is_isolated():
    """One claim raising must not abort the document.

    The failure is injected in the SEARCH client rather than an agent, because
    every agent wraps its own LLM call in try/except — an exception raised there
    is swallowed and would never reach process_text, so the test would pass
    without exercising the isolation path at all.
    """
    class ExplodingSearch(StaticSearchClient):
        async def search(self, query, k=5):
            if "BOOM" in query:
                raise RuntimeError("search backend exploded")
            return await super().search(query, k=k)

    class ArticleLLM(ScriptedLLM):
        async def complete(self, prompt, **kwargs):
            if "JSON array" in prompt:
                return (
                    '[{"claim_text": "good claim", "sentence_index": 0, "importance": 0.9},'
                    ' {"claim_text": "BOOM claim", "sentence_index": 1, "importance": 0.8}]'
                )
            return await super().complete(prompt, **kwargs)

    llm = ArticleLLM()
    pipeline = FactCheckingPipeline(
        llm=llm,
        # llm=None on retrieval -> queries are the claim text, so "BOOM" reaches search.
        retrieval_agent=RetrievalAgent(
            llm=None, search_client=ExplodingSearch(default=[R("https://a.com/1", 1)])),
        max_iterations=1,
        # Rule-based normalization preserves the raw text (LLM normalization would
        # rewrite it and hide the marker).
        use_llm_normalization=False,
    )
    article = await pipeline.process_text("Text one. Text two.")

    check("the good claim still produced a result",
          len(article.results) == 1, f"{len(article.results)} results")
    check("the failing claim was recorded as a failure, not lost",
          len(article.failures) == 1, f"{len(article.failures)} failures")
    check("the failure carries the claim text and the error",
          "BOOM" in article.failures[0].claim_text
          and "exploded" in article.failures[0].error,
          f"{article.failures[0].claim_text!r} / {article.failures[0].error!r}")


# ---------------------------------------------------------------------- step 5
SAMPLE_HTML = """
<html><head><title>T</title><script>var x=1;</script></head><body>
<nav>Home About Contact Subscribe</nav>
<p>Saudi Arabia produced 9.6 million barrels per day in 2024 according to OPEC data
released this month, making it the largest single producer in the cartel.</p>
<p>Unrelated paragraph about a local football match between two teams that finished
very late in the evening yesterday before the crowds went home.</p>
<p>The kingdom has pledged to maintain output levels through the remainder of the year
despite pressure from other oil producing nations to reduce supply.</p>
</body></html>
"""


def test_extraction_and_passages():
    from factcheck_agent.content_fetch import _looks_binary, extract_main_text, select_passages

    text = extract_main_text(SAMPLE_HTML, url="https://example.com/a")
    check("extraction returns article body", text is not None and len(text) > 100)
    check("extraction strips scripts", "var x=1" not in text)

    passages = select_passages(text, "Saudi Arabia oil production barrels OPEC", max_chars=300)
    check("passage selection prefers claim-relevant text",
          "OPEC" in passages or "barrels" in passages)
    check("passage selection respects the char cap", len(passages) <= 300, f"{len(passages)} chars")
    check("passage selection drops irrelevant paragraphs",
          "football" not in passages)

    check("PDF magic bytes are detected", _looks_binary(b"%PDF-1.7 ...."))
    check("HTML is not flagged binary", not _looks_binary(b"<html><body>hi"))


async def test_enrich_and_fallback():
    from factcheck_agent.content_fetch import ContentFetcher

    def snippet(i, url):
        return EvidenceSnippet(id=url, source="search", text=f"short snippet {i}",
                               score=1.0 - i * 0.1, metadata={"url": url})

    snips = [snippet(i, f"https://site{i}.com/a") for i in range(4)]

    class OkFetcher(ContentFetcher):
        async def _fetch(self, url):
            return SAMPLE_HTML

    fetcher = OkFetcher(cache_dir="/tmp/afc-test-pages-ok", max_chars=500)
    out = await fetcher.enrich(snips, "Saudi Arabia oil production OPEC barrels", top_k=2)
    check("enrichment preserves every source", len(out) == len(snips))
    check("top-k snippets are replaced with article text",
          len(out[0].text) > len(snips[0].text) and out[0].metadata.get("full_text") == "true")
    check("beyond top-k is left untouched",
          out[2].text == snips[2].text and out[3].text == snips[3].text)

    class DeadFetcher(ContentFetcher):
        async def _fetch(self, url):
            return None                      # 403 / timeout / paywall

    dead = DeadFetcher(cache_dir="/tmp/afc-test-pages-dead")
    out = await dead.enrich(snips, "claim text", top_k=2)
    check("failed fetch keeps the original snippet",
          [s.text for s in out] == [s.text for s in snips])
    check("failed fetch never drops a source", len(out) == len(snips))

    class ExplodingFetcher(ContentFetcher):
        async def _fetch(self, url):
            raise RuntimeError("connection reset")

    boom = ExplodingFetcher(cache_dir="/tmp/afc-test-pages-boom")
    out = await boom.enrich(snips, "claim text", top_k=2)
    check("raising fetch still keeps all sources and their snippets",
          len(out) == len(snips) and [s.text for s in out] == [s.text for s in snips])


async def test_wayback_fallback():
    """A failed live fetch should retry via the Wayback Machine.

    Measured on 150 real AVeriTeC evidence URLs: live URLs succeed 64.5% of the
    time, Wayback-origin URLs 80.7%, and the retry recovered 16 of 106 total
    successes. All fetches here are stubbed — no network access.
    """
    from factcheck_agent.content_fetch import ContentFetcher

    ARCHIVED = "<html><body><p>" + ("archived article body text " * 20) + "</p></body></html>"

    class Stub(ContentFetcher):
        """Live fetches fail; only web.archive.org returns content."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.tried = []

        async def _fetch(self, url):
            self.tried.append(url)
            return ARCHIVED if "web.archive.org" in url else None

    f = Stub(cache_dir="/tmp/afc-wb-a")
    text = await f._article_text("https://dead.example.com/a")
    check("failed live fetch recovers via Wayback", bool(text), f"{len(text or '')} chars")
    check("it tried the original first, then the archive",
          len(f.tried) == 2 and "web.archive.org" in f.tried[1], f"{f.tried}")
    check("recovery is counted", f.wayback_recoveries == 1)

    # An archive URL that fails must NOT be re-wrapped into a second archive URL.
    # The stub records every attempt and always fails, so the assertion below
    # depends on the guard rather than on nothing having been called.
    class AlwaysFails(ContentFetcher):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.tried = []

        async def _fetch(self, url):
            self.tried.append(url)
            return None

    g = AlwaysFails(cache_dir="/tmp/afc-wb-b")
    await g._article_text("https://web.archive.org/web/2/https://x.com/a")
    check("an archive URL is not re-wrapped into a second archive URL",
          len(g.tried) == 1 and g.tried[0].count("web.archive.org") == 1, f"{g.tried}")

    # A live URL that fails everywhere should be tried exactly twice, not looped.
    h = AlwaysFails(cache_dir="/tmp/afc-wb-e")
    await h._article_text("https://dead.example.com/c")
    check("a total failure tries original + archive exactly once each",
          len(h.tried) == 2 and h.wayback_recoveries == 0, f"{h.tried}")

    off = Stub(cache_dir="/tmp/afc-wb-c", wayback_fallback=False)
    await off._article_text("https://dead.example.com/b")
    check("fallback can be disabled", len(off.tried) == 1 and off.wayback_recoveries == 0,
          f"{off.tried}")

    # A successful live fetch must not trigger the archive at all.
    class Live(ContentFetcher):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.tried = []

        async def _fetch(self, url):
            self.tried.append(url)
            return ARCHIVED

    live = Live(cache_dir="/tmp/afc-wb-d")
    await live._article_text("https://alive.example.com/a")
    check("a working live fetch never hits the archive",
          len(live.tried) == 1 and live.wayback_recoveries == 0, f"{live.tried}")


async def main():
    print("--- Loop B: retrieval refinement ---")
    await test_loop_iterates_with_new_queries()
    await test_no_second_round_when_sufficient()
    await test_stops_when_no_gaps_given()
    await test_max_iterations_honoured()

    print("\n--- Scoring ---")
    await test_scores_and_rrf()
    test_refutation_boost_is_bounded()

    print("\n--- Full-text enrichment ---")
    test_extraction_and_passages()
    await test_enrich_and_fallback()
    await test_wayback_fallback()

    print("\n--- Leakage ---")
    await test_leakage_measured_and_filtered()

    print("\n--- Loop A: article orchestration ---")
    await test_process_text_loop_a()
    await test_failing_claim_is_isolated()

    print("\nAll Stage 2 behavioural checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
