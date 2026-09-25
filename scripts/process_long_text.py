#!/usr/bin/env python3
"""
Process long text: detect claims and fact-check each one.

The orchestration this script used to contain now lives in
`FactCheckingPipeline.process_text()`, so the library and the CLI share one
implementation. This is a thin caller kept for convenience.

Equivalent CLI:  factcheck --file article.txt
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factcheck_agent.cli import _print_article  # noqa: E402
from factcheck_agent.llm_client import get_default_llm_client  # noqa: E402
from factcheck_agent.pipeline import FactCheckingPipeline  # noqa: E402


SAMPLE_TEXT = """
Saudi Arabia announced plans to invest $100 billion in renewable energy by 2030,
according to a statement from the Ministry of Energy. The country, which is currently
the world's largest oil producer, aims to diversify its energy sources and reduce
carbon emissions. Climate experts have praised the initiative, noting that it represents
a significant shift in the country's energy policy. The investment will focus on solar
and wind power projects across the kingdom. Some analysts believe this move could
transform the global energy market. The announcement came during a climate summit
in Riyadh, where officials also revealed plans to achieve net-zero emissions by 2060.
"""


async def main() -> int:
    text = SAMPLE_TEXT.strip()
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text().strip()

    try:
        llm = get_default_llm_client(use_dummy_if_missing_key=False)
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1

    print(f"📄 Processing {len(text)} characters...\n")
    pipeline = FactCheckingPipeline(llm=llm)
    article = await pipeline.process_text(text, min_importance=0.5)
    _print_article(article)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
