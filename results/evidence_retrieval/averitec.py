"""
AVeriTeC loader for Stage 2 (evidence retrieval) evaluation.

Dataset shape (per claim):
    claim, label, justification, claim_date, speaker, reporting_source,
    fact_checking_article, claim_types, questions[]
      questions[].question
      questions[].answers[] -> answer, answer_type, source_url,
                               cached_source_url, source_medium

What this module handles, and why each matters:

* **Wayback unwrapping.** 31.9% of dev gold `source_url`s are
  `web.archive.org/web/<ts>/<original>` snapshots, and one is double-wrapped.
  Live web search never returns archive.org URLs, so comparing raw strings
  would report near-zero source recall for purely cosmetic reasons.

* **Scorable subset.** Not every gold answer can be matched against retrieved
  text:
    - `Boolean` answers (~21% of dev) are "Yes"/"No" — no passage can
      semantically "cover" them.
    - `Unanswerable` answers have nothing to find.
    - Non-`Web text` media (PDF, Image/graphic, Video, Web table, Metadata,
      ~25%) are not retrievable by a text pipeline.
    - ~9% of answers carry an empty or non-URL source.
  Scoring over all answers would therefore charge the retriever for things it
  could not possibly retrieve. `scorable_answers` is the defensible subset:
  Extractive/Abstractive + Web text + a usable http URL. On dev that is 820 of
  1399 answers across 391 of 500 claims, and those counts are reported so the
  denominator is never silently narrowed.

* **test.json is unusable.** All 2215 records have `label: None` and zero
  questions — it is the blind shared-task split. Use dev (500, fully
  annotated) or train (3068).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlparse

_WAYBACK = re.compile(r"^https?://web\.archive\.org/web/[^/]+/(https?://.*)$")

SCORABLE_ANSWER_TYPES = frozenset({"Extractive", "Abstractive"})
SCORABLE_MEDIA = frozenset({"Web text"})


def unwrap_url(url: Optional[str]) -> str:
    """Strip Wayback Machine wrappers, including double-wrapped URLs."""
    current = url or ""
    for _ in range(3):
        match = _WAYBACK.match(current)
        if not match:
            break
        current = match.group(1)
    return current


def domain_of(url: Optional[str]) -> str:
    """Registrable-ish host for a URL, Wayback-unwrapped and www-stripped."""
    try:
        netloc = urlparse(unwrap_url(url)).netloc.lower()
    except Exception:
        return ""
    return netloc[4:] if netloc.startswith("www.") else netloc


def is_usable_url(url: Optional[str]) -> bool:
    unwrapped = unwrap_url(url)
    return unwrapped.startswith("http") and bool(urlparse(unwrapped).netloc)


@dataclass(frozen=True)
class GoldAnswer:
    """One annotated evidence answer."""

    question: str
    answer: str
    answer_type: str
    source_medium: str
    source_url: str          # Wayback-unwrapped
    raw_source_url: str

    @property
    def domain(self) -> str:
        return domain_of(self.source_url)

    @property
    def is_scorable(self) -> bool:
        """Can a retrieved passage plausibly be matched against this answer?"""
        return (
            self.answer_type in SCORABLE_ANSWER_TYPES
            and self.source_medium in SCORABLE_MEDIA
            and is_usable_url(self.source_url)
            and len(self.answer.strip()) > 0
        )


@dataclass
class AveritecClaim:
    """A claim plus its annotated evidence."""

    claim_id: str
    claim: str
    label: Optional[str]
    justification: Optional[str]
    claim_date: Optional[str]
    speaker: Optional[str]
    reporting_source: Optional[str]
    fact_checking_article: Optional[str]
    claim_types: List[str] = field(default_factory=list)
    answers: List[GoldAnswer] = field(default_factory=list)

    @property
    def scorable_answers(self) -> List[GoldAnswer]:
        return [a for a in self.answers if a.is_scorable]

    @property
    def gold_urls(self) -> List[str]:
        return sorted({a.source_url for a in self.answers if is_usable_url(a.source_url)})

    @property
    def gold_domains(self) -> List[str]:
        return sorted({d for d in (a.domain for a in self.answers) if d})

    @property
    def fact_check_domain(self) -> str:
        """Domain of the fact-checking article this claim was taken from.

        Retrieving this domain is the sharpest possible leakage signal: the
        system would be reading the very article the claim was derived from.
        """
        return domain_of(self.fact_checking_article)


def load_split(path: str) -> List[AveritecClaim]:
    """Load an AVeriTeC split into AveritecClaim objects."""
    with open(path) as f:
        raw = json.load(f)

    claims: List[AveritecClaim] = []
    for index, record in enumerate(raw):
        answers: List[GoldAnswer] = []
        for question in record.get("questions") or []:
            q_text = question.get("question") or ""
            for ans in question.get("answers") or []:
                raw_url = ans.get("source_url") or ""
                answers.append(GoldAnswer(
                    question=q_text,
                    answer=(ans.get("answer") or "").strip(),
                    answer_type=ans.get("answer_type") or "",
                    source_medium=ans.get("source_medium") or "",
                    source_url=unwrap_url(raw_url),
                    raw_source_url=raw_url,
                ))

        claims.append(AveritecClaim(
            claim_id=f"av-{index:05d}",
            claim=(record.get("claim") or "").strip(),
            label=record.get("label"),
            justification=record.get("justification"),
            claim_date=record.get("claim_date"),
            speaker=record.get("speaker"),
            reporting_source=record.get("reporting_source"),
            fact_checking_article=record.get("fact_checking_article"),
            claim_types=list(record.get("claim_types") or []),
            answers=answers,
        ))
    return claims


def evaluable(claims: Sequence[AveritecClaim]) -> List[AveritecClaim]:
    """Claims with at least one scorable gold answer."""
    return [c for c in claims if c.scorable_answers]


def describe(claims: Sequence[AveritecClaim]) -> Dict[str, object]:
    """Summary stats, so a run always reports what it could and could not score."""
    total_answers = sum(len(c.answers) for c in claims)
    scorable = sum(len(c.scorable_answers) for c in claims)
    labels: Dict[str, int] = {}
    for c in claims:
        labels[str(c.label)] = labels.get(str(c.label), 0) + 1
    return {
        "claims": len(claims),
        "claims_with_scorable_answers": len(evaluable(claims)),
        "answers_total": total_answers,
        "answers_scorable": scorable,
        "answers_scorable_pct": round(100 * scorable / total_answers, 1) if total_answers else 0.0,
        "labels": labels,
    }


def default_data_dir() -> str:
    """Where the AVeriTeC json files live; override with AVERITEC_DIR."""
    return os.environ.get(
        "AVERITEC_DIR",
        os.path.expanduser("~/Downloads/AVeriTeC dataset"),
    )


if __name__ == "__main__":
    import sys

    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    path = os.path.join(default_data_dir(), f"{split}.json")
    claims = load_split(path)
    stats = describe(claims)
    print(f"{path}")
    for key, value in stats.items():
        print(f"  {key:30} {value}")
