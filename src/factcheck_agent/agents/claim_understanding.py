"""Claim understanding agent with detection and normalization capabilities."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import List, Optional

from factcheck_agent.llm_client import LLMClient
from factcheck_agent.models import Claim, DetectedClaim


logger = logging.getLogger(__name__)


class ClaimDetectionError(RuntimeError):
    """Raised when claim detection fails due to an LLM error or unparseable
    output — as opposed to legitimately finding zero claims in the text."""


class ClaimUnderstandingAgent:
    """
    Claim understanding agent with two main responsibilities:
    1. Detecting check-worthy claims from long input text
    2. Performing advanced, LLM-based normalization on claims
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        """
        Initialize the claim understanding agent.

        Args:
            llm: LLM client for advanced operations. If None, only basic
                 rule-based normalization will be available.
        """
        self.llm = llm

    def _parse_claims_json(self, response: str) -> Optional[list]:
        """Extract the JSON array from an LLM response. Returns None on failure."""
        candidates = []
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            candidates.append(match.group())
        candidates.append(response.strip())

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue
        return None

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using nltk's punkt tokenizer, which correctly
        handles abbreviations (U.S., Dr.), decimals (3.5%), and initials.

        Falls back to naive regex splitting if nltk/punkt is unavailable, so
        detection degrades gracefully rather than crashing.
        """
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except (ImportError, LookupError) as e:
            logger.warning(
                "nltk punkt unavailable (%s); falling back to regex sentence split. "
                "Run: python -c \"import nltk; nltk.download('punkt_tab')\"", e
            )
            sentences = re.split(r"[.!?]+", text)

        return [s.strip() for s in sentences if s.strip()]

    def normalize_claim(self, claim: Claim) -> Claim:
        """
        Basic rule-based normalization (synchronous, lightweight).

        This is a fallback method that performs simple text cleanup:
        - Strips whitespace
        - Removes duplicate spaces

        Case is PRESERVED. Named-entity casing ("Saudi Arabia", "U.S.") is
        meaningful for retrieval and entity matching, and the LLM path
        (`normalize_claim_llm`) preserves it — so this fallback must too, or the
        two paths would emit different-cased text for the same claim.

        For advanced normalization with LLM, use `normalize_claim_llm()` instead.

        Args:
            claim: The claim to normalize.

        Returns:
            The claim with normalized_text set.
        """
        normalized = claim.raw_text.strip()
        # Remove duplicate spaces (case is preserved on purpose — see docstring)
        normalized = re.sub(r"\s+", " ", normalized)
        claim.normalized_text = normalized
        return claim

    async def normalize_claim_llm(
        self,
        claim: Claim,
        context: Optional[str] = None,
        max_context_chars: int = 2000,
    ) -> Claim:
        """
        Advanced LLM-based normalization of a claim.

        Uses the LLM to:
        - Clean and standardize the claim text
        - Remove filler words and unnecessary qualifiers
        - Normalize entities (dates, numbers, locations)
        - Ensure the claim is in a fact-checkable form

        Args:
            claim: The claim to normalize.
            context: Optional context text to help resolve pronouns without
                    fact-correction. Callers should pass context RELEVANT to this
                    claim (e.g. a local window around it), not just the document
                    head — see `_context_window`. The cap below is only a token
                    safety bound, not a substitute for passing relevant context.
            max_context_chars: Upper bound on context length included in the
                    prompt, to avoid blowing the token budget on long inputs.

        Returns:
            The claim with normalized_text set using LLM processing.

        Raises:
            RuntimeError: If LLM is not available.
        """
        if self.llm is None:
            raise RuntimeError(
                "LLM is required for advanced normalization. "
                "Initialize ClaimUnderstandingAgent with an LLM client, "
                "or use normalize_claim() for basic rule-based normalization."
            )

        context_instruction = ""
        if context:
            # Cap only as a token-budget safeguard. Callers pass claim-relevant
            # context (a local window), so this should rarely truncate.
            context_instruction = (
                "\n\nContext (for resolving pronouns only, DO NOT use to correct facts):\n"
                f"{context[:max_context_chars]}"
            )

        prompt = f"""You are a fact-checking assistant. Normalize the following claim to make it clear, concise, and fact-checkable.

CRITICAL RULES:
- DO NOT correct or change any facts, even if you know they are incorrect
- DO NOT replace entities with what you believe are the "correct" entities
- PRESERVE the claim exactly as stated, only clean up the language
- Your job is to normalize the LANGUAGE, not to fact-check or correct the claim

Normalization Rules:
- Remove filler words, qualifiers, and unnecessary phrases
- Standardize dates, numbers, and locations to canonical forms (but keep the same entities)
- Keep the core factual assertion intact
- Output ONLY the normalized claim text, no explanations
- Keep only factual content; remove opinions and emotional language.
- Make the claim standalone by replacing pronouns with specific entities FROM THE CONTEXT (if provided)
- Ensure the claim expresses only one atomic fact.
- Clarify vague or relative terms into explicit, concrete language when possible.
- Remove unnecessary context such as "I heard," "someone said," etc.
- Rewrite the claim as a neutral, declarative sentence.
- Correct grammar; make the statement concise and unambiguous.
- If the claim is too vague, rewrite it into the clearest possible version without adding new facts.

Claim: {claim.raw_text}{context_instruction}

Normalized claim:"""

        try:
            normalized_text = await self.llm.complete(prompt, temperature=0.2, max_tokens=200)
            # Clean up the response (remove quotes, extra whitespace)
            normalized_text = normalized_text.strip().strip('"').strip("'").strip()
            claim.normalized_text = normalized_text
            return claim
        except Exception:
            # Fallback to basic normalization if LLM fails
            return self.normalize_claim(claim)

    def _context_window(
        self,
        sentences: List[str],
        sentence_index: Optional[int],
        full_text: str,
        before: int = 3,
        after: int = 1,
    ) -> str:
        """Return local context around a claim for pronoun resolution.

        A claim late in an article cannot resolve its pronouns from the document
        head, so we hand the normalizer the sentences immediately surrounding the
        claim instead. Antecedents almost always precede the pronoun, so we take
        more sentences before than after. Falls back to the full text when the
        sentence index is missing or out of range.
        """
        if sentence_index is not None and 0 <= sentence_index < len(sentences):
            lo = max(0, sentence_index - before)
            hi = min(len(sentences), sentence_index + after + 1)
            return " ".join(sentences[lo:hi])
        return full_text

    def _build_detection_prompt(self, text: str, min_importance: float) -> str:
        """Build the claim-detection prompt for the given text.

        Subclasses can override this to swap in a different prompt template
        while inheriting all of detect_claims' parsing/error handling. This is
        the single seam used by prompt-ablation experiments, so that arms
        differ ONLY by the prompt text and share identical extraction logic.
        """
        return f"""You are a fact-checking assistant. Analyze the following text and identify factual claims that are worth fact-checking.

CRITICAL: Each claim must contain ONLY ONE atomic fact. If a sentence contains multiple facts, split them into separate claims.

IMPORTANT OUTPUT LIMIT:
- Return AT MOST 20 claims total.
- Prefer the most check-worthy, specific, and verifiable claims (numbers, dates, named entities, concrete events).
- If you find more than 20 candidate claims, select the best 20 by importance.

OUTPUT STYLE (match benchmark "claim_text"):
- Each claim must be a short, standalone, decontextualized factual statement.
- Do NOT write narrative framing like "The article says..." / "The case alleges...".
- Prefer direct statements like "X happened on DATE" / "Person Y said Z" / "Charges were dropped because ...".
- Avoid hedging words unless present in the text.

A check-worthy claim is:
- A factual assertion that can be verified
- Specific enough to be fact-checked (has dates, numbers, names, locations)
- Not an opinion or subjective statement
- Not a question
- Contains ONLY ONE atomic fact (not multiple facts combined)

Examples:
- GOOD (atomic): "Saudi Arabia is the world's largest oil producer"
- GOOD (atomic): "Saudi Arabia aims to diversify its energy sources"
- BAD (compound): "Saudi Arabia, which is the world's largest oil producer, aims to diversify its energy sources"
  → Split into two claims:
    1. "Saudi Arabia is the world's largest oil producer"
    2. "Saudi Arabia aims to diversify its energy sources"

For each claim you find, provide:
1. The exact claim text (one atomic fact only)
2. An importance score (0.0-1.0) indicating how check-worthy it is
3. The sentence index (0-based) where it appears

Text to analyze:
{text}

Respond with a JSON array of objects, each with:
- "claim_text": the claim text (one atomic fact only)
- "sentence_index": the sentence index (0-based)
- "importance": a float between 0.0 and 1.0

Output format:
[
  {{"claim_text": "...", "sentence_index": 0, "importance": 0.8}},
  ...
]

Only include claims with importance >= {min_importance}. Output ONLY the JSON array, no other text."""

    async def detect_claims(
        self,
        text: str,
        min_importance: float = 0.5,
        normalize_detected: bool = True,
        on_error: str = "raise",  # "raise" | "empty"
    ) -> List[DetectedClaim]:
        """
        Detect check-worthy claims from a long input text.

        Uses LLM to identify factual claims that are worth fact-checking.
        Returns a list of DetectedClaim objects with positions and importance scores.

        Args:
            text: The input text (article, post, etc.) to analyze.
            min_importance: Minimum importance score (0-1) for claims to include.
                           Default is 0.5.
            normalize_detected: If True, normalize each detected claim after detection.
                               Default is True.
            on_error: What to do when detection fails due to an LLM or parse error.
                     "raise" (default) raises ClaimDetectionError — recommended for
                     eval so failures are visible. "empty" returns [] and logs the
                     error — matches the old behavior but with logging.

        Returns:
            List of DetectedClaim objects, sorted by importance (highest first).

        Raises:
            RuntimeError: If LLM is not available.
            ClaimDetectionError: If the LLM call or JSON parsing fails and
                                on_error="raise".
        """
        if self.llm is None:
            raise RuntimeError(
                "LLM is required for claim detection. "
                "Initialize ClaimUnderstandingAgent with an LLM client."
            )

        # Split text into sentences for better processing
        sentences = self._split_sentences(text)

        if not sentences:
            return []  # legitimately empty input, not an error

        prompt = self._build_detection_prompt(text, min_importance)

        # Stage 1 — LLM call
        try:
            response = await self.llm.complete(prompt, temperature=0.3, max_tokens=1000)
        except Exception as e:
            logger.error(
                "Claim detection LLM call failed (text_len=%d): %s", len(text), e
            )
            if on_error == "empty":
                return []
            raise ClaimDetectionError(f"LLM call failed during detection: {e}") from e

        # Stage 2 — JSON parsing
        claims_data = self._parse_claims_json(response)
        if claims_data is None:
            logger.error(
                "Could not parse claims JSON. Raw response (truncated):\n%s",
                response[:1000],
            )
            if on_error == "empty":
                return []
            raise ClaimDetectionError("LLM returned unparseable JSON for claim detection")

        # Stage 3 — build DetectedClaim objects, tolerant per-item
        detected_claims: List[DetectedClaim] = []

        for claim_data in claims_data:
            try:
                if not isinstance(claim_data, dict):
                    continue

                claim_text = claim_data.get("claim_text", "").strip()
                if not claim_text:
                    continue

                importance = float(claim_data.get("importance", 0.5))
                if importance < min_importance:
                    continue

                sentence_idx = claim_data.get("sentence_index")
                if sentence_idx is not None:
                    sentence_idx = int(sentence_idx)

                # Character offsets are intentionally not computed: the prompt asks
                # the model to rewrite/decontextualize claims, so claim_text is rarely
                # a verbatim substring of the source. sentence_index is still kept as a
                # coarse locator. If offsets are ever needed, the model would have to
                # return the exact source span it derived each claim from.
                detected_claim = DetectedClaim(
                    id=f"detected-{uuid.uuid4().hex[:8]}",
                    raw_text=claim_text,
                    normalized_text=None,
                    start_char=None,
                    end_char=None,
                    sentence_index=sentence_idx,
                    importance=importance,
                    metadata={"detection_method": "llm"},
                )
                detected_claims.append(detected_claim)

            except (ValueError, TypeError, KeyError) as e:
                # one bad item doesn't take down the whole document
                logger.warning("Skipping malformed claim item %r: %s", claim_data, e)
                continue

        # Normalize each detected claim (if requested).
        # The per-claim LLM normalizations are independent, so run them
        # concurrently instead of sequentially — for a document with N claims
        # this collapses N awaited round-trips into a single gather. Each claim
        # keeps its own basic-normalization fallback, so one failed call never
        # rejects the whole batch.
        if normalize_detected and detected_claims:
            if self.llm:

                async def _normalize_one(dc: DetectedClaim) -> None:
                    claim_obj = Claim(id=dc.id, raw_text=dc.raw_text)
                    # Give each claim context local to where it appears, so claims
                    # late in the article can still resolve their pronouns.
                    ctx = self._context_window(sentences, dc.sentence_index, text)
                    try:
                        claim_obj = await self.normalize_claim_llm(claim_obj, context=ctx)
                    except Exception:
                        claim_obj = self.normalize_claim(claim_obj)
                    dc.normalized_text = claim_obj.normalized_text

                results = await asyncio.gather(
                    *(_normalize_one(dc) for dc in detected_claims),
                    return_exceptions=True,
                )
                # _normalize_one falls back internally, but guard against any
                # unexpected error surfaced by gather itself.
                for dc, res in zip(detected_claims, results):
                    if isinstance(res, Exception) and dc.normalized_text is None:
                        dc.normalized_text = self.normalize_claim(
                            Claim(id=dc.id, raw_text=dc.raw_text)
                        ).normalized_text
            else:
                for dc in detected_claims:
                    dc.normalized_text = self.normalize_claim(
                        Claim(id=dc.id, raw_text=dc.raw_text)
                    ).normalized_text

        # Sort by importance (highest first)
        detected_claims.sort(key=lambda c: c.importance or 0.0, reverse=True)

        return detected_claims


# # """Claim understanding agent for basic normalization."""

# # from factcheck_agent.llm_client import LLMClient
# # from factcheck_agent.models import Claim


# # class ClaimUnderstandingAgent:
# #     """Perform lightweight claim normalization (placeholder for richer LLM logic)."""

# #     def __init__(self, llm: LLMClient):
# #         self.llm = llm

# #     def normalize_claim(self, claim: Claim) -> Claim:
# #         """Normalize claim text via simple rule-based cleanup."""
# #         normalized = claim.raw_text.strip()
# #         normalized = normalized.lower()
# #         claim.normalized_text = normalized
# #         return claim

# """Claim understanding agent with detection and normalization capabilities."""

# from __future__ import annotations

# import re
# import uuid
# import logging
# import json
# from typing import List, Optional

# from factcheck_agent.llm_client import LLMClient
# from factcheck_agent.models import Claim, DetectedClaim


# logger = logging.getLogger(__name__)



# class ClaimDetectionError(RuntimeError):
#     """Raised when claim detection fails due to an LLM error or unparseable
#     output — as opposed to legitimately finding zero claims in the text."""

# class ClaimUnderstandingAgent:
#     """
#     Claim understanding agent with two main responsibilities:
#     1. Detecting check-worthy claims from long input text
#     2. Performing advanced, LLM-based normalization on claims
#     """

#     def __init__(self, llm: LLMClient | None = None) -> None:
#         """
#         Initialize the claim understanding agent.

#         Args:
#             llm: LLM client for advanced operations. If None, only basic
#                  rule-based normalization will be available.
#         """
#         self.llm = llm

#     def _parse_claims_json(self, response: str) -> Optional[list]:
#         """Extract the JSON array from an LLM response. Returns None on failure."""
#         import json

#         candidates = []
#         match = re.search(r"\[.*\]", response, re.DOTALL)
#         if match:
#             candidates.append(match.group())
#         candidates.append(response.strip())

#         for candidate in candidates:
#             try:
#                 data = json.loads(candidate)
#                 if isinstance(data, list):
#                     return data
#             except json.JSONDecodeError:
#                 continue
#         return None

#     def normalize_claim(self, claim: Claim) -> Claim:
#         """
#         Basic rule-based normalization (synchronous, lightweight).

#         This is a fallback method that performs simple text cleanup:
#         - Strips whitespace
#         - Converts to lowercase
#         - Removes duplicate spaces

#         For advanced normalization with LLM, use `normalize_claim_llm()` instead.

#         Args:
#             claim: The claim to normalize.

#         Returns:
#             The claim with normalized_text set.
#         """
#         normalized = claim.raw_text.strip()
#         # Remove duplicate spaces
#         normalized = re.sub(r"\s+", " ", normalized)
#         normalized = normalized.lower()
#         claim.normalized_text = normalized
#         return claim

#     async def normalize_claim_llm(self, claim: Claim, context: Optional[str] = None) -> Claim:
#         """
#         Advanced LLM-based normalization of a claim.

#         Uses the LLM to:
#         - Clean and standardize the claim text
#         - Remove filler words and unnecessary qualifiers
#         - Normalize entities (dates, numbers, locations)
#         - Ensure the claim is in a fact-checkable form

#         Args:
#             claim: The claim to normalize.
#             context: Optional context text (e.g., original article) to help resolve pronouns
#                     without fact-correction.

#         Returns:
#             The claim with normalized_text set using LLM processing.

#         Raises:
#             RuntimeError: If LLM is not available.
#         """
#         if self.llm is None:
#             raise RuntimeError(
#                 "LLM is required for advanced normalization. "
#                 "Initialize ClaimUnderstandingAgent with an LLM client, "
#                 "or use normalize_claim() for basic rule-based normalization."
#             )

#         context_instruction = ""
#         if context:
#             # Limit context length to avoid token limits
#             context_instruction = f"\n\nContext (for resolving pronouns only, DO NOT use to correct facts):\n{context[:500]}"

#         prompt = f"""You are a fact-checking assistant. Normalize the following claim to make it clear, concise, and fact-checkable.

# CRITICAL RULES:
# - DO NOT correct or change any facts, even if you know they are incorrect
# - DO NOT replace entities with what you believe are the "correct" entities
# - PRESERVE the claim exactly as stated, only clean up the language
# - Your job is to normalize the LANGUAGE, not to fact-check or correct the claim

# Normalization Rules:
# - Remove filler words, qualifiers, and unnecessary phrases
# - Standardize dates, numbers, and locations to canonical forms (but keep the same entities)
# - Keep the core factual assertion intact
# - Output ONLY the normalized claim text, no explanations
# - Keep only factual content; remove opinions and emotional language.
# - Make the claim standalone by replacing pronouns with specific entities FROM THE CONTEXT (if provided)
# - Ensure the claim expresses only one atomic fact.
# - Clarify vague or relative terms into explicit, concrete language when possible.
# - Remove unnecessary context such as "I heard," "someone said," etc.
# - Rewrite the claim as a neutral, declarative sentence.
# - Correct grammar; make the statement concise and unambiguous.
# - If the claim is too vague, rewrite it into the clearest possible version without adding new facts.

# Claim: {claim.raw_text}{context_instruction}

# Normalized claim:"""

#         try:
#             normalized_text = await self.llm.complete(prompt, temperature=0.2, max_tokens=200)
#             # Clean up the response (remove quotes, extra whitespace)
#             normalized_text = normalized_text.strip().strip('"').strip("'").strip()
#             claim.normalized_text = normalized_text
#             return claim
#         except Exception as e:
#             # Fallback to basic normalization if LLM fails
#             return self.normalize_claim(claim)


# def _split_sentences(self, text: str) -> List[str]:
#     """
#     Split text into sentences using nltk's punkt tokenizer, which correctly
#     handles abbreviations (U.S., Dr.), decimals (3.5%), and initials.

#     Falls back to naive regex splitting if nltk/punkt is unavailable, so
#     detection degrades gracefully rather than crashing.
#     """
#     try:
#         from nltk.tokenize import sent_tokenize
#         sentences = sent_tokenize(text)
#     except (ImportError, LookupError) as e:
#         logger.warning(
#             "nltk punkt unavailable (%s); falling back to regex sentence split. "
#             "Run: python -c \"import nltk; nltk.download('punkt_tab')\"", e
#         )
#         sentences = re.split(r"[.!?]+", text)

#     return [s.strip() for s in sentences if s.strip()]


# async def detect_claims(
#     self,
#     text: str,
#     min_importance: float = 0.5,
#     normalize_detected: bool = True,
#     on_error: str = "raise",  # "raise" | "empty"
# ) -> List[DetectedClaim]:
#     """
#     Detect check-worthy claims from a long input text.

#     Uses LLM to identify factual claims that are worth fact-checking.
#     Returns a list of DetectedClaim objects with positions and importance scores.

#     Args:
#         text: The input text (article, post, etc.) to analyze.
#         min_importance: Minimum importance score (0-1) for claims to include.
#                        Default is 0.5.
#         normalize_detected: If True, normalize each detected claim after detection.
#                            Default is True.
#         on_error: What to do when detection fails due to an LLM or parse error.
#                  "raise" (default) raises ClaimDetectionError — recommended for
#                  eval so failures are visible. "empty" returns [] and logs the
#                  error — matches the old behavior but with logging.

#     Returns:
#         List of DetectedClaim objects, sorted by importance (highest first).

#     Raises:
#         RuntimeError: If LLM is not available.
#         ClaimDetectionError: If the LLM call or JSON parsing fails and
#                             on_error="raise".
#     """
#     if self.llm is None:
#         raise RuntimeError(
#             "LLM is required for claim detection. "
#             "Initialize ClaimUnderstandingAgent with an LLM client."
#         )

#     # Split text into sentences for better processing
#     # sentences = re.split(r"[.!?]+", text)
#     # sentences = [s.strip() for s in sentences if s.strip()]
#     sentences = self._split_sentences(text)


#     if not sentences:
#         return []  # legitimately empty input, not an error

#     prompt = f"""You are a fact-checking assistant. Analyze the following text and identify factual claims that are worth fact-checking.

# CRITICAL: Each claim must contain ONLY ONE atomic fact. If a sentence contains multiple facts, split them into separate claims.

# IMPORTANT OUTPUT LIMIT:
# - Return AT MOST 20 claims total.
# - Prefer the most check-worthy, specific, and verifiable claims (numbers, dates, named entities, concrete events).
# - If you find more than 20 candidate claims, select the best 20 by importance.

# OUTPUT STYLE (match benchmark "claim_text"):
# - Each claim must be a short, standalone, decontextualized factual statement.
# - Do NOT write narrative framing like "The article says..." / "The case alleges...".
# - Prefer direct statements like "X happened on DATE" / "Person Y said Z" / "Charges were dropped because ...".
# - Avoid hedging words unless present in the text.

# A check-worthy claim is:
# - A factual assertion that can be verified
# - Specific enough to be fact-checked (has dates, numbers, names, locations)
# - Not an opinion or subjective statement
# - Not a question
# - Contains ONLY ONE atomic fact (not multiple facts combined)

# Examples:
# - GOOD (atomic): "Saudi Arabia is the world's largest oil producer"
# - GOOD (atomic): "Saudi Arabia aims to diversify its energy sources"
# - BAD (compound): "Saudi Arabia, which is the world's largest oil producer, aims to diversify its energy sources"
#   → Split into two claims:
#     1. "Saudi Arabia is the world's largest oil producer"
#     2. "Saudi Arabia aims to diversify its energy sources"

# For each claim you find, provide:
# 1. The exact claim text (one atomic fact only)
# 2. An importance score (0.0-1.0) indicating how check-worthy it is
# 3. The sentence index (0-based) where it appears

# Text to analyze:
# {text}

# Respond with a JSON array of objects, each with:
# - "claim_text": the claim text (one atomic fact only)
# - "sentence_index": the sentence index (0-based)
# - "importance": a float between 0.0 and 1.0

# Output format:
# [
#   {{"claim_text": "...", "sentence_index": 0, "importance": 0.8}},
#   ...
# ]

# Only include claims with importance >= {min_importance}. Output ONLY the JSON array, no other text."""

#     # Stage 1 — LLM call
#     try:
#         response = await self.llm.complete(prompt, temperature=0.3, max_tokens=1000)
#     except Exception as e:
#         logger.error(
#             "Claim detection LLM call failed (text_len=%d): %s", len(text), e
#         )
#         if on_error == "empty":
#             return []
#         raise ClaimDetectionError(f"LLM call failed during detection: {e}") from e

#     # Stage 2 — JSON parsing
#     claims_data = self._parse_claims_json(response)
#     if claims_data is None:
#         logger.error(
#             "Could not parse claims JSON. Raw response (truncated):\n%s",
#             response[:1000],
#         )
#         if on_error == "empty":
#             return []
#         raise ClaimDetectionError("LLM returned unparseable JSON for claim detection")

#     # Stage 3 — build DetectedClaim objects, tolerant per-item
#     detected_claims: List[DetectedClaim] = []

#     for claim_data in claims_data:
#         try:
#             if not isinstance(claim_data, dict):
#                 continue

#             claim_text = claim_data.get("claim_text", "").strip()
#             if not claim_text:
#                 continue

#             importance = float(claim_data.get("importance", 0.5))
#             if importance < min_importance:
#                 continue

#             sentence_idx = claim_data.get("sentence_index")
#             if sentence_idx is not None:
#                 sentence_idx = int(sentence_idx)

#          # Character offsets are intentionally not computed: the prompt asks
#             # the model to rewrite/decontextualize claims, so claim_text is rarely
#             # a verbatim substring of the source. sentence_index is still kept as a
#             # coarse locator. See [your decision note / ticket] if offsets are ever
#             # needed — they'd require the model to return the exact source span.
#             detected_claim = DetectedClaim(
#                 id=f"detected-{uuid.uuid4().hex[:8]}",
#                 raw_text=claim_text,
#                 normalized_text=None,
#                 start_char=None,
#                 end_char=None,
#                 sentence_index=sentence_idx,
#                 importance=importance,
#                 metadata={"detection_method": "llm"},
#             )
#             detected_claims.append(detected_claim)

#         except (ValueError, TypeError, KeyError) as e:
#             # one bad item doesn't take down the whole document
#             logger.warning("Skipping malformed claim item %r: %s", claim_data, e)
#             continue

#     # Normalize each detected claim (if requested)
#     if normalize_detected:
#         for detected_claim in detected_claims:
#             claim_obj = Claim(id=detected_claim.id, raw_text=detected_claim.raw_text)
#             if self.llm:
#                 try:
#                     claim_obj = await self.normalize_claim_llm(claim_obj, context=text)
#                 except Exception:
#                     claim_obj = self.normalize_claim(claim_obj)
#             else:
#                 claim_obj = self.normalize_claim(claim_obj)
#             detected_claim.normalized_text = claim_obj.normalized_text

#     # Sort by importance (highest first)
#     detected_claims.sort(key=lambda c: c.importance or 0.0, reverse=True)

#     return detected_claims

# #     async def detect_claims(
# #         self, 
# #         text: str, 
# #         min_importance: float = 0.5,
# #         normalize_detected: bool = True
# #     ) -> List[DetectedClaim]:
# #         """
# #         Detect check-worthy claims from a long input text.

# #         Uses LLM to identify factual claims that are worth fact-checking.
# #         Returns a list of DetectedClaim objects with positions and importance scores.

# #         Args:
# #             text: The input text (article, post, etc.) to analyze.
# #             min_importance: Minimum importance score (0-1) for claims to include.
# #                            Default is 0.5.
# #             normalize_detected: If True, normalize each detected claim after detection.
# #                                Default is True.

# #         Returns:
# #             List of DetectedClaim objects, sorted by importance (highest first).

# #         Raises:
# #             RuntimeError: If LLM is not available.
# #         """
# #         if self.llm is None:
# #             raise RuntimeError(
# #                 "LLM is required for claim detection. "
# #                 "Initialize ClaimUnderstandingAgent with an LLM client."
# #             )

# #         # Split text into sentences for better processing
# #         sentences = re.split(r"[.!?]+", text)
# #         sentences = [s.strip() for s in sentences if s.strip()]

# #         if not sentences:
# #             return []

# #         # Use LLM to detect claims from the text
# #         prompt = f"""You are a fact-checking assistant. Analyze the following text and identify factual claims that are worth fact-checking.

# # CRITICAL: Each claim must contain ONLY ONE atomic fact. If a sentence contains multiple facts, split them into separate claims.

# # IMPORTANT OUTPUT LIMIT:
# # - Return AT MOST 20 claims total.
# # - Prefer the most check-worthy, specific, and verifiable claims (numbers, dates, named entities, concrete events).
# # - If you find more than 20 candidate claims, select the best 20 by importance.

# # OUTPUT STYLE (match benchmark "claim_text"):
# # - Each claim must be a short, standalone, decontextualized factual statement.
# # - Do NOT write narrative framing like "The article says..." / "The case alleges...".
# # - Prefer direct statements like "X happened on DATE" / "Person Y said Z" / "Charges were dropped because ...".
# # - Avoid hedging words unless present in the text.

# # A check-worthy claim is:
# # - A factual assertion that can be verified
# # - Specific enough to be fact-checked (has dates, numbers, names, locations)
# # - Not an opinion or subjective statement
# # - Not a question
# # - Contains ONLY ONE atomic fact (not multiple facts combined)

# # Examples:
# # - GOOD (atomic): "Saudi Arabia is the world's largest oil producer"
# # - GOOD (atomic): "Saudi Arabia aims to diversify its energy sources"
# # - BAD (compound): "Saudi Arabia, which is the world's largest oil producer, aims to diversify its energy sources"
# #   → Split into two claims:
# #     1. "Saudi Arabia is the world's largest oil producer"
# #     2. "Saudi Arabia aims to diversify its energy sources"

# # For each claim you find, provide:
# # 1. The exact claim text (one atomic fact only)
# # 2. An importance score (0.0-1.0) indicating how check-worthy it is
# # 3. The sentence index (0-based) where it appears

# # Text to analyze:
# # {text}

# # Respond with a JSON array of objects, each with:
# # - "claim_text": the claim text (one atomic fact only)
# # - "sentence_index": the sentence index (0-based)
# # - "importance": a float between 0.0 and 1.0

# # Output format:
# # [
# #   {{"claim_text": "...", "sentence_index": 0, "importance": 0.8}},
# #   ...
# # ]

# # Only include claims with importance >= {min_importance}. Output ONLY the JSON array, no other text."""

# #         try:
# #             response = await self.llm.complete(prompt, temperature=0.3, max_tokens=1000)
            
# #             # Parse JSON response
# #             import json
# #             import re as regex
            
# #             # Extract JSON from response (handle markdown code blocks)
# #             json_match = regex.search(r'\[.*\]', response, regex.DOTALL)
# #             if json_match:
# #                 claims_data = json.loads(json_match.group())
# #             else:
# #                 # Try parsing entire response
# #                 claims_data = json.loads(response.strip())

# #             detected_claims: List[DetectedClaim] = []
            
# #             for i, claim_data in enumerate(claims_data):
# #                 if not isinstance(claim_data, dict):
# #                     continue
                
# #                 claim_text = claim_data.get("claim_text", "").strip()
# #                 if not claim_text:
# #                     continue
                
# #                 importance = float(claim_data.get("importance", 0.5))
# #                 if importance < min_importance:
# #                     continue
                
# #                 sentence_idx = claim_data.get("sentence_index")
# #                 if sentence_idx is not None:
# #                     sentence_idx = int(sentence_idx)
                
# #                 # Find character positions in original text
# #                 start_char = None
# #                 end_char = None
# #                 if sentence_idx is not None and 0 <= sentence_idx < len(sentences):
# #                     # Find the sentence in the original text
# #                     sentence = sentences[sentence_idx]
# #                     start_char = text.find(sentence)
# #                     if start_char >= 0:
# #                         # Find the claim within the sentence
# #                         claim_start = sentence.find(claim_text)
# #                         if claim_start >= 0:
# #                             start_char += claim_start
# #                             end_char = start_char + len(claim_text)
                
# #                 detected_claim = DetectedClaim(
# #                     id=f"detected-{uuid.uuid4().hex[:8]}",
# #                     raw_text=claim_text,
# #                     normalized_text=None,  # Will be set below if normalize_detected is True
# #                     start_char=start_char,
# #                     end_char=end_char,
# #                     sentence_index=sentence_idx,
# #                     importance=importance,
# #                     metadata={"detection_method": "llm"},
# #                 )
# #                 detected_claims.append(detected_claim)
            
# #             # Normalize each detected claim (if requested)
# #             if normalize_detected:
# #                 for detected_claim in detected_claims:
# #                     # Create a Claim object to normalize
# #                     claim_obj = Claim(id=detected_claim.id, raw_text=detected_claim.raw_text)
                    
# #                     # Normalize using LLM if available, otherwise basic normalization
# #                     if self.llm:
# #                         try:
# #                             # Pass the original text as context to help resolve pronouns
# #                             # without fact-correction
# #                             claim_obj = await self.normalize_claim_llm(claim_obj, context=text)
# #                         except Exception:
# #                             # Fallback to basic normalization if LLM fails
# #                             claim_obj = self.normalize_claim(claim_obj)
# #                     else:
# #                         claim_obj = self.normalize_claim(claim_obj)
                    
# #                     # Set the normalized text on the detected claim
# #                     detected_claim.normalized_text = claim_obj.normalized_text
            
# #             # Sort by importance (highest first)
# #             detected_claims.sort(key=lambda c: c.importance or 0.0, reverse=True)
            
# #             return detected_claims
            
# #         except Exception as e:
# #             # Fallback: return empty list or simple sentence-based detection
# #             # For now, return empty list on error
# #             return []
