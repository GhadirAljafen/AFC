"""Claim understanding agent using LangChain for comparison with custom implementation."""

from __future__ import annotations

import re
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field

from factcheck_agent.models import Claim, DetectedClaim
from factcheck_agent.config import get_config

# LangChain imports - handle gracefully if not installed
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatPromptTemplate = None
    PydanticOutputParser = None
    ChatOpenAI = None
    ChatAnthropic = None


# Pydantic model for structured claim detection output
class ClaimDetectionOutput(BaseModel):
    """Structured output for claim detection."""

    claims: List[dict] = Field(
        description="List of detected claims, each with claim_text, sentence_index, and importance"
    )


class ClaimUnderstandingAgentLangChain:
    """
    LangChain-based implementation of ClaimUnderstandingAgent.
    
    Uses LangChain's prompt templates, output parsers, and LLM chains
    for comparison with the custom implementation.
    
    Note: Context is passed explicitly (not via LangChain memory) because:
    - This is a one-shot operation, not a conversation
    - Context is the source document/article, not conversation history
    - Each claim normalization is independent
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3,  # Match custom implementation (was 0.2)
        provider: str = "openai",  # "openai" or "anthropic"
    ) -> None:
        """
        Initialize the LangChain-based agent.
        
        Args:
            model_name: Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022")
            temperature: Temperature for LLM calls
            provider: "openai" or "anthropic"
            
        Raises:
            ImportError: If LangChain packages are not installed
            RuntimeError: If required API keys are missing
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain packages are not installed. "
                "Install with: pip install langchain langchain-core langchain-openai langchain-anthropic"
            )

        config = get_config()

        # Determine provider and model
        if provider == "openai":
            api_key = getattr(config, "OPENAI_API_KEY", None)
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for LangChain OpenAI agent")
            self.llm = ChatOpenAI(
                model=model_name or "gpt-4o-mini",
                temperature=temperature,
                api_key=api_key,
                max_tokens=1000,  # Match custom implementation to prevent truncation
            )
        elif provider == "anthropic":
            api_key = getattr(config, "ANTHROPIC_API_KEY", None)
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for LangChain Anthropic agent")
            self.llm = ChatAnthropic(
                model=model_name or "claude-3-5-sonnet-20241022",
                temperature=temperature,
                api_key=api_key,
                max_tokens=1000,  # Match custom implementation to prevent truncation
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'")

        self.provider = provider

        # Build normalization chain
        self._build_normalization_chain()

        # Build detection chain
        self._build_detection_chain()

    def _build_normalization_chain(self):
        """Build the normalization chain using LangChain."""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a fact-checking assistant. Normalize the following claim to make it clear, concise, and fact-checkable.

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
- If the claim is too vague, rewrite it into the clearest possible version without adding new facts.""",
                ),
                (
                    "human",
                    """Claim: {claim_text}

{context}

Normalized claim:""",
                ),
            ]
        )

        self.normalization_chain = prompt_template | self.llm

    def _build_detection_chain(self):
        """Build the claim detection chain with structured output."""
        # Use output parser approach (more reliable than with_structured_output for nested schemas)
        # OpenAI's native structured output doesn't support nested schemas well
        output_parser = PydanticOutputParser(pydantic_object=ClaimDetectionOutput)
        self.use_structured_output = False

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a fact-checking assistant. Analyze the following text and identify factual claims that are worth fact-checking.

CRITICAL: Each claim must contain ONLY ONE atomic fact. If a sentence contains multiple facts, split them into separate claims.

IMPORTANT - Be AGGRESSIVE about splitting compound sentences:
- If a sentence contains multiple facts, create separate claims for EACH fact
- Split relative clauses, appositives, and compound statements
- Each claim should be independently verifiable
- Don't combine related facts into one claim

Examples:
- GOOD (atomic): "Saudi Arabia is the world's largest oil producer"
- GOOD (atomic): "Saudi Arabia aims to diversify its energy sources"
- BAD (compound): "Saudi Arabia, which is the world's largest oil producer, aims to diversify its energy sources"
  → MUST split into TWO separate claims:
    1. "Saudi Arabia is the world's largest oil producer"
    2. "Saudi Arabia aims to diversify its energy sources"
- BAD (compound): "Saudi Arabia has been working to diversify its economy through Vision 2030, a strategic framework launched in 2016"
  → MUST split into TWO separate claims:
    1. "Saudi Arabia has been working to diversify its economy through Vision 2030"
    2. "Vision 2030 is a strategic framework launched in 2016"

A check-worthy claim is:
- A factual assertion that can be verified
- Specific enough to be fact-checked (has dates, numbers, names, locations)
- Not an opinion or subjective statement
- Not a question
- Contains ONLY ONE atomic fact (not multiple facts combined)
- Even if related to another claim, split them if they are separate facts

For each claim you find, provide:
1. The exact claim text (one atomic fact only)
2. An importance score (0.0-1.0) indicating how check-worthy it is
3. The sentence index (0-based) where it appears

Importance scoring guidelines:
- 0.9-1.0: Highly check-worthy (specific numbers, dates, verifiable facts, major claims)
- 0.7-0.8: Moderately check-worthy (general factual statements, specific entities)
- 0.5-0.6: Somewhat check-worthy (less specific but verifiable claims)
- Below 0.5: Low priority (general statements, less verifiable)

Be generous with importance scores - only filter claims below the threshold AFTER scoring, not during scoring.

{format_instructions}""",
                ),
                (
                    "human",
                    """Text to analyze:
{text}

Only include claims with importance >= {min_importance}. 

Respond with a JSON object containing a "claims" array. Each claim should have:
- "claim_text": the claim text (one atomic fact only)
- "sentence_index": the sentence index (0-based) where it appears
- "importance": a float between 0.0 and 1.0

Output format:
{{
  "claims": [
    {{"claim_text": "...", "sentence_index": 0, "importance": 0.8}},
    ...
  ]
}}""",
                ),
            ]
        )

        # Use output parser approach (more reliable)
        self.detection_chain = prompt_template | self.llm | output_parser

    def normalize_claim(self, claim: Claim) -> Claim:
        """
        Basic rule-based normalization (same as custom implementation).
        
        This is kept for consistency and fallback purposes.
        """
        normalized = claim.raw_text.strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.lower()
        claim.normalized_text = normalized
        return claim

    async def normalize_claim_llm(
        self, claim: Claim, context: Optional[str] = None
    ) -> Claim:
        """
        Advanced LLM-based normalization using LangChain.
        
        Note: Context is passed explicitly (not via LangChain memory) because:
        - This is a one-shot operation, not a conversation
        - Context is the source document/article, not conversation history
        - Each claim normalization is independent
        
        Args:
            claim: The claim to normalize
            context: Optional context text (e.g., original article) to help resolve pronouns
                    without fact-correction.
            
        Returns:
            Claim with normalized_text set
        """
        context_instruction = ""
        if context:
            # Limit context length to avoid token limits
            context_instruction = (
                f"\n\nContext (for resolving pronouns only, DO NOT use to correct facts):\n"
                f"{context[:500]}"
            )

        try:
            # Invoke the chain
            response = await self.normalization_chain.ainvoke(
                {
                    "claim_text": claim.raw_text,
                    "context": context_instruction,
                }
            )

            # Extract content from LangChain message
            normalized_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            normalized_text = normalized_text.strip().strip('"').strip("'").strip()

            claim.normalized_text = normalized_text
            return claim
        except Exception as e:
            # Fallback to basic normalization if LLM fails
            return self.normalize_claim(claim)

    async def detect_claims(
        self,
        text: str,
        min_importance: float = 0.5,
        normalize_detected: bool = True,
    ) -> List[DetectedClaim]:
        """
        Detect check-worthy claims using LangChain structured output.
        
        Args:
            text: Input text to analyze
            min_importance: Minimum importance score threshold
            normalize_detected: Whether to normalize detected claims
            
        Returns:
            List of DetectedClaim objects, sorted by importance
        """
        # Split text into sentences for position tracking
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        try:
            # Get format instructions for the prompt
            output_parser = PydanticOutputParser(pydantic_object=ClaimDetectionOutput)
            format_instructions = output_parser.get_format_instructions()

            # Invoke the detection chain
            import os
            debug_mode = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
            import json
            
            try:
                result = await self.detection_chain.ainvoke(
                    {
                        "text": text,
                        "min_importance": min_importance,
                        "format_instructions": format_instructions,
                    }
                )
                
                # Debug logging
                if debug_mode:
                    print(f"🔍 Raw result type: {type(result)}")
                    if hasattr(result, "claims"):
                        print(f"🔍 Claims from parser: {len(result.claims)}")
                    print(f"🔍 Result preview: {str(result)[:300]}")
                    
            except Exception as chain_error:
                # Log chain invocation errors for debugging
                if debug_mode:
                    print(f"⚠️  Chain invocation error: {chain_error}")
                    import traceback
                    traceback.print_exc()
                # Try to continue with fallback parsing
                raise

            # Handle result - should be ClaimDetectionOutput object from parser
            
            if isinstance(result, ClaimDetectionOutput):
                claims_data = result.claims
                if debug_mode:
                    print(f"✅ Parsed as ClaimDetectionOutput: {len(claims_data)} claims")
            elif isinstance(result, dict) and "claims" in result:
                claims_data = result["claims"]
                if debug_mode:
                    print(f"✅ Parsed as dict with 'claims': {len(claims_data)} claims")
            elif hasattr(result, "claims"):
                claims_data = result.claims
                if debug_mode:
                    print(f"✅ Parsed from object.claims: {len(claims_data)} claims")
            else:
                # Fallback parsing - try multiple strategies (similar to Custom implementation)
                if debug_mode:
                    print(f"⚠️  Primary parsing failed, trying fallback. Result type: {type(result)}")
                
                if isinstance(result, str):
                    # Strategy 1: Try to extract JSON object with "claims" array
                    json_match = re.search(
                        r'\{[^}]*"claims"[^}]*\[.*?\]', result, re.DOTALL
                    )
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            claims_data = parsed.get("claims", [])
                            if debug_mode:
                                print(f"✅ Fallback 1 (claims object): {len(claims_data)} claims")
                        except json.JSONDecodeError:
                            claims_data = []
                    else:
                        # Strategy 2: Try parsing entire response as JSON
                        try:
                            parsed = json.loads(result)
                            claims_data = (
                                parsed.get("claims", [])
                                if isinstance(parsed, dict)
                                else []
                            )
                            if debug_mode:
                                print(f"✅ Fallback 2 (full JSON): {len(claims_data)} claims")
                        except json.JSONDecodeError:
                            # Strategy 3: Try to find array directly (like Custom implementation)
                            array_match = re.search(r'\[.*\]', result, re.DOTALL)
                            if array_match:
                                try:
                                    claims_data = json.loads(array_match.group())
                                    if debug_mode:
                                        print(f"✅ Fallback 3 (direct array): {len(claims_data)} claims")
                                except json.JSONDecodeError:
                                    claims_data = []
                            else:
                                if debug_mode:
                                    print(
                                        f"❌ All fallback strategies failed. Result preview: {result[:300]}"
                                    )
                                claims_data = []
                else:
                    # Try to convert to string and parse
                    result_str = str(result)
                    # Try same strategies as above
                    json_match = re.search(
                        r'\{[^}]*"claims"[^}]*\[.*?\]', result_str, re.DOTALL
                    )
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            claims_data = parsed.get("claims", [])
                            if debug_mode:
                                print(f"✅ Fallback (string conversion): {len(claims_data)} claims")
                        except json.JSONDecodeError:
                            claims_data = []
                    else:
                        # Try direct array extraction
                        array_match = re.search(r'\[.*\]', result_str, re.DOTALL)
                        if array_match:
                            try:
                                claims_data = json.loads(array_match.group())
                                if debug_mode:
                                    print(f"✅ Fallback (array from string): {len(claims_data)} claims")
                            except json.JSONDecodeError:
                                claims_data = []
                        else:
                            if debug_mode:
                                print(
                                    f"⚠️  Unexpected result type: {type(result)}, value: {str(result)[:200]}"
                                )
                            claims_data = []
            
            # Additional check: if we got very few claims, try one more fallback
            # by calling the LLM directly (bypassing parser) and using Custom's parsing method
            if len(claims_data) < 3 and len(sentences) > 3:
                if debug_mode:
                    print(f"⚠️  Only {len(claims_data)} claims detected, trying direct LLM call as final fallback...")
                
                # Try direct LLM call with simpler prompt (like Custom implementation)
                try:
                    direct_prompt = f"""You are a fact-checking assistant. Analyze the following text and identify factual claims that are worth fact-checking.

CRITICAL: Each claim must contain ONLY ONE atomic fact. Split compound sentences aggressively.

Text to analyze:
{text}

Respond with ONLY a JSON array of objects, each with:
- "claim_text": the claim text (one atomic fact only)
- "sentence_index": the sentence index (0-based)
- "importance": a float between 0.0 and 1.0

Output format:
[
  {{"claim_text": "...", "sentence_index": 0, "importance": 0.8}},
  ...
]

Only include claims with importance >= {min_importance}. Output ONLY the JSON array, no other text."""
                    
                    direct_response = await self.llm.ainvoke(direct_prompt)
                    direct_content = direct_response.content if hasattr(direct_response, "content") else str(direct_response)
                    
                    # Parse like Custom implementation
                    array_match = re.search(r'\[.*\]', direct_content, re.DOTALL)
                    if array_match:
                        try:
                            fallback_claims = json.loads(array_match.group())
                            if isinstance(fallback_claims, list) and len(fallback_claims) > len(claims_data):
                                if debug_mode:
                                    print(f"✅ Fallback direct LLM call found {len(fallback_claims)} claims (using {len(fallback_claims)} instead of {len(claims_data)})")
                                claims_data = fallback_claims
                        except json.JSONDecodeError:
                            pass
                except Exception as e:
                    if debug_mode:
                        print(f"⚠️  Direct LLM fallback failed: {e}")

            detected_claims: List[DetectedClaim] = []

            for claim_data in claims_data:
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

                # Find character positions
                start_char = None
                end_char = None
                if sentence_idx is not None and 0 <= sentence_idx < len(sentences):
                    sentence = sentences[sentence_idx]
                    start_char = text.find(sentence)
                    if start_char >= 0:
                        claim_start = sentence.find(claim_text)
                        if claim_start >= 0:
                            start_char += claim_start
                            end_char = start_char + len(claim_text)

                detected_claim = DetectedClaim(
                    id=f"detected-lc-{uuid.uuid4().hex[:8]}",
                    raw_text=claim_text,
                    normalized_text=None,
                    start_char=start_char,
                    end_char=end_char,
                    sentence_index=sentence_idx,
                    importance=importance,
                    metadata={
                        "detection_method": "langchain",
                        "provider": self.provider,
                    },
                )
                detected_claims.append(detected_claim)

            # Normalize detected claims if requested
            if normalize_detected:
                for detected_claim in detected_claims:
                    claim_obj = Claim(id=detected_claim.id, raw_text=detected_claim.raw_text)
                    try:
                        claim_obj = await self.normalize_claim_llm(claim_obj, context=text)
                    except Exception:
                        claim_obj = self.normalize_claim(claim_obj)
                    detected_claim.normalized_text = claim_obj.normalized_text

            # Sort by importance
            detected_claims.sort(key=lambda c: c.importance or 0.0, reverse=True)

            return detected_claims

        except Exception as e:
            # Log the error for debugging
            import traceback

            print(f"❌ Error in detect_claims: {e}")
            print(f"   Error type: {type(e).__name__}")
            # Only print full traceback in debug mode (can be verbose)
            import os

            if os.getenv("DEBUG", "").lower() in ("1", "true", "yes"):
                print(f"   Traceback: {traceback.format_exc()}")
            return []

