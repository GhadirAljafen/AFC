# Claim Normalization: Fact Preservation Issue and Resolution

## Problem Statement

During the implementation of the `ClaimUnderstandingAgent`, we encountered a critical issue where the LLM-based normalization was **incorrectly correcting facts** instead of preserving the claim as stated.

### Example of the Issue

**Input Text:**
```
Saudi Arabia announced plans to invest $100 billion in renewable energy by 2030, 
according to a statement from the Ministry of Energy. The country, which is currently 
the world's largest oil producer, aims to diversify its energy sources...
```

**Detected Claim (Raw):**
```
The country, which is currently the world's largest oil producer, aims to diversify 
its energy sources and reduce carbon emissions.
```

**Incorrect Normalization:**
```
The United States is the world's largest oil producer and aims to diversify its energy 
sources and reduce carbon emissions.
```

**Expected Normalization:**
```
Saudi Arabia is the world's largest oil producer and aims to diversify its energy 
sources and reduce carbon emissions.
```

### Root Cause

The LLM was using its internal knowledge to "correct" the claim, replacing "The country" (referring to Saudi Arabia from context) with "United States" because it knows that the United States is actually the world's largest oil producer. This violated the core principle of claim normalization: **preserve the claim as stated, only clean up the language**.

## Solution

We implemented a two-part solution:

### 1. Enhanced Prompt with Explicit Fact-Preservation Rules

We updated the `normalize_claim_llm()` method to include explicit instructions that prevent fact-correction:

```python
CRITICAL RULES:
- DO NOT correct or change any facts, even if you know they are incorrect
- DO NOT replace entities with what you believe are the "correct" entities
- PRESERVE the claim exactly as stated, only clean up the language
- Your job is to normalize the LANGUAGE, not to fact-check or correct the claim
```

### 2. Context-Aware Normalization

We added a `context` parameter to `normalize_claim_llm()` that allows passing the original text as context:

```python
async def normalize_claim_llm(self, claim: Claim, context: Optional[str] = None) -> Claim:
    """
    Args:
        claim: The claim to normalize.
        context: Optional context text (e.g., original article) to help resolve pronouns
                without fact-correction.
    """
```

When detecting claims, the original text is automatically passed as context:

```python
# In detect_claims() method
claim_obj = await self.normalize_claim_llm(claim_obj, context=text)
```

This enables the LLM to:
- **Resolve pronouns** from context (e.g., "The country" → "Saudi Arabia")
- **Preserve facts** as stated in the original claim
- **Avoid fact-correction** by using context only for pronoun resolution, not for "correcting" entities

## Implementation Details

### Updated Normalization Prompt

The prompt now includes:

1. **CRITICAL RULES section** - Explicitly forbids fact-correction
2. **Context instruction** - When context is provided, it's clearly marked as "for resolving pronouns only, DO NOT use to correct facts"
3. **Normalization Rules** - Emphasizes preserving entities while cleaning up language

### Key Changes in Code

**File:** `src/factcheck_agent/agents/claim_understanding.py`

1. **Method signature updated:**
   ```python
   async def normalize_claim_llm(self, claim: Claim, context: Optional[str] = None) -> Claim:
   ```

2. **Context handling:**
   ```python
   context_instruction = ""
   if context:
       context_instruction = f"\n\nContext (for resolving pronouns only, DO NOT use to correct facts):\n{context[:500]}"
   ```

3. **Automatic context passing in detect_claims():**
   ```python
   claim_obj = await self.normalize_claim_llm(claim_obj, context=text)
   ```

## Results

After implementing the solution:

✅ **Pronouns are correctly resolved** from context (e.g., "The country" → "Saudi Arabia")  
✅ **Facts are preserved** as stated in the original claim  
✅ **No fact-correction** occurs, even when the LLM knows different information  
✅ **Claims remain fact-checkable** while being properly normalized

### Example of Correct Behavior

**Input Text:**
```
Saudi Arabia announced plans to invest $100 billion in renewable energy by 2030. 
The country, which is currently the world's largest oil producer, aims to diversify 
its energy sources...
```

**Detected Claim:**
```
The country, which is currently the world's largest oil producer, aims to diversify 
its energy sources and reduce carbon emissions.
```

**Correct Normalization:**
```
Saudi Arabia is the world's largest oil producer and aims to diversify its energy 
sources and reduce carbon emissions.
```

The normalization:
- ✅ Resolved "The country" to "Saudi Arabia" (from context)
- ✅ Preserved the claim that Saudi Arabia is the largest oil producer (as stated)
- ✅ Did NOT change it to "United States" (even though that might be factually correct)

## Best Practices

When using `normalize_claim_llm()`:

1. **Always provide context** when normalizing detected claims to enable pronoun resolution
2. **Trust the prompt** - The explicit rules prevent fact-correction
3. **Test with edge cases** - Claims that might be factually incorrect should still be preserved
4. **Monitor normalization output** - Ensure claims are being preserved, not corrected

## Related Files

- `src/factcheck_agent/agents/claim_understanding.py` - Main implementation
- `src/factcheck_agent/models.py` - `DetectedClaim` model with `normalized_text` field
- `notebooks/claim_understanding_agent_demo.ipynb` - Demo notebook with examples

## Future Considerations

Potential improvements:
- Add validation to detect if normalization changed factual content
- Implement a confidence score for pronoun resolution
- Add logging to track when context is used vs. not used
- Consider fine-tuning a model specifically for claim normalization

## Summary

The issue was resolved by:
1. Adding explicit fact-preservation rules to the normalization prompt
2. Implementing context-aware normalization to resolve pronouns without fact-correction
3. Automatically passing original text as context when normalizing detected claims

This ensures that claim normalization focuses on **language cleanup** rather than **fact correction**, which is essential for accurate fact-checking pipelines.






