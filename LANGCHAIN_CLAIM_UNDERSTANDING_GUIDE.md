# LangChain vs Custom ClaimUnderstandingAgent: Implementation Guide

## Overview

This document describes the LangChain-based implementation of `ClaimUnderstandingAgent` and how to compare it with the custom implementation.

## Installation

### Install LangChain Dependencies

```bash
pip install langchain langchain-core langchain-openai langchain-anthropic
```

Or using Poetry:

```bash
poetry add langchain langchain-core langchain-openai langchain-anthropic
```

## Implementation Details

### File Structure

- **Custom Implementation**: `src/factcheck_agent/agents/claim_understanding.py`
- **LangChain Implementation**: `src/factcheck_agent/agents/claim_understanding_langchain.py`

### Key Differences

#### 1. **Chain Types Used**

- **Normalization**: `ChatPromptTemplate` + `ChatOpenAI`/`ChatAnthropic` (using `|` operator for composition)
- **Detection**: `ChatPromptTemplate` + `PydanticOutputParser` or `with_structured_output()` for structured JSON output

#### 2. **Context Handling**

Both implementations pass context **explicitly** (not via LangChain memory) because:
- This is a one-shot operation, not a conversation
- Context is the source document/article, not conversation history
- Each claim normalization is independent

#### 3. **Structured Output**

The LangChain version uses:
- `PydanticOutputParser` for claim detection (fallback)
- `with_structured_output()` if available (preferred, newer LangChain versions)

## Usage

### Basic Usage

```python
from factcheck_agent.models import Claim
from factcheck_agent.agents.claim_understanding_langchain import ClaimUnderstandingAgentLangChain

# Initialize LangChain agent
agent = ClaimUnderstandingAgentLangChain(
    provider="openai",  # or "anthropic"
    model_name="gpt-4o-mini",  # optional
    temperature=0.2
)

# Normalize a claim
claim = Claim(id="1", raw_text="I heard that Saudi Arabia is the largest oil producer")
normalized = await agent.normalize_claim_llm(claim)
print(normalized.normalized_text)

# Detect claims from text
text = "Saudi Arabia is the world's largest oil producer. The country aims to diversify."
claims = await agent.detect_claims(text, min_importance=0.5)
for claim in claims:
    print(f"{claim.raw_text} (importance: {claim.importance:.2f})")
```

### Comparison with Custom Implementation

```python
from factcheck_agent.agents.claim_understanding import ClaimUnderstandingAgent
from factcheck_agent.agents.claim_understanding_langchain import ClaimUnderstandingAgentLangChain
from factcheck_agent.llm_client import get_default_llm_client

# Initialize both
llm = get_default_llm_client(use_dummy_if_missing_key=False)
custom_agent = ClaimUnderstandingAgent(llm=llm)
langchain_agent = ClaimUnderstandingAgentLangChain(provider="openai")

# Compare normalization
claim = Claim(id="1", raw_text="I heard that...")
custom_result = await custom_agent.normalize_claim_llm(claim)
langchain_result = await langchain_agent.normalize_claim_llm(claim)

print(f"Custom:    {custom_result.normalized_text}")
print(f"LangChain: {langchain_result.normalized_text}")
```

## Running Comparisons

### Quick Comparison Test

```bash
python -m pytest tests/test_claim_understanding_comparison.py -v
```

Or run directly:

```bash
python tests/test_claim_understanding_comparison.py
```

### Benchmark Performance

```bash
python scripts/benchmark_claim_understanding.py
```

This will:
- Compare normalization accuracy and speed
- Compare detection accuracy and speed
- Save results to `benchmark_results.json`

## Comparison Metrics

### 1. **Accuracy**
- Normalization: Text similarity between outputs
- Detection: Number of claims detected, overlap in detected claims

### 2. **Performance**
- Latency: Time to complete operations
- Token usage: Cost comparison (if tracked)

### 3. **Reliability**
- Error handling and fallbacks
- Consistency across runs

### 4. **Code Quality**
- Maintainability
- Extensibility
- Debugging ease

## Expected Results

### Normalization
- **Similarity**: Typically 80-95% (same core meaning, may differ in phrasing)
- **Performance**: Similar latency (both use same LLM API)

### Detection
- **Count**: May differ slightly due to different prompt engineering
- **Quality**: Both should detect similar check-worthy claims
- **Performance**: LangChain may have slight overhead due to parsing

## Advantages of Each Approach

### Custom Implementation
- ✅ **Lightweight**: No additional dependencies
- ✅ **Direct control**: Full control over prompts and parsing
- ✅ **Simple**: Easier to understand and debug
- ✅ **Flexible**: Easy to customize for specific needs

### LangChain Implementation
- ✅ **Standardized**: Uses LangChain patterns
- ✅ **Structured output**: Built-in Pydantic validation
- ✅ **Extensible**: Easy to add chains, tools, memory
- ✅ **Ecosystem**: Integrates with other LangChain components

## When to Use Which?

### Use Custom Implementation When:
- You want minimal dependencies
- You need fine-grained control
- You're building a standalone system
- Performance is critical

### Use LangChain Implementation When:
- You're building a larger LangChain-based system
- You want structured output validation
- You plan to integrate with other LangChain tools
- You want to leverage LangChain's ecosystem

## Troubleshooting

### ImportError: LangChain packages not installed

```bash
pip install langchain langchain-core langchain-openai langchain-anthropic
```

### RuntimeError: API key missing

Ensure your `.env` file contains:
```
OPENAI_API_KEY=your_key_here
# or
ANTHROPIC_API_KEY=your_key_here
```

### Structured Output Not Working

The implementation falls back to `PydanticOutputParser` if `with_structured_output()` is not available. This is normal for older LangChain versions.

## Next Steps

1. Run the comparison tests to see differences
2. Run benchmarks to measure performance
3. Evaluate which approach fits your use case
4. Consider using both (custom for production, LangChain for experimentation)

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Structured Output](https://python.langchain.com/docs/modules/model_io/output_parsers/structured)
- [Custom Implementation](../src/factcheck_agent/agents/claim_understanding.py)

