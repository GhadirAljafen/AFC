# Setup Guide for Real Fact-Checking System

This guide will help you set up the fact-checking system to work with real APIs (no dummies or static data).

## Required API Keys

### 1. LLM Provider (Required - Choose One)

#### Option A: OpenAI
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Set environment variable:
   ```bash
   export OPENAI_API_KEY=sk-your-key-here
   ```

#### Option B: Anthropic
1. Go to https://console.anthropic.com/
2. Create an API key
3. Set environment variable:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

### 2. Google Custom Search (Required for Web Search)

1. **Get Google Custom Search API Key:**
   - Go to https://console.cloud.google.com/
   - Create a new project (or select existing)
   - Enable "Custom Search API"
   - Go to "Credentials" → "Create Credentials" → "API Key"
   - Copy your API key

2. **Create Custom Search Engine:**
   - Go to https://programmablesearchengine.google.com/
   - Click "Add" to create a new search engine
   - Enter any website (e.g., `*` to search entire web)
   - Click "Create"
   - Go to "Control Panel" → "Setup" → "Basics"
   - Copy your "Search engine ID"

3. **Set Environment Variables:**
   ```bash
   export GOOGLE_SEARCH_API_KEY=your_api_key_here
   export GOOGLE_SEARCH_ENGINE_ID=your_engine_id_here
   ```

## Installation

1. **Install Python Dependencies:**
   ```bash
   pip install pydantic python-dotenv httpx openai
   ```

2. **Create .env File (Optional but Recommended):**
   ```bash
   # .env file
   OPENAI_API_KEY=sk-your-key-here
   GOOGLE_SEARCH_API_KEY=your-api-key
   GOOGLE_SEARCH_ENGINE_ID=your-engine-id
   ```

## Usage

### Command Line Interface

```bash
# Basic usage
factcheck "Saudi Arabia is the largest oil producer in the world."

# With source
factcheck "The Earth is flat." --source "social_media"
```

### Python API

```python
import asyncio
from factcheck_agent.llm_client import get_default_llm_client
from factcheck_agent.pipeline import FactCheckingPipeline
from factcheck_agent.models import Claim

# Get real LLM (will fail if no API key)
llm = get_default_llm_client(use_dummy_if_missing_key=False)

# Create pipeline
pipeline = FactCheckingPipeline(llm=llm)

# Fact-check a claim
claim = Claim(id="test-1", raw_text="Saudi Arabia is the largest oil producer.")
result = asyncio.run(pipeline.process(claim))

print(f"Verdict: {result.verdict.label}")
print(f"Confidence: {result.verdict.confidence:.0%}")
print(f"Explanation: {result.explanation}")
```

## Verification

Test that everything is configured correctly:

```bash
# Check if API keys are set
python -c "from factcheck_agent.config import get_config; c = get_config(); print('OpenAI:', '✅' if c.OPENAI_API_KEY else '❌'); print('Google Search:', '✅' if c.GOOGLE_SEARCH_API_KEY and c.GOOGLE_SEARCH_ENGINE_ID else '❌')"

# Test the CLI
factcheck "Test claim"
```

## Troubleshooting

### "No LLM API key provided"
- Make sure you've set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- Check that your `.env` file is in the project root
- Verify the key is correct (no extra spaces)

### "Google Search API is not configured"
- This is a warning, not an error
- The system will work but won't retrieve web evidence
- Set `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` for full functionality

### "httpx package is required"
- Install with: `pip install httpx`

### API Errors (401, 403, etc.)
- Check that your API keys are valid
- Verify you have credits/quota available
- For OpenAI: Check https://platform.openai.com/usage
- For Google: Check https://console.cloud.google.com/apis/credentials

## Cost Considerations

- **OpenAI API:** Pay per token (~$0.01-0.06 per 1000 tokens)
- **Google Custom Search:** Free tier: 100 queries/day, then $5 per 1000 queries
- **Anthropic:** Check pricing at https://www.anthropic.com/pricing

Each fact-check typically uses:
- 1-3 LLM calls (normalization, verdict, explanation)
- 1-3 Google Search queries (evidence retrieval)
- Total cost: ~$0.01-0.10 per fact-check

