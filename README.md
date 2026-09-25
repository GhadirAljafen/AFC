# Fact-Check Agent

An agentic automated fact-checking system that verifies claims through a structured pipeline.

## Overview

This project implements an automated fact-checking system that uses AI agents to verify claims by retrieving evidence, analyzing sources, and generating verdicts with explanations.

## Pipeline Stages

The fact-checking pipeline follows these stages:

1. **Claim** - Input claim to be verified
2. **Retrieval** - Search and retrieve relevant evidence from multiple sources
3. **Evidence** - Analyze and evaluate the quality and relevance of evidence
4. **Verdict** - Determine the truthfulness of the claim (True, False, Partially True, Unverifiable)
5. **Explanation** - Generate a clear explanation of the verdict with supporting evidence

## Installation

```bash
# Install dependencies
poetry install

# Or with pip
pip install -e .
```

## Quick Start

1. **Set up API keys** (see `setup_guide.md` for details):
   ```bash
   export OPENAI_API_KEY=your_key_here
   export GOOGLE_SEARCH_API_KEY=your_key_here
   export GOOGLE_SEARCH_ENGINE_ID=your_engine_id
   ```

2. **Verify setup:**
   ```bash
   python scripts/verify_setup.py
   ```

3. **Run fact-checking:**
   ```bash
   factcheck "Saudi Arabia is the largest oil producer in the world."
   ```

## Usage

```bash
# Basic usage
factcheck "Your claim here"

# With source
factcheck "The Earth is flat." --source "social_media"
```

## Development

```bash
# Run tests
pytest

# Format code
black src tests

# Lint code
ruff check src tests
```

## Project Structure

```
src/factcheck_agent/
├── __init__.py
├── config.py          # Configuration management
├── models.py          # Data models
├── llm_client.py      # LLM client wrapper
├── pipeline.py        # Main fact-checking pipeline
├── cli.py             # Command-line interface
└── agents/            # Agent implementations
    └── __init__.py
```

## License

MIT

