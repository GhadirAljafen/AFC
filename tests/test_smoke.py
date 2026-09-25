"""Smoke tests to verify basic package functionality."""

import pytest


def test_import_package():
    """Test that the package can be imported."""
    import factcheck_agent
    assert factcheck_agent.__version__ == "0.1.0"


def test_import_modules():
    """Test that main modules can be imported."""
    from factcheck_agent import config
    from factcheck_agent import models
    from factcheck_agent import llm_client
    from factcheck_agent import pipeline
    from factcheck_agent import cli
    
    assert config is not None
    assert models is not None
    assert llm_client is not None
    assert pipeline is not None
    assert cli is not None


def test_config():
    """Test that config can be instantiated."""
    from factcheck_agent.config import get_config
    
    config = get_config()
    assert config is not None
    assert isinstance(config.LOG_LEVEL, str)
    assert isinstance(config.MAX_EVIDENCE_SOURCES, int)


def test_models():
    """Test that models can be instantiated."""
    from factcheck_agent.models import Claim, Verdict, Evidence, FactCheckResult
    
    claim = Claim(text="Test claim")
    assert claim.text == "Test claim"
    
    verdict = Verdict.TRUE
    assert verdict.value == "true"
    
    evidence = Evidence(source="test", content="test content")
    assert evidence.source == "test"
    
    result = FactCheckResult(
        claim=claim,
        verdict=verdict,
        explanation="Test explanation",
        evidence=[evidence]
    )
    assert result.claim == claim
    assert result.verdict == verdict

