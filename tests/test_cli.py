"""Tests for the CLI entry point."""

import sys
from unittest.mock import patch, MagicMock

from factcheck_agent import cli
from factcheck_agent.models import Claim, FactCheckResult, FactCheckVerdict, EvidenceSnippet


def test_cli_main_exists():
    """Test that main function exists and is callable."""
    assert hasattr(cli, "main")
    assert callable(cli.main)


def test_cli_main_without_args():
    """Test that CLI shows help when no claim is provided."""
    result = cli.main([])
    assert result == 1


def test_cli_main_with_claim():
    """Test that CLI processes a claim successfully."""
    test_claim = "The Earth orbits the Sun."
    
    # Create a mock result
    mock_result = FactCheckResult(
        claim=Claim(id="test", raw_text=test_claim),
        verdict=FactCheckVerdict(
            label="NOT_ENOUGH_INFO",
            confidence=0.2,
            used_evidence_ids=[],
        ),
        explanation="Test explanation",
        evidence=[],
    )
    
    # Mock the pipeline to avoid actual processing
    with patch("factcheck_agent.cli.FactCheckingPipeline") as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = mock_result
        mock_pipeline_class.return_value = mock_pipeline
        
        # Mock print to avoid cluttering test output
        with patch("builtins.print"):
            result = cli.main([test_claim])
            assert result == 0
            mock_pipeline.process.assert_called_once()


def test_dummy_llm_client():
    """Test that DummyLLMClient can be instantiated."""
    llm = cli.DummyLLMClient()
    assert isinstance(llm, cli.DummyLLMClient)

