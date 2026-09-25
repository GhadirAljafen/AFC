#!/usr/bin/env python3
"""Verify that all required API keys and dependencies are configured."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factcheck_agent.config import get_config

def check_dependencies():
    """Check if required Python packages are installed."""
    print("📦 Checking dependencies...")
    
    required = {
        "pydantic": "pydantic",
        "httpx": "httpx",
        "openai": "openai",
        "dotenv": "python-dotenv",
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (install with: pip install {package})")
            missing.append(package)
    
    return len(missing) == 0

def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 Checking API keys...")
    
    config = get_config()
    
    # Check LLM keys
    has_openai = bool(getattr(config, "OPENAI_API_KEY", None))
    has_anthropic = bool(getattr(config, "ANTHROPIC_API_KEY", None))
    
    if has_openai:
        print("  ✅ OPENAI_API_KEY")
    else:
        print("  ❌ OPENAI_API_KEY (not set)")
    
    if has_anthropic:
        print("  ✅ ANTHROPIC_API_KEY")
    else:
        print("  ❌ ANTHROPIC_API_KEY (not set)")
    
    if not (has_openai or has_anthropic):
        print("\n  ⚠️  No LLM API key found! Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return False
    
    # Check Google Search
    has_google_key = bool(getattr(config, "GOOGLE_SEARCH_API_KEY", None))
    has_google_id = bool(getattr(config, "GOOGLE_SEARCH_ENGINE_ID", None))
    
    if has_google_key and has_google_id:
        print("  ✅ GOOGLE_SEARCH_API_KEY")
        print("  ✅ GOOGLE_SEARCH_ENGINE_ID")
    else:
        print("  ⚠️  GOOGLE_SEARCH_API_KEY (not set - web search will be limited)")
        print("  ⚠️  GOOGLE_SEARCH_ENGINE_ID (not set - web search will be limited)")
    
    return True

def main():
    """Run all checks."""
    print("=" * 60)
    print("🔍 Fact-Check Agent Setup Verification")
    print("=" * 60)
    
    deps_ok = check_dependencies()
    keys_ok = check_api_keys()
    
    print("\n" + "=" * 60)
    if deps_ok and keys_ok:
        print("✅ Setup looks good! You're ready to fact-check.")
        return 0
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        print("\nSee setup_guide.md for detailed instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

