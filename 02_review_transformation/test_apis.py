#!/usr/bin/env python3
"""
Test script to verify all enabled LLM APIs are working correctly.

This script tests all LLM providers configured in config.yaml by sending
a simple test prompt to each enabled provider/model combination.

Run from PeerPrism repo root: python 01_review_transformation/test_apis.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from utils import load_config
from utils.config_loader import get_llms, get_api_config
from llm_provider import create_provider


def test_provider(provider_name: str, model: str, api_config: dict) -> tuple[bool, str]:
    """
    Test a single LLM provider with a simple prompt.
    
    Args:
        provider_name: Name of the provider
        model: Model identifier
        api_config: API configuration dictionary
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Use a simple, safe prompt that shouldn't trigger filters
    test_prompt = "Hello"
    
    try:
        print(f"  Testing {provider_name}/{model}...", end=" ", flush=True)
        
        provider = create_provider(
            provider_name=provider_name,
            model=model,
            **api_config
        )
        
        # For Google Gemini, use default config (low max_tokens values cause API issues)
        # For others, use a small max_tokens for testing
        if provider_name == 'google':
            response = provider.generate(test_prompt)
        else:
            response = provider.generate(test_prompt, max_tokens=50)
        
        # Basic validation - check if we got a response
        if response and len(response.strip()) > 0:
            print("✓")
            return True, f"Success: Got response ({len(response)} chars)"
        else:
            print("✗")
            return False, "Error: Empty response"
            
    except ValueError as e:
        # API key missing
        print("✗")
        return False, f"Error: {str(e)}"
    except ImportError as e:
        # Package not installed
        print("✗")
        return False, f"Error: {str(e)}"
    except Exception as e:
        print("✗")
        return False, f"Error: {type(e).__name__}: {str(e)}"


def main():
    """Main function to test all enabled APIs."""
    print("=" * 60)
    print("LLM API Test Script")
    print("=" * 60)
    
    # Load configuration
    print("\nLoading configuration...")
    try:
        config = load_config()
        llm_configs = get_llms(config)
        api_config = get_api_config(config)
        
        if not llm_configs:
            print("  ⚠ No LLM providers configured in config.yaml")
            print("  Please enable at least one provider in config.yaml")
            return
        
        print(f"  Found {len(llm_configs)} LLM provider(s) configured")
        
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return
    
    # Test each provider
    print("\nTesting LLM providers...")
    print("-" * 60)
    
    results = []
    for llm_config in llm_configs:
        provider_name = llm_config['provider']
        model = llm_config['model']
        
        success, message = test_provider(provider_name, model, api_config)
        results.append({
            'provider': provider_name,
            'model': model,
            'success': success,
            'message': message
        })
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"\n{status} - {result['provider']}/{result['model']}")
        print(f"  {result['message']}")
    
    print("\n" + "-" * 60)
    print(f"Total: {successful}/{total} providers working correctly")
    
    if successful == total:
        print("✓ All enabled APIs are working!")
        return 0
    else:
        print("✗ Some APIs failed. Check the error messages above.")
        print("\nCommon issues:")
        print("  - Missing API keys in .env file")
        print("  - Missing Python packages (run: pip install -r requirements.txt)")
        print("  - Invalid API keys")
        print("  - Network/API issues")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

