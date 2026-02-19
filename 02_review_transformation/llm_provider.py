"""
LLM Provider abstraction for multiple providers.

Supports:
- OpenAI (GPT-4o, GPT-4-turbo, etc.)
- AWS Bedrock (Claude 3.5 Sonnet)
- Google Gemini
- Together AI
- OpenRouter (meta-llama/llama-4-scout, Claude via Anthropic, etc.)
- Anthropic (Claude Haiku 4.5, Sonnet, etc. via native API)
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.temperature = kwargs.get('temperature', 0.7)
        self.timeout = kwargs.get('timeout', 60)
        self.max_retries = kwargs.get('max_retries', 3)
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Optional override for max tokens (uses instance default if None)
            
        Returns:
            Generated text
        """
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI provider using OpenAI API."""
    
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI provider. "
                "Install with: pip install openai"
            )
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        # GPT-5 and newer models (like o1, o3, o4) use max_completion_tokens and do not support temperature
        model_lower = (self.model or "").lower()
        is_gpt5_style = "gpt-5" in model_lower or model_lower.startswith("o1-") or model_lower.startswith("o3-") or model_lower.startswith("o4-") or model_lower == "o1" or model_lower == "o3" or model_lower == "o4"
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.timeout,
        }
        if is_gpt5_style:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = self.temperature

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")


class BedrockProvider(LLMProvider):
    """AWS Bedrock provider for Claude models."""
    
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 package is required for Bedrock provider. "
                "Install with: pip install boto3"
            )
        
        # Check for Bedrock API key (preferred) or AWS credentials
        bearer_token = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_REGION', 'us-east-1')
        
        if bearer_token:
            # Use bearer token authentication
            import requests
            self.bedrock_runtime = None
            self.bearer_token = bearer_token
            self.region = region
            self.use_bearer = True
        elif access_key and secret_key:
            # Use AWS credentials
            self.bedrock_runtime = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
            self.use_bearer = False
        else:
            raise ValueError(
                "AWS credentials not found. Please set either:\n"
                "  - AWS_BEARER_TOKEN_BEDROCK (preferred), or\n"
                "  - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION\n"
                "in your .env file."
            )
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        
        # Prepare request body
        import json
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            if self.use_bearer:
                # Use bearer token API
                import requests
                url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model}/invoke"
                headers = {
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Content-Type": "application/json"
                }
                response = requests.post(
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
            else:
                # Use boto3
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model,
                    body=json.dumps(request_body)
                )
                result = json.loads(response['body'].read())
            
            # Extract text from Claude response
            if 'content' in result and len(result['content']) > 0:
                return result['content'][0]['text']
            else:
                raise Exception("Empty response from Bedrock")
        except Exception as e:
            raise Exception(f"Bedrock API error: {e}")


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""
    
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            # Use the newer google-genai package (simplified approach matching official example)
            from google import genai
            self.genai_module = genai
            self.use_new_api = True
        except ImportError:
            try:
                # Fall back to older google-generativeai
                import google.generativeai as genai
                self.genai_module = genai
                self.use_new_api = False
            except ImportError:
                raise ImportError(
                    "google-genai package is required for Google provider. "
                    "Install with: pip install google-genai"
                )
        
        # Check for API key - Google uses GEMINI_API_KEY (or GOOGLE_API_KEY as fallback)
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        if self.use_new_api:
            # Simple client creation - matches Google's official example
            # The Client() constructor reads from GEMINI_API_KEY env var automatically
            # Ensure it's set in environment
            os.environ['GEMINI_API_KEY'] = api_key
            self.client = self.genai_module.Client()
            self.model_name = model
        else:
            # google-generativeai: use configure method
            self.genai_module.configure(api_key=api_key)
            self.client = self.genai_module.GenerativeModel(model_name=self.model)
            self.model_name = None  # Not needed for old API
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        # Retry logic for network errors
        max_retries = self.max_retries
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if self.use_new_api:
                    # google-genai: use client.models.generate_content (simplified)
                    # Build generation config - don't set max_output_tokens to allow full generation
                    generation_config = {
                        "temperature": self.temperature,
                    }
                    # Note: Not setting max_output_tokens allows the model to generate longer responses
                    # This prevents truncation issues with Google Gemini Flash
                    
                    # Simple API call - matches Google's official example
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=generation_config
                    )
                    
                    # Extract text from response - use _get_text() method (most reliable)
                    text = None
                    
                    # Primary method: _get_text() (internal method that properly extracts text)
                    if hasattr(response, '_get_text'):
                        try:
                            text = response._get_text()
                            if text:
                                text = str(text).strip()
                        except Exception:
                            pass
                    
                    # Fallback 1: Try response.text property
                    if not text and hasattr(response, 'text'):
                        try:
                            text_val = response.text
                            if text_val:
                                text = str(text_val).strip()
                        except Exception:
                            pass
                    
                    # Fallback 2: Extract from candidates structure
                    if not text and hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                content = candidate.content
                                
                                # Try parts
                                if hasattr(content, 'parts') and content.parts:
                                    for part in content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            text = str(part.text).strip()
                                            break
                                
                                # Try direct text on content
                                if not text and hasattr(content, 'text') and content.text:
                                    text = str(content.text).strip()
                            
                            # Try direct text on candidate
                            if not text and hasattr(candidate, 'text') and candidate.text:
                                text = str(candidate.text).strip()
                            
                            if text:
                                break
                    
                    if not text or not text.strip():
                        raise Exception("Google Gemini API returned empty text. Response may be blocked or invalid.")
                    return text
                else:
                    # google-generativeai: use generate_content method
                    generation_config = {
                        "temperature": self.temperature,
                    }
                    # Note: Not setting max_output_tokens allows the model to generate longer responses
                    # This prevents truncation issues with Google Gemini Flash
                    
                    response = self.client.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    return response.text
                    
            except Exception as e:
                # Check if it's a network/timeout error that we should retry
                error_str = str(e).lower()
                is_network_error = any(keyword in error_str for keyword in ['timeout', 'connection', 'ssl', 'network', 'socket', 'broken pipe'])
                
                if is_network_error and attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                    print(f"  Network error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    last_exception = e
                    continue
                else:
                    # Not retryable or out of retries
                    raise Exception(f"Google Gemini API error: {e}")
        
        # If we get here, all retries failed
        if last_exception:
            raise Exception(f"Google Gemini API error after {max_retries} attempts: {last_exception}")
        else:
            raise Exception("Google Gemini API error: Unknown error")


class TogetherProvider(LLMProvider):
    """Together AI provider."""
    
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package is required for Together AI provider. "
                "Install with: pip install requests"
            )
        
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        self.api_key = api_key
        self.api_url = "https://api.together.xyz/v1/chat/completions"
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        import requests
        
        max_tokens = max_tokens or self.max_tokens
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                raise Exception("Empty response from Together AI")
        except Exception as e:
            raise Exception(f"Together AI API error: {e}")


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider (OpenAI-compatible API). Supports many models including meta-llama/llama-4-scout, Anthropic Claude, etc."""

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package is required for OpenRouter provider. "
                "Install with: pip install requests"
            )

        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Please set it in your .env file. Get a key at https://openrouter.ai/keys"
            )

        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        import requests

        max_tokens = max_tokens or self.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                raise Exception("Empty response from OpenRouter")
        except Exception as e:
            raise Exception(f"OpenRouter API error: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic provider for Claude models (e.g. Claude Haiku 4.5) via native API."""

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic provider. "
                "Install with: pip install anthropic"
            )

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables. "
                "Please set it in your .env file. Get a key at https://console.anthropic.com/settings/keys"
            )

        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_tokens

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                raise Exception("Empty response from Anthropic")
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")


def create_provider(provider_name: str, model: str, **kwargs) -> LLMProvider:
    """
    Create an LLM provider instance.
    
    Args:
        provider_name: Name of the provider ('openai', 'bedrock', 'google', 'together', 'openrouter', 'anthropic')
        model: Model identifier
        **kwargs: Additional configuration (max_tokens, temperature, timeout, max_retries)
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If provider_name is not supported
        ImportError: If required package is not installed
        ValueError: If API key is missing
    """
    provider_name_lower = provider_name.lower()
    
    if provider_name_lower == 'openai':
        return OpenAIProvider(model, **kwargs)
    elif provider_name_lower == 'bedrock':
        return BedrockProvider(model, **kwargs)
    elif provider_name_lower == 'google':
        return GoogleProvider(model, **kwargs)
    elif provider_name_lower == 'together':
        return TogetherProvider(model, **kwargs)
    elif provider_name_lower == 'openrouter':
        return OpenRouterProvider(model, **kwargs)
    elif provider_name_lower == 'anthropic':
        return AnthropicProvider(model, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported providers: openai, bedrock, google, together, openrouter, anthropic"
        )

