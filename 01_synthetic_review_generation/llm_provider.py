"""
LLM Provider abstraction for multiple providers.

Supports:
- OpenAI (GPT-5, GPT-O4, etc.) via OpenAI Responses API
- AWS Bedrock (Claude on Bedrock)  [existing]
- Google Gemini (google-genai / google-generativeai)  [existing]
- Together AI  [existing]
- Anthropic (Claude direct API)  [NEW]
- OpenRouter (Llama / DeepSeek / other OpenAI-compatible chat models)  [NEW]
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.temperature = kwargs.get("temperature", 0.7)
        self.timeout = kwargs.get("timeout", 60)
        self.max_retries = kwargs.get("max_retries", 3)

    def generate(self, prompt: str, max_tokens: Optional[int] = None, system: Optional[str] = None) -> str:
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

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        # SDK supports timeout on client
        self.client = OpenAI(api_key=api_key, timeout=self.timeout)

    def generate(self, prompt: str, max_tokens: Optional[int] = None, system: Optional[str] = None) -> str:
        max_tokens = max_tokens or self.max_tokens

        req = {
            "model": self.model,
            "max_output_tokens": max_tokens,
        }

        # NOTE: Do not pass temperature for GPT-5/GPT-O4 if unsupported.
        # Keep temperature only for models that accept it (safe default: omit).
        # If you REALLY want temperature for certain OpenAI models, add it conditionally.

        if system:
            req["input"] = [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ]
        else:
            req["input"] = prompt

        # Reasoning control: keep for GPT-5 family; also OK to apply to GPT-O4 if you want.
        # (If model rejects it, remove/condition it.)
        if self.model.startswith("gpt-5"):
            req["reasoning"] = {"effort": "low"}

        response = self.client.responses.create(**req)
        return response.output_text


class BedrockProvider(LLMProvider):
    """AWS Bedrock provider for Claude models (existing)."""

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 package is required for Bedrock provider. "
                "Install with: pip install boto3"
            )

        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION", "us-east-1")

        if bearer_token:
            self.bedrock_runtime = None
            self.bearer_token = bearer_token
            self.region = region
            self.use_bearer = True
        elif access_key and secret_key:
            self.bedrock_runtime = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
            )
            self.use_bearer = False
        else:
            raise ValueError(
                "AWS credentials not found. Please set either:\n"
                "  - AWS_BEARER_TOKEN_BEDROCK (preferred), or\n"
                "  - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION\n"
                "in your .env file."
            )

    def generate(self, prompt: str, max_tokens: Optional[int] = None, system: Optional[str] = None) -> str:
        max_tokens = max_tokens or self.max_tokens

        import json
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Bedrock Claude supports a system field in some model variants; keep simple here.
        if system:
            request_body["system"] = system

        try:
            if self.use_bearer:
                import requests
                url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model}/invoke"
                headers = {
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Content-Type": "application/json",
                }
                response = requests.post(url, headers=headers, json=request_body, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
            else:
                response = self.bedrock_runtime.invoke_model(modelId=self.model, body=json.dumps(request_body))
                result = json.loads(response["body"].read())

            if "content" in result and len(result["content"]) > 0:
                return result["content"][0]["text"]
            raise Exception("Empty response from Bedrock")
        except Exception as e:
            raise Exception(f"Bedrock API error: {e}")


class GoogleProvider(LLMProvider):
    """Google Gemini provider (existing, with optional system prompt support)."""

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            from google import genai
            self.genai_module = genai
            self.use_new_api = True
        except ImportError:
            try:
                import google.generativeai as genai
                self.genai_module = genai
                self.use_new_api = False
            except ImportError:
                raise ImportError(
                    "google-genai package is required for Google provider. "
                    "Install with: pip install google-genai"
                )

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        if self.use_new_api:
            os.environ["GEMINI_API_KEY"] = api_key
            self.client = self.genai_module.Client()
            self.model_name = model
        else:
            self.genai_module.configure(api_key=api_key)
            self.client = self.genai_module.GenerativeModel(model_name=self.model)
            self.model_name = None

    def generate(self, prompt: str, max_tokens: Optional[int] = None, system: Optional[str] = None) -> str:
        # IMPORTANT: For Gemini, keeping output short is usually best.
        # If you want max tokens control, add model-specific config here.

        max_retries = self.max_retries
        last_exception = None

        for attempt in range(max_retries):
            try:
                # For both APIs: prepend system instruction into prompt if provided
                # (Gemini APIs differ in how "system" is passed; this is reliable.)
                final_prompt = prompt if not system else (system.strip() + "\n\n" + prompt)

                if self.use_new_api:
                    generation_config = {"temperature": self.temperature}
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=final_prompt,
                        config=generation_config,
                    )

                    text = None
                    if hasattr(response, "_get_text"):
                        try:
                            text = response._get_text()
                            if text:
                                text = str(text).strip()
                        except Exception:
                            pass

                    if not text and hasattr(response, "text"):
                        try:
                            if response.text:
                                text = str(response.text).strip()
                        except Exception:
                            pass

                    if not text and hasattr(response, "candidates") and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, "content") and candidate.content:
                                content = candidate.content
                                if hasattr(content, "parts") and content.parts:
                                    for part in content.parts:
                                        if hasattr(part, "text") and part.text:
                                            text = str(part.text).strip()
                                            break
                                if not text and hasattr(content, "text") and content.text:
                                    text = str(content.text).strip()
                            if not text and hasattr(candidate, "text") and candidate.text:
                                text = str(candidate.text).strip()
                            if text:
                                break

                    if not text:
                        raise Exception("Google Gemini API returned empty text.")
                    return text

                else:
                    generation_config = {"temperature": self.temperature}
                    response = self.client.generate_content(final_prompt, generation_config=generation_config)
                    return response.text

            except Exception as e:
                error_str = str(e).lower()
                is_network_error = any(k in error_str for k in ["timeout", "connection", "ssl", "network", "socket", "broken pipe"])
                if is_network_error and attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2
                    print(f"  Network error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    last_exception = e
                    continue
                raise Exception(f"Google Gemini API error: {e}")

        raise Exception(f"Google Gemini API error after {max_retries} attempts: {last_exception}")


class TogetherProvider(LLMProvider):
    """Together AI provider (existing)."""

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package is required for Together AI provider. "
                "Install with: pip install requests"
            )

        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        self.api_key = api_key
        self.api_url = "https://api.together.xyz/v1/chat/completions"

    def generate(self, prompt: str, max_tokens: Optional[int] = None, system: Optional[str] = None) -> str:
        import requests
        max_tokens = max_tokens or self.max_tokens

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(self.api_url, headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"]
            raise Exception("Empty response from Together AI")
        except Exception as e:
            raise Exception(f"Together AI API error: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic provider (Claude direct API)."""

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic provider. "
                "Install with: pip install anthropic"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        self.client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout)

    def generate(self, prompt: str, max_tokens: Optional[int] = None, system: Optional[str] = None) -> str:
        max_tokens = max_tokens or self.max_tokens

        # Claude supports system separately
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        # Temperature is supported on Claude, but keep it optional if you want strict determinism
        kwargs["temperature"] = self.temperature

        try:
            resp = self.client.messages.create(**kwargs)
            # resp.content is list of content blocks
            if resp.content and len(resp.content) > 0 and hasattr(resp.content[0], "text"):
                return resp.content[0].text
            # fallback: stringify
            return str(resp)
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter provider.

    OpenRouter offers an OpenAI-compatible Chat Completions API.
    We'll call: https://openrouter.ai/api/v1/chat/completions
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package is required for OpenRouter provider. "
                "Install with: pip install requests"
            )

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        self.api_key = api_key
        self.api_url = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")

        # Optional but recommended by OpenRouter for attribution / routing
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER")  # e.g., https://your-project
        self.app_title = os.getenv("OPENROUTER_APP_TITLE")        # e.g., review_generation

    def generate(self, prompt: str, max_tokens: Optional[int] = None, system: Optional[str] = None) -> str:
        import requests
        max_tokens = max_tokens or self.max_tokens

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title

        payload = {
            "model": self.model,  # e.g. "meta-llama/llama-4-scout-72b" or "deepseek/deepseek-r1"
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"]
            raise Exception("Empty response from OpenRouter")
        except Exception as e:
            raise Exception(f"OpenRouter API error: {e}")


def create_provider(provider_name: str, model: str, **kwargs) -> LLMProvider:
    """
    Create an LLM provider instance.

    provider_name options:
      - openai
      - bedrock
      - google
      - together
      - anthropic   (Claude direct)
      - openrouter  (Llama / DeepSeek / etc.)
    """
    p = provider_name.lower()

    if p == "openai":
        return OpenAIProvider(model, **kwargs)
    if p == "bedrock":
        return BedrockProvider(model, **kwargs)
    if p == "google":
        return GoogleProvider(model, **kwargs)
    if p == "together":
        return TogetherProvider(model, **kwargs)
    if p == "anthropic":
        return AnthropicProvider(model, **kwargs)
    if p == "openrouter":
        return OpenRouterProvider(model, **kwargs)

    raise ValueError(
        f"Unknown provider: {provider_name}. "
        f"Supported providers: openai, bedrock, google, together, anthropic, openrouter"
    )
