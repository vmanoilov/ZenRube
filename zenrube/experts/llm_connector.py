import aiohttp
import json
import logging
from typing import Dict, Any, Optional

EXPERT_METADATA = {
    "name": "llm_connector",
    "version": "1.0",
    "description": "Connects to external LLM providers (OpenAI, Qwen, Grok, Claude, Gemini) at runtime with dynamic configuration.",
    "author": "vladinc@gmail.com"
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMConnectorExpert:
    """
    LLM Connector Expert that dynamically connects to various LLM providers.
    Supports OpenAI, Qwen, Grok, Claude, and Gemini APIs with runtime configuration.
    """

    def __init__(self):
        """Initialize the LLM Connector Expert."""
        logger.info("LLMConnectorExpert initialized")

    def run(self, prompt: str) -> str:
        """
        Execute a prompt using the configured LLM provider.
        
        Args:
            prompt (str): The input prompt to send to the LLM.
            
        Returns:
            str: The LLM's response, or error message if configuration is missing.
        """
        try:
            # Load LLM configuration
            from zenrube.config.llm_config_loader import get_llm_config
            config = get_llm_config()
            
            # Validate configuration
            if not self._validate_config(config):
                return "llm_error: Missing or invalid LLM configuration. Use /zenrube/configure_llm to set up provider settings."
            
            # Route to appropriate provider
            return self._call_provider(prompt, config)
            
        except Exception as e:
            logger.error(f"LLM Connector execution failed: {e}")
            return f"llm_error: {str(e)}"

    def _validate_config(self, config: Optional[Dict[str, Any]]) -> bool:
        """Validate that required LLM configuration is present."""
        if not config:
            logger.warning("No LLM configuration found")
            return False
            
        required_fields = ["provider", "api_key", "model"]
        missing_fields = [field for field in required_fields if not config.get(field)]
        
        if missing_fields:
            logger.warning(f"Missing LLM config fields: {missing_fields}")
            return False
            
        return True

    def _call_provider(self, prompt: str, config: Dict[str, Any]) -> str:
        """Call the appropriate LLM provider based on configuration."""
        provider = config["provider"].lower()
        api_key = config["api_key"]
        model = config["model"]
        endpoint = config.get("endpoint", "")
        
        if provider == "openai":
            return self._call_openai(prompt, api_key, model, endpoint)
        elif provider == "qwen":
            return self._call_qwen(prompt, api_key, model, endpoint)
        elif provider == "grok":
            return self._call_grok(prompt, api_key, model, endpoint)
        elif provider == "claude":
            return self._call_claude(prompt, api_key, model, endpoint)
        elif provider == "gemini":
            return self._call_gemini(prompt, api_key, model, endpoint)
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            return f"llm_error: Unsupported provider '{provider}'. Supported providers: openai, qwen, grok, claude, gemini"

    def _call_openai(self, prompt: str, api_key: str, model: str, endpoint: str) -> str:
        """Call OpenAI API."""
        try:
            import aiohttp
            import asyncio
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            url = endpoint or "https://api.openai.com/v1/chat/completions"
            
            return asyncio.run(self._make_http_request(url, headers, data))
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return f"openai_error: {str(e)}"

    def _call_qwen(self, prompt: str, api_key: str, model: str, endpoint: str) -> str:
        """Call Qwen API."""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "input": {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                },
                "parameters": {
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            }
            
            url = endpoint or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
            
            return asyncio.run(self._make_http_request(url, headers, data))
            
        except Exception as e:
            logger.error(f"Qwen API call failed: {e}")
            return f"qwen_error: {str(e)}"

    def _call_grok(self, prompt: str, api_key: str, model: str, endpoint: str) -> str:
        """Call Grok (xAI) API."""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "model": model,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            url = endpoint or "https://api.x.ai/v1/chat/completions"
            
            return asyncio.run(self._make_http_request(url, headers, data))
            
        except Exception as e:
            logger.error(f"Grok API call failed: {e}")
            return f"grok_error: {str(e)}"

    def _call_claude(self, prompt: str, api_key: str, model: str, endpoint: str) -> str:
        """Call Claude API."""
        try:
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": model,
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            url = endpoint or "https://api.anthropic.com/v1/messages"
            
            return asyncio.run(self._make_http_request(url, headers, data))
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return f"claude_error: {str(e)}"

    def _call_gemini(self, prompt: str, api_key: str, model: str, endpoint: str) -> str:
        """Call Gemini API."""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 1000,
                    "temperature": 0.7
                }
            }
            
            url = endpoint or f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            
            return asyncio.run(self._make_http_request(url, headers, data, include_api_key_in_url=True))
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return f"gemini_error: {str(e)}"

    async def _make_http_request(self, url: str, headers: Dict[str, str], data: Dict[str, Any], include_api_key_in_url: bool = False) -> str:
        """Make an async HTTP request to the LLM API."""
        try:
            async with aiohttp.ClientSession() as session:
                if include_api_key_in_url and "api_key" in data:
                    # For Gemini, the API key goes in the URL, not headers
                    url = url.replace("{api_key}", data.pop("api_key"))
                
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._extract_response(result, headers.get("anthropic-version"))
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        return f"api_error: HTTP {response.status} - {error_text}"
                        
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return f"http_error: {str(e)}"

    def _extract_response(self, result: Dict[str, Any], is_claude: bool = False) -> str:
        """Extract the response text from API response."""
        try:
            if is_claude:
                # Claude API response format
                if "content" in result and len(result["content"]) > 0:
                    return result["content"][0].get("text", "No response text")
                return str(result)
            else:
                # OpenAI/Qwen/Grok/Gemini response format
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("message", {}).get("content", "No response text")
                elif "text" in result:
                    return result["text"]
                elif "candidates" in result and len(result["candidates"]) > 0:
                    return result["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "No response text")
                else:
                    return str(result)
        except Exception as e:
            logger.error(f"Failed to extract response: {e}")
            return f"parse_error: {str(e)}"
