"""
Embeddings Client for ZenRube

This module provides a client for generating text embeddings using various providers.
Currently supports OpenAI-compatible APIs.

Author: ZenRube Core Engineer
"""

import logging
import requests
from typing import List, Dict, Any, Optional
import time

from .config_loader import load_config

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """
    Client for generating text embeddings.
    """

    def __init__(self):
        self.config = load_config()
        if self.config is None:
            logger.warning("Embeddings config not found - embeddings disabled")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"Embeddings client initialized with provider: {self.config.get('provider', 'unknown')}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            RuntimeError: If embeddings are disabled or request fails
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embeddings are disabled or request fails
        """
        if not self.enabled or not self.config:
            raise RuntimeError("Embeddings are disabled - no valid configuration found")

        provider = self.config.get("provider", "").lower()

        if provider == "openai":
            return self._embed_openai(texts)
        else:
            raise RuntimeError(f"Unsupported embeddings provider: {provider}")

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI-compatible API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If the API request fails
        """
        api_key = self.config.get("api_key", "")
        if not api_key:
            raise RuntimeError("OpenAI API key not configured")

        base_url = self.config.get("base_url", "https://api.openai.com/v1")
        model = self.config.get("model", "text-embedding-3-small")

        url = f"{base_url}/embeddings"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": texts,
            "model": model
        }

        try:
            logger.debug(f"Making embeddings request to {url} for {len(texts)} texts")
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]

            logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"Embeddings API request failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid response format from embeddings API: {e}")
            raise RuntimeError(f"Invalid embeddings API response: {e}")


# Global client instance
_client_instance: Optional[EmbeddingsClient] = None


def get_embeddings_client() -> EmbeddingsClient:
    """
    Get the global embeddings client instance.

    Returns:
        EmbeddingsClient instance
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = EmbeddingsClient()
    return _client_instance


def embed_text(text: str) -> List[float]:
    """
    Convenience function to embed a single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector

    Raises:
        RuntimeError: If embeddings fail
    """
    return get_embeddings_client().embed_text(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convenience function to embed multiple texts.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors

    Raises:
        RuntimeError: If embeddings fail
    """
    return get_embeddings_client().embed_texts(texts)