"""
Ollama Client for CAE System.

Provides async interface to local Ollama server for:
- Text generation
- Embeddings
- Model management
"""

from typing import Optional, List, Dict, Any, AsyncGenerator
import httpx

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_EMBEDDING_MODEL,
    SUPPORTED_OLLAMA_MODELS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)
from utils.logging import get_logger
from utils.exceptions import OllamaConnectionError, ModelNotFoundError, LLMError

logger = get_logger(__name__)


class OllamaClient:
    """Async client for Ollama API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # =========================================================================
    # Health & Model Management
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if Ollama server is running."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning("Ollama health check failed", error=str(e))
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models on Ollama server."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except httpx.ConnectError:
            raise OllamaConnectionError(self.base_url)
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            return []

    async def get_ready_models(self) -> List[Dict[str, Any]]:
        """Get models that are ready to use (pulled and available)."""
        models = await self.list_models()
        ready = []
        for model in models:
            model_name = model.get("name", "").split(":")[0]
            if model_name in SUPPORTED_OLLAMA_MODELS:
                info = SUPPORTED_OLLAMA_MODELS[model_name]
                ready.append({
                    "id": model.get("name"),
                    "name": info["name"],
                    "size_gb": info["size_gb"],
                    "description": info["description"],
                    "status": "ready",
                })
        return ready

    async def check_model_exists(self, model: str) -> bool:
        """Check if a model is available."""
        models = await self.list_models()
        model_names = [m.get("name", "") for m in models]
        return model in model_names or any(model in name for name in model_names)

    async def set_model(self, model: str) -> bool:
        """Set active model for generation."""
        if not await self.check_model_exists(model):
            raise ModelNotFoundError(model)
        self.model = model
        logger.info("Model changed", model=model)
        return True

    async def pull_model(self, model: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Pull a model with streaming progress.

        Yields progress updates as dictionaries.
        """
        try:
            client = await self._get_client()
            async with client.stream(
                "POST",
                "/api/pull",
                json={"name": model},
                timeout=None,  # No timeout for pulls
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            raise OllamaConnectionError(self.base_url)

    # =========================================================================
    # Text Generation
    # =========================================================================

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        stream: bool = False,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model to use (defaults to self.model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response

        Returns:
            Generated text
        """
        model = model or self.model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            client = await self._get_client()
            response = await client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

        except httpx.ConnectError:
            raise OllamaConnectionError(self.base_url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(model)
            raise LLMError(f"Ollama error: {e.response.text}")
        except Exception as e:
            raise LLMError(f"Generation failed: {str(e)}")

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming.

        Yields text chunks as they are generated.
        """
        model = model or self.model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            client = await self._get_client()
            async with client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            ) as response:
                import json
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except httpx.ConnectError:
            raise OllamaConnectionError(self.base_url)

    # =========================================================================
    # Embeddings
    # =========================================================================

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed
            model: Embedding model (defaults to OLLAMA_EMBEDDING_MODEL)

        Returns:
            Embedding vector
        """
        model = model or OLLAMA_EMBEDDING_MODEL

        try:
            client = await self._get_client()
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])

        except httpx.ConnectError:
            raise OllamaConnectionError(self.base_url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(model)
            raise LLMError(f"Embedding error: {e.response.text}")

    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Embedding model

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self.embed(text, model)
            embeddings.append(embedding)
        return embeddings


# =============================================================================
# SINGLETON
# =============================================================================

_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get singleton Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


async def shutdown_ollama():
    """Shutdown Ollama client."""
    global _ollama_client
    if _ollama_client:
        await _ollama_client.close()
        _ollama_client = None
