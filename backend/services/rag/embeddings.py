"""
Embedding Service for CAE System.

Provides text embeddings via:
1. Ollama (primary) - uses nomic-embed-text
2. Sentence Transformers (fallback) - local model
"""

from typing import Optional, List

from config import OLLAMA_EMBEDDING_MODEL, EMBEDDING_DIMENSION
from utils.logging import get_logger
from utils.exceptions import EmbeddingError

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self):
        self._ollama_client = None
        self._sentence_transformer = None
        self._use_ollama = True

    async def initialize(self) -> None:
        """Initialize embedding service."""
        # Try Ollama first
        try:
            from services.llm.ollama_client import get_ollama_client
            self._ollama_client = get_ollama_client()

            # Test embedding
            test_embedding = await self._ollama_client.embed("test")
            if test_embedding:
                logger.info(
                    "Embedding service initialized with Ollama",
                    model=OLLAMA_EMBEDDING_MODEL,
                    dimension=len(test_embedding),
                )
                self._use_ollama = True
                return
        except Exception as e:
            logger.warning("Ollama embedding not available", error=str(e))

        # Fallback to sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_transformer = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu",
            )
            self._use_ollama = False
            logger.info("Embedding service initialized with SentenceTransformers")
        except ImportError:
            raise EmbeddingError(
                "No embedding backend available. Install sentence-transformers or run Ollama."
            )

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not text.strip():
            return [0.0] * EMBEDDING_DIMENSION

        if self._use_ollama and self._ollama_client:
            try:
                return await self._ollama_client.embed(text)
            except Exception as e:
                logger.warning("Ollama embedding failed, trying fallback", error=str(e))

        if self._sentence_transformer:
            try:
                embedding = self._sentence_transformer.encode(
                    text,
                    convert_to_numpy=True,
                )
                return embedding.tolist()
            except Exception as e:
                raise EmbeddingError(f"Sentence transformer embedding failed: {str(e)}")

        raise EmbeddingError("No embedding backend available")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if self._use_ollama and self._ollama_client:
            try:
                return await self._ollama_client.embed_batch(texts)
            except Exception as e:
                logger.warning("Ollama batch embedding failed", error=str(e))

        if self._sentence_transformer:
            try:
                embeddings = self._sentence_transformer.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return [e.tolist() for e in embeddings]
            except Exception as e:
                raise EmbeddingError(f"Batch embedding failed: {str(e)}")

        # Fallback to sequential embedding
        return [await self.embed(text) for text in texts]


# =============================================================================
# SINGLETON
# =============================================================================

_embedding_service: Optional[EmbeddingService] = None


async def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        await _embedding_service.initialize()
    return _embedding_service
