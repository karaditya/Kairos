"""
Protocol Store for CAE System.

Stores and searches medical/triage protocols in Qdrant.
"""

from typing import Optional, List, Dict, Any
import uuid

from config import QDRANT_COLLECTION_PROTOCOLS
from services.rag.qdrant_client import QdrantService, create_qdrant_client
from services.rag.embeddings import EmbeddingService, get_embedding_service
from utils.logging import get_logger
from utils.exceptions import RAGError

logger = get_logger(__name__)


class ProtocolStore:
    """Store and search medical protocols."""

    def __init__(
        self,
        qdrant: QdrantService,
        embeddings: EmbeddingService,
    ):
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.collection = QDRANT_COLLECTION_PROTOCOLS

    async def ingest_protocol(
        self,
        title: str,
        content: str,
        source: str,
        language: str = "fr",
        category: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Ingest a single protocol.

        Args:
            title: Protocol title
            content: Protocol content/text
            source: Source of protocol (e.g., "SFMU", "Hospital XYZ")
            language: Language code
            category: Protocol category
            keywords: List of keywords
            metadata: Additional metadata

        Returns:
            Protocol ID
        """
        protocol_id = str(uuid.uuid4())

        # Generate embedding from title + content
        text_for_embedding = f"{title}\n\n{content}"
        embedding = await self.embeddings.embed(text_for_embedding)

        # Build payload
        payload = {
            "id": protocol_id,
            "title": title,
            "content": content,
            "source": source,
            "language": language,
            "category": category or "general",
            "keywords": keywords or [],
            **(metadata or {}),
        }

        # Upsert to Qdrant
        await self.qdrant.upsert(
            collection=self.collection,
            vectors=[embedding],
            payloads=[payload],
            ids=[protocol_id],
        )

        logger.info("Protocol ingested", protocol_id=protocol_id, title=title)
        return protocol_id

    async def ingest_batch(
        self,
        protocols: List[Dict[str, Any]],
    ) -> int:
        """
        Ingest multiple protocols.

        Args:
            protocols: List of protocol dicts with title, content, source, etc.

        Returns:
            Number of protocols ingested
        """
        if not protocols:
            return 0

        # Generate embeddings
        texts = [
            f"{p.get('title', '')}\n\n{p.get('content', '')}"
            for p in protocols
        ]
        embeddings = await self.embeddings.embed_batch(texts)

        # Build payloads
        ids = []
        payloads = []
        for p in protocols:
            protocol_id = str(uuid.uuid4())
            ids.append(protocol_id)
            payloads.append({
                "id": protocol_id,
                "title": p.get("title", ""),
                "content": p.get("content", ""),
                "source": p.get("source", "unknown"),
                "language": p.get("language", "fr"),
                "category": p.get("category", "general"),
                "keywords": p.get("keywords", []),
            })

        # Upsert to Qdrant
        await self.qdrant.upsert(
            collection=self.collection,
            vectors=embeddings,
            payloads=payloads,
            ids=ids,
        )

        logger.info("Protocols batch ingested", count=len(protocols))
        return len(protocols)

    async def search(
        self,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        category: Optional[str] = None,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search protocols by semantic similarity.

        Args:
            query: Search query
            limit: Maximum results
            language: Filter by language
            category: Filter by category
            min_score: Minimum similarity score

        Returns:
            List of matching protocols with scores
        """
        # Generate query embedding
        query_embedding = await self.embeddings.embed(query)

        # Build filter
        filter_conditions = {}
        if language:
            filter_conditions["language"] = language
        if category:
            filter_conditions["category"] = category

        # Search
        results = await self.qdrant.search(
            collection=self.collection,
            query_vector=query_embedding,
            limit=limit,
            filter_conditions=filter_conditions if filter_conditions else None,
            score_threshold=min_score,
        )

        return [
            {
                "id": r["payload"]["id"],
                "title": r["payload"]["title"],
                "content": r["payload"]["content"],
                "source": r["payload"]["source"],
                "category": r["payload"]["category"],
                "score": r["score"],
            }
            for r in results
        ]

    async def get_by_id(self, protocol_id: str) -> Optional[Dict[str, Any]]:
        """Get protocol by ID."""
        # Search with exact ID match
        results = await self.qdrant.search(
            collection=self.collection,
            query_vector=[0.0] * 768,  # Dummy vector
            limit=1,
            filter_conditions={"id": protocol_id},
        )

        if results:
            return results[0]["payload"]
        return None

    async def clear(self) -> bool:
        """Clear all protocols."""
        await self.qdrant.clear_collection(self.collection)
        logger.info("Protocol store cleared")
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get protocol store statistics."""
        return await self.qdrant.get_collection_stats(self.collection)


# =============================================================================
# FACTORY
# =============================================================================

_protocol_store: Optional[ProtocolStore] = None


async def get_protocol_store() -> ProtocolStore:
    """Get singleton protocol store."""
    global _protocol_store
    if _protocol_store is None:
        qdrant = await create_qdrant_client()
        embeddings = await get_embedding_service()
        _protocol_store = ProtocolStore(qdrant, embeddings)
    return _protocol_store
