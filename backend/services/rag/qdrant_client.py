"""
Qdrant Vector Database Client for CAE System.

Provides embedded Qdrant for:
- Protocol storage and search
- Patient history storage
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from config import (
    QDRANT_PATH,
    QDRANT_COLLECTION_PROTOCOLS,
    QDRANT_COLLECTION_PATIENTS,
    EMBEDDING_DIMENSION,
)
from utils.logging import get_logger
from utils.exceptions import QdrantError

logger = get_logger(__name__)


class QdrantService:
    """Qdrant embedded vector database service."""

    def __init__(self, path: str = QDRANT_PATH):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client: Optional[QdrantClient] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Qdrant client and collections."""
        if self._initialized:
            return

        try:
            # Create embedded client
            self._client = QdrantClient(path=str(self.path))

            # Create collections if they don't exist
            await self._ensure_collection(
                QDRANT_COLLECTION_PROTOCOLS,
                EMBEDDING_DIMENSION,
            )
            await self._ensure_collection(
                QDRANT_COLLECTION_PATIENTS,
                EMBEDDING_DIMENSION,
            )

            self._initialized = True
            logger.info("Qdrant initialized", path=str(self.path))

        except Exception as e:
            raise QdrantError(f"Failed to initialize Qdrant: {str(e)}")

    async def _ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
    ) -> None:
        """Ensure collection exists, create if not."""
        try:
            collections = self._client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if not exists:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Collection created", collection=collection_name)
        except Exception as e:
            raise QdrantError(f"Failed to ensure collection: {str(e)}")

    @property
    def client(self) -> QdrantClient:
        """Get Qdrant client."""
        if not self._client:
            raise QdrantError("Qdrant not initialized")
        return self._client

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def upsert(
        self,
        collection: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Insert or update vectors.

        Args:
            collection: Collection name
            vectors: List of embedding vectors
            payloads: List of metadata payloads
            ids: Optional list of IDs (generated if not provided)

        Returns:
            Number of points upserted
        """
        if not self._initialized:
            await self.initialize()

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        points = [
            PointStruct(
                id=id_,
                vector=vector,
                payload=payload,
            )
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]

        try:
            self.client.upsert(
                collection_name=collection,
                points=points,
            )
            logger.debug("Upserted points", collection=collection, count=len(points))
            return len(points)
        except Exception as e:
            raise QdrantError(f"Upsert failed: {str(e)}")

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            collection: Collection name
            query_vector: Query embedding vector
            limit: Maximum results to return
            filter_conditions: Optional filter conditions
            score_threshold: Minimum similarity score

        Returns:
            List of results with payload and score
        """
        if not self._initialized:
            await self.initialize()

        # Build filter
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            query_filter = Filter(must=conditions)

        try:
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )

            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in results
            ]
        except Exception as e:
            raise QdrantError(f"Search failed: {str(e)}")

    async def delete(
        self,
        collection: str,
        ids: List[str],
    ) -> int:
        """
        Delete points by ID.

        Args:
            collection: Collection name
            ids: List of point IDs to delete

        Returns:
            Number of points deleted
        """
        if not self._initialized:
            await self.initialize()

        try:
            self.client.delete(
                collection_name=collection,
                points_selector=ids,
            )
            return len(ids)
        except Exception as e:
            raise QdrantError(f"Delete failed: {str(e)}")

    async def clear_collection(self, collection: str) -> bool:
        """Clear all points from a collection."""
        if not self._initialized:
            await self.initialize()

        try:
            # Delete and recreate collection
            self.client.delete_collection(collection)
            await self._ensure_collection(collection, EMBEDDING_DIMENSION)
            logger.info("Collection cleared", collection=collection)
            return True
        except Exception as e:
            raise QdrantError(f"Clear collection failed: {str(e)}")

    # =========================================================================
    # Stats
    # =========================================================================

    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._initialized:
            await self.initialize()

        try:
            info = self.client.get_collection(collection)
            return {
                "name": collection,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value,
            }
        except Exception as e:
            logger.warning("Failed to get collection stats", error=str(e))
            return {"name": collection, "error": str(e)}

    async def get_all_stats(self) -> Dict[str, Any]:
        """Get stats for all collections."""
        if not self._initialized:
            await self.initialize()

        return {
            "protocols": await self.get_collection_stats(QDRANT_COLLECTION_PROTOCOLS),
            "patients": await self.get_collection_stats(QDRANT_COLLECTION_PATIENTS),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_qdrant_service: Optional[QdrantService] = None


async def create_qdrant_client() -> QdrantService:
    """Create and initialize Qdrant service."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
        await _qdrant_service.initialize()
    return _qdrant_service


async def get_qdrant_status() -> str:
    """Get Qdrant service status."""
    global _qdrant_service
    if _qdrant_service is None or not _qdrant_service._initialized:
        return "not_initialized"
    return "ready"
