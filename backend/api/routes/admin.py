"""
Admin API Routes for CAE System.

Endpoints:
- POST /admin/protocols/ingest - Ingest medical protocols
- GET /admin/protocols/search - Search protocols
- GET /admin/protocols/stats - Get protocol statistics
- DELETE /admin/protocols/clear - Clear all protocols
- GET /admin/models - List available models
"""

from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from api.deps import verify_staff_pin
from services.rag.protocol_store import get_protocol_store
from services.llm.ollama_client import get_ollama_client
from utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ProtocolInput(BaseModel):
    """Protocol to ingest."""
    title: str
    content: str
    source: str
    language: str = "fr"
    category: Optional[str] = None
    keywords: Optional[List[str]] = None


class ProtocolIngestRequest(BaseModel):
    """Request to ingest protocols."""
    protocols: List[ProtocolInput]


class ProtocolSearchRequest(BaseModel):
    """Request to search protocols."""
    query: str
    limit: int = 5
    language: Optional[str] = None
    category: Optional[str] = None


# =============================================================================
# PROTOCOL MANAGEMENT
# =============================================================================

@router.post("/protocols/ingest")
async def ingest_protocols(
    request: ProtocolIngestRequest,
    _: bool = Depends(verify_staff_pin),
):
    """
    Ingest medical protocols into Qdrant.

    Protocols are embedded and stored for RAG retrieval.
    """
    protocol_store = await get_protocol_store()

    # Convert to dict list
    protocols = [
        {
            "title": p.title,
            "content": p.content,
            "source": p.source,
            "language": p.language,
            "category": p.category,
            "keywords": p.keywords,
        }
        for p in request.protocols
    ]

    count = await protocol_store.ingest_batch(protocols)

    logger.info("Protocols ingested", count=count)

    return {"ingested": count, "message": f"Ingested {count} protocols"}


@router.post("/protocols/search")
async def search_protocols(request: ProtocolSearchRequest):
    """Search protocols using semantic similarity."""
    protocol_store = await get_protocol_store()

    results = await protocol_store.search(
        query=request.query,
        limit=request.limit,
        language=request.language,
        category=request.category,
    )

    return {"results": results, "count": len(results)}


@router.get("/protocols/stats")
async def protocol_stats():
    """Get protocol store statistics."""
    protocol_store = await get_protocol_store()
    stats = await protocol_store.get_stats()

    return stats


@router.delete("/protocols/clear")
async def clear_protocols(_: bool = Depends(verify_staff_pin)):
    """Clear all protocols from the store."""
    protocol_store = await get_protocol_store()
    await protocol_store.clear()

    logger.info("Protocol store cleared")

    return {"cleared": True}


# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

@router.get("/models")
async def list_models():
    """List available Ollama models."""
    client = get_ollama_client()

    try:
        # Check Ollama is running
        if not await client.health_check():
            return {
                "available": False,
                "error": "Ollama not running",
                "models": [],
            }

        models = await client.get_ready_models()
        return {
            "available": True,
            "models": models,
            "active_model": client.model,
        }

    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "models": [],
        }


@router.post("/models/select")
async def select_model(
    model_id: str,
    _: bool = Depends(verify_staff_pin),
):
    """Select active Ollama model."""
    client = get_ollama_client()

    try:
        await client.set_model(model_id)
        return {"selected": model_id}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/models/pull/{model_id}")
async def pull_model(
    model_id: str,
    _: bool = Depends(verify_staff_pin),
):
    """
    Pull a model from Ollama registry.

    Returns streaming progress updates.
    """
    from fastapi.responses import StreamingResponse
    import json

    client = get_ollama_client()

    async def generate():
        async for progress in client.pull_model(model_id):
            yield json.dumps(progress) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


# =============================================================================
# SYSTEM STATUS
# =============================================================================

@router.get("/status")
async def system_status():
    """Get detailed system status."""
    from services.parlant.agent_factory import get_parlant_status
    from services.audio.whisper_engine import get_whisper_status
    from services.vision.deepseek_vl2 import get_vision_status
    from services.rag.qdrant_client import get_qdrant_status

    return {
        "database": "ready",
        "ollama": await _check_ollama(),
        "parlant": await get_parlant_status(),
        "whisper": await get_whisper_status(),
        "vision": await get_vision_status(),
        "qdrant": await get_qdrant_status(),
    }


async def _check_ollama() -> str:
    """Check Ollama status."""
    try:
        client = get_ollama_client()
        if await client.health_check():
            return "ready"
        return "not_running"
    except Exception:
        return "error"


@router.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "healthy"}
