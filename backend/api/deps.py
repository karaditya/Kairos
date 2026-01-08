"""
FastAPI Dependency Injection for CAE System.

Provides dependencies for:
- Database access
- Service initialization (Parlant, Whisper, Vision, Qdrant)
- Authentication
"""

from typing import Optional

from fastapi import Depends, HTTPException, Header, status

from config import STAFF_PIN
from db.database import get_db
from utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATABASE DEPENDENCY
# =============================================================================

async def get_database():
    """Get database connection."""
    return get_db()


# =============================================================================
# AUTHENTICATION
# =============================================================================

async def verify_staff_pin(x_staff_pin: Optional[str] = Header(None)) -> bool:
    """
    Verify staff PIN from header.

    Used for endpoints that require staff authentication.
    """
    if not x_staff_pin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Staff PIN required (X-Staff-Pin header)",
        )

    if x_staff_pin != STAFF_PIN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid staff PIN",
        )

    return True


# =============================================================================
# SERVICE DEPENDENCIES (Lazy Loading)
# =============================================================================

# Singleton instances for services
_parlant_agent = None
_whisper_engine = None
_vision_service = None
_qdrant_client = None


async def get_parlant_agent():
    """
    Get Parlant Scribe Agent instance.

    Lazy initialization - agent is created on first request.
    """
    global _parlant_agent

    if _parlant_agent is None:
        from services.parlant.agent_factory import create_scribe_agent
        _parlant_agent = await create_scribe_agent()
        logger.info("Parlant Scribe Agent initialized")

    return _parlant_agent


async def get_whisper_engine():
    """
    Get Faster-Whisper engine instance.

    Lazy initialization - engine is created on first request.
    """
    global _whisper_engine

    if _whisper_engine is None:
        from services.audio.whisper_engine import create_whisper_engine
        _whisper_engine = await create_whisper_engine()
        logger.info("Whisper engine initialized")

    return _whisper_engine


async def get_vision_service():
    """
    Get DeepSeek-VL2 vision service instance.

    Lazy initialization - model is loaded on first request.
    """
    global _vision_service

    if _vision_service is None:
        from services.vision.deepseek_vl2 import create_vision_service
        _vision_service = await create_vision_service()
        logger.info("Vision service initialized")

    return _vision_service


async def get_qdrant():
    """
    Get Qdrant client instance.

    Lazy initialization - client is created on first request.
    """
    global _qdrant_client

    if _qdrant_client is None:
        from services.rag.qdrant_client import create_qdrant_client
        _qdrant_client = await create_qdrant_client()
        logger.info("Qdrant client initialized")

    return _qdrant_client


# =============================================================================
# COMBINED DEPENDENCIES
# =============================================================================

async def get_all_services():
    """
    Get all services as a dictionary.

    Useful for endpoints that need multiple services.
    """
    return {
        "parlant": await get_parlant_agent(),
        "whisper": await get_whisper_engine(),
        "vision": await get_vision_service(),
        "qdrant": await get_qdrant(),
    }


# =============================================================================
# CLEANUP
# =============================================================================

async def cleanup_services():
    """Cleanup all service instances."""
    global _parlant_agent, _whisper_engine, _vision_service, _qdrant_client

    if _parlant_agent:
        from services.parlant.agent_factory import shutdown_parlant
        await shutdown_parlant()
        _parlant_agent = None

    if _whisper_engine:
        from services.audio.whisper_engine import shutdown_whisper
        await shutdown_whisper()
        _whisper_engine = None

    if _vision_service:
        from services.vision.deepseek_vl2 import shutdown_vision
        await shutdown_vision()
        _vision_service = None

    if _qdrant_client:
        _qdrant_client = None

    logger.info("All services cleaned up")
