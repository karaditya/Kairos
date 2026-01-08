"""
Clinical Admin Edge (CAE) System - FastAPI Application

Ambient Admin Assistant for medical clinics.
100% local/offline for data privacy.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import DEBUG_MODE
from db.database import init_db, close_db
from utils.logging import get_logger
from utils.exceptions import CAEException

# Import routers
from api.routes.session import router as session_router
from api.routes.vision import router as vision_router
from api.routes.agent import router as agent_router
from api.routes.audio import router as audio_router
from api.routes.admin import router as admin_router

logger = get_logger(__name__)


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Startup:
        - Initialize database
        - Initialize services (Parlant, Whisper, Vision, Qdrant)

    Shutdown:
        - Close database connections
        - Cleanup service resources
    """
    logger.info("CAE System starting up...")

    # Initialize database
    await init_db()

    # Initialize services (lazy loading - will be initialized on first use)
    # Services are imported and initialized by dependency injection

    logger.info("CAE System ready", debug_mode=DEBUG_MODE)

    yield

    # Shutdown
    logger.info("CAE System shutting down...")

    # Close database
    await close_db()

    # Cleanup services
    try:
        from services.parlant.agent_factory import shutdown_parlant
        await shutdown_parlant()
    except ImportError:
        pass  # Service not yet implemented

    try:
        from services.audio.whisper_engine import shutdown_whisper
        await shutdown_whisper()
    except ImportError:
        pass

    try:
        from services.vision.deepseek_vl2 import shutdown_vision
        await shutdown_vision()
    except ImportError:
        pass

    logger.info("CAE System shutdown complete")


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Clinical Admin Edge (CAE) System",
        description="Ambient Admin Assistant for medical clinics. 100% local/offline.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if DEBUG_MODE else None,
        redoc_url="/redoc" if DEBUG_MODE else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(CAEException)
    async def cae_exception_handler(request: Request, exc: CAEException):
        """Handle CAE custom exceptions."""
        return JSONResponse(
            status_code=400,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error("Unexpected error", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"error": str(exc)} if DEBUG_MODE else {},
            },
        )

    # Register routers
    app.include_router(session_router, prefix="/session", tags=["Session"])
    app.include_router(vision_router, prefix="/vision", tags=["Vision"])
    app.include_router(agent_router, prefix="/agent", tags=["Agent"])
    app.include_router(audio_router, prefix="/audio", tags=["Audio"])
    app.include_router(admin_router, prefix="/admin", tags=["Admin"])

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "CAE System",
            "version": "1.0.0",
        }

    # System status endpoint
    @app.get("/status", tags=["Health"])
    async def system_status():
        """Get detailed system status."""
        status = {
            "database": "ready",
            "parlant": "not_initialized",
            "whisper": "not_initialized",
            "vision": "not_initialized",
            "qdrant": "not_initialized",
        }

        # Check Parlant
        try:
            from services.parlant.agent_factory import get_parlant_status
            status["parlant"] = await get_parlant_status()
        except ImportError:
            pass

        # Check Whisper
        try:
            from services.audio.whisper_engine import get_whisper_status
            status["whisper"] = await get_whisper_status()
        except ImportError:
            pass

        # Check Vision
        try:
            from services.vision.deepseek_vl2 import get_vision_status
            status["vision"] = await get_vision_status()
        except ImportError:
            pass

        # Check Qdrant
        try:
            from services.rag.qdrant_client import get_qdrant_status
            status["qdrant"] = await get_qdrant_status()
        except ImportError:
            pass

        return status

    return app


# Create application instance
app = create_app()


# =============================================================================
# DEVELOPMENT SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG_MODE,
        log_level="debug" if DEBUG_MODE else "info",
    )
