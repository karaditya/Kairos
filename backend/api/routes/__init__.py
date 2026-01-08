"""CAE API Routes."""

from api.routes.session import router as session_router
from api.routes.vision import router as vision_router
from api.routes.agent import router as agent_router
from api.routes.audio import router as audio_router
from api.routes.admin import router as admin_router

__all__ = [
    "session_router",
    "vision_router",
    "agent_router",
    "audio_router",
    "admin_router",
]
