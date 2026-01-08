"""
Screen Capture Service for CAE System.

Cross-platform screen capture using mss (primary) or pyautogui (fallback).
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import io
import base64
from datetime import datetime
import uuid

from PIL import Image

from config import VISION_MAX_IMAGE_SIZE
from utils.logging import get_logger
from utils.exceptions import ScreenCaptureError

logger = get_logger(__name__)


@dataclass
class CaptureResult:
    """Screen capture result."""
    image: Image.Image
    screenshot_id: str
    timestamp: datetime
    width: int
    height: int
    region: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    window_title: Optional[str] = None


class ScreenCapture:
    """Cross-platform screen capture service."""

    def __init__(self):
        self._mss = None
        self._use_mss = True

    def _init_mss(self):
        """Initialize mss for capture."""
        if self._mss is None:
            try:
                import mss
                self._mss = mss.mss()
            except ImportError:
                self._use_mss = False
                logger.warning("mss not available, using pyautogui fallback")

    async def capture_screen(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        window_title: Optional[str] = None,
    ) -> CaptureResult:
        """
        Capture screen or region.

        Args:
            region: Optional (x, y, width, height) tuple
            window_title: Optional window title to focus

        Returns:
            CaptureResult with image and metadata
        """
        # Focus window if specified
        if window_title:
            await self._focus_window(window_title)

        # Capture
        if self._use_mss:
            image = await self._capture_mss(region)
        else:
            image = await self._capture_pyautogui(region)

        # Resize if too large
        image = self._resize_if_needed(image)

        return CaptureResult(
            image=image,
            screenshot_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            width=image.width,
            height=image.height,
            region=region,
            window_title=window_title,
        )

    async def _capture_mss(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Image.Image:
        """Capture using mss."""
        self._init_mss()

        try:
            import mss

            if region:
                x, y, w, h = region
                monitor = {"left": x, "top": y, "width": w, "height": h}
            else:
                # Capture primary monitor
                monitor = self._mss.monitors[1]

            screenshot = self._mss.grab(monitor)

            # Convert to PIL Image
            image = Image.frombytes(
                "RGB",
                screenshot.size,
                screenshot.bgra,
                "raw",
                "BGRX",
            )

            return image

        except Exception as e:
            raise ScreenCaptureError(f"mss capture failed: {str(e)}")

    async def _capture_pyautogui(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Image.Image:
        """Capture using pyautogui."""
        try:
            import pyautogui

            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()

            return screenshot

        except Exception as e:
            raise ScreenCaptureError(f"pyautogui capture failed: {str(e)}")

    async def _focus_window(self, window_title: str) -> bool:
        """Focus window by title."""
        try:
            import pygetwindow as gw

            windows = gw.getWindowsWithTitle(window_title)
            if windows:
                win = windows[0]
                if win.isMinimized:
                    win.restore()
                win.activate()
                return True

            logger.warning("Window not found", title=window_title)
            return False

        except ImportError:
            logger.warning("pygetwindow not available")
            return False
        except Exception as e:
            logger.warning("Window focus failed", error=str(e))
            return False

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if larger than max size."""
        max_dim = max(image.width, image.height)
        if max_dim > VISION_MAX_IMAGE_SIZE:
            ratio = VISION_MAX_IMAGE_SIZE / max_dim
            new_size = (int(image.width * ratio), int(image.height * ratio))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()

    async def get_window_list(self) -> List[Dict[str, Any]]:
        """Get list of visible windows."""
        try:
            import pygetwindow as gw

            windows = []
            for win in gw.getAllWindows():
                if win.title and win.visible:
                    windows.append({
                        "title": win.title,
                        "left": win.left,
                        "top": win.top,
                        "width": win.width,
                        "height": win.height,
                    })

            return windows

        except ImportError:
            return []
        except Exception as e:
            logger.warning("Get window list failed", error=str(e))
            return []

    async def capture_window(self, window_title: str) -> Optional[CaptureResult]:
        """Capture specific window by title."""
        try:
            import pygetwindow as gw

            windows = gw.getWindowsWithTitle(window_title)
            if not windows:
                logger.warning("Window not found", title=window_title)
                return None

            win = windows[0]
            region = (win.left, win.top, win.width, win.height)

            # Focus and capture
            if win.isMinimized:
                win.restore()
            win.activate()

            import asyncio
            await asyncio.sleep(0.3)  # Wait for window to activate

            return await self.capture_screen(region=region, window_title=window_title)

        except ImportError:
            # Fall back to full screen
            return await self.capture_screen(window_title=window_title)
        except Exception as e:
            raise ScreenCaptureError(f"Window capture failed: {str(e)}")


# =============================================================================
# SINGLETON
# =============================================================================

_screen_capture: Optional[ScreenCapture] = None


def get_screen_capture() -> ScreenCapture:
    """Get singleton screen capture service."""
    global _screen_capture
    if _screen_capture is None:
        _screen_capture = ScreenCapture()
    return _screen_capture
