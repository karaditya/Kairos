"""
RPA Action Executor for CAE System.

Executes atomic RPA actions using pyautogui.
Cross-platform support (Windows, Mac, Linux).
Includes visual verification and retry logic for production reliability.
"""

from typing import Optional, List, Dict, Any, Tuple
import asyncio
import time
import re

from config import RPA_ENABLED, RPA_TYPING_INTERVAL, RPA_CLICK_PAUSE, RPA_FAILSAFE, VISION_ENABLED
from models.rpa_action import RPAAction, RPAActionType
from utils.logging import get_logger
from utils.exceptions import RPAExecutionError, RPAError, VisionError

logger = get_logger(__name__)

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_MS = 500
VERIFICATION_CONFIDENCE_THRESHOLD = 0.7


class RPAExecutor:
    """
    Executes RPA actions using pyautogui.

    All actions require prior human approval (enforced by RPASafetyGate).
    Includes visual verification and retry logic for production reliability.
    """

    def __init__(self):
        self._initialized = False
        self._pyautogui = None
        self._screen_capture = None
        self._vision_service = None
        self._ocr_fallback = None

    def _init_pyautogui(self):
        """Initialize pyautogui with safety settings."""
        if self._initialized:
            return

        try:
            import pyautogui

            # Configure safety
            pyautogui.FAILSAFE = RPA_FAILSAFE  # Move mouse to corner to abort
            pyautogui.PAUSE = RPA_CLICK_PAUSE

            self._pyautogui = pyautogui
            self._initialized = True

            logger.info("PyAutoGUI initialized", failsafe=RPA_FAILSAFE)

        except ImportError:
            raise RPAError("pyautogui not installed. Run: pip install pyautogui")

    async def _get_screen_capture(self):
        """Get screen capture service lazily."""
        if self._screen_capture is None:
            from services.vision.screen_capture import get_screen_capture
            self._screen_capture = get_screen_capture()
        return self._screen_capture

    async def _get_vision_service(self):
        """Get vision service lazily."""
        if self._vision_service is None and VISION_ENABLED:
            try:
                from services.vision.deepseek_vl2 import create_vision_service
                self._vision_service = await create_vision_service()
            except Exception as e:
                logger.warning("Vision service unavailable", error=str(e))
        return self._vision_service

    async def _get_ocr_fallback(self):
        """Get OCR fallback service lazily."""
        if self._ocr_fallback is None:
            try:
                from services.vision.ocr_fallback import get_ocr_fallback
                self._ocr_fallback = get_ocr_fallback()
            except Exception as e:
                logger.warning("OCR fallback unavailable", error=str(e))
        return self._ocr_fallback

    async def execute_action(self, action: RPAAction) -> Dict[str, Any]:
        """
        Execute a single RPA action.

        Args:
            action: RPAAction to execute

        Returns:
            Dict with execution result
        """
        if not RPA_ENABLED:
            return {"success": False, "error": "RPA disabled"}

        self._init_pyautogui()

        try:
            if action.action_type == RPAActionType.CLICK:
                return await self._execute_click(action)
            elif action.action_type == RPAActionType.TYPE:
                return await self._execute_type(action)
            elif action.action_type == RPAActionType.CLEAR:
                return await self._execute_clear(action)
            elif action.action_type == RPAActionType.SCROLL:
                return await self._execute_scroll(action)
            elif action.action_type == RPAActionType.WAIT:
                return await self._execute_wait(action)
            elif action.action_type == RPAActionType.VERIFY:
                return await self._execute_verify(action)
            else:
                return {"success": False, "error": f"Unknown action type: {action.action_type}"}

        except Exception as e:
            logger.error("RPA action failed", action=action.action_type.value, error=str(e))
            raise RPAExecutionError(action.action_type.value, str(e))

    async def _execute_click(self, action: RPAAction) -> Dict[str, Any]:
        """Execute click action."""
        if not action.coordinates or len(action.coordinates) < 2:
            return {"success": False, "error": "Coordinates required for click"}

        x, y = action.coordinates[0], action.coordinates[1]

        # Move and click
        self._pyautogui.moveTo(x, y, duration=0.2)
        await asyncio.sleep(0.1)
        self._pyautogui.click(x, y)

        logger.debug("Click executed", x=x, y=y, target=action.target_field)
        return {"success": True, "action": "click", "x": x, "y": y}

    async def _execute_type(self, action: RPAAction) -> Dict[str, Any]:
        """Execute type action."""
        if not action.value:
            return {"success": False, "error": "Value required for type action"}

        # Click to focus if coordinates provided
        if action.coordinates and len(action.coordinates) >= 2:
            x, y = action.coordinates[0], action.coordinates[1]
            self._pyautogui.click(x, y)
            await asyncio.sleep(0.2)

        # Type text
        self._pyautogui.typewrite(
            action.value,
            interval=RPA_TYPING_INTERVAL,
        )

        logger.debug("Type executed", target=action.target_field, chars=len(action.value))
        return {"success": True, "action": "type", "chars_typed": len(action.value)}

    async def _execute_clear(self, action: RPAAction) -> Dict[str, Any]:
        """Execute clear action (select all + delete)."""
        # Click to focus if coordinates provided
        if action.coordinates and len(action.coordinates) >= 2:
            x, y = action.coordinates[0], action.coordinates[1]
            self._pyautogui.click(x, y)
            await asyncio.sleep(0.1)

        # Select all and delete
        self._pyautogui.hotkey('ctrl', 'a')
        await asyncio.sleep(0.1)
        self._pyautogui.press('delete')

        logger.debug("Clear executed", target=action.target_field)
        return {"success": True, "action": "clear"}

    async def _execute_scroll(self, action: RPAAction) -> Dict[str, Any]:
        """Execute scroll action."""
        # Value is scroll amount (positive = up, negative = down)
        amount = int(action.value) if action.value else 3

        if action.coordinates and len(action.coordinates) >= 2:
            x, y = action.coordinates[0], action.coordinates[1]
            self._pyautogui.moveTo(x, y)

        self._pyautogui.scroll(amount)

        logger.debug("Scroll executed", amount=amount)
        return {"success": True, "action": "scroll", "amount": amount}

    async def _execute_wait(self, action: RPAAction) -> Dict[str, Any]:
        """Execute wait action."""
        wait_ms = action.timeout_ms
        await asyncio.sleep(wait_ms / 1000)

        logger.debug("Wait executed", ms=wait_ms)
        return {"success": True, "action": "wait", "ms": wait_ms}

    async def _execute_verify(self, action: RPAAction) -> Dict[str, Any]:
        """
        Verify screen state using vision OCR.

        Captures the screen and verifies that expected content is present.
        Uses DeepSeek-VL2 as primary, EasyOCR as fallback.

        Args:
            action: RPAAction with:
                - value: Expected text to find (substring match)
                - coordinates: Optional [x, y, w, h] region to verify
                - target_field: Field name being verified
        """
        expected_text = action.value
        if not expected_text:
            logger.warning("Verify action missing expected text", target=action.target_field)
            return {"success": True, "action": "verify", "verified": True, "skipped": True}

        try:
            # Capture screen region
            screen_capture = await self._get_screen_capture()

            region = None
            if action.coordinates and len(action.coordinates) >= 4:
                region = tuple(action.coordinates[:4])

            capture_result = await screen_capture.capture_screen(region=region)

            # Try verification with vision service first
            verified = False
            confidence = 0.0
            method = "none"
            extracted_text = ""

            # Attempt 1: DeepSeek-VL2 Vision
            vision_service = await self._get_vision_service()
            if vision_service and vision_service.is_ready:
                try:
                    prompt = f"Read the text in this image. Look for the text: '{expected_text}'. Return JSON: {{\"found\": true/false, \"confidence\": 0.0-1.0, \"extracted_text\": \"...\"}}"
                    result = await vision_service.extract_from_image(
                        capture_result.image,
                        prompt=prompt,
                    )
                    # Check if expected text is in extracted content
                    extracted_text = result.raw_text.lower()
                    if expected_text.lower() in extracted_text:
                        verified = True
                        confidence = result.confidence
                        method = "deepseek_vl2"
                    elif result.confidence >= VERIFICATION_CONFIDENCE_THRESHOLD:
                        # Check extracted fields
                        for field_data in result.extracted_fields.values():
                            if isinstance(field_data, dict) and expected_text.lower() in str(field_data.get("value", "")).lower():
                                verified = True
                                confidence = field_data.get("confidence", result.confidence)
                                method = "deepseek_vl2"
                                break
                except Exception as e:
                    logger.warning("Vision verification failed, trying OCR fallback", error=str(e))

            # Attempt 2: OCR Fallback (EasyOCR/Tesseract)
            if not verified:
                ocr_service = await self._get_ocr_fallback()
                if ocr_service:
                    try:
                        ocr_result = await ocr_service.extract_text(capture_result.image)
                        extracted_text = ocr_result.get("text", "").lower()
                        if expected_text.lower() in extracted_text:
                            verified = True
                            confidence = ocr_result.get("confidence", 0.8)
                            method = ocr_result.get("method", "ocr_fallback")
                    except Exception as e:
                        logger.warning("OCR fallback also failed", error=str(e))

            logger.debug(
                "Verify executed",
                target=action.target_field,
                verified=verified,
                confidence=confidence,
                method=method,
            )

            return {
                "success": True,
                "action": "verify",
                "verified": verified,
                "confidence": confidence,
                "method": method,
                "expected": expected_text,
                "found_in_text": expected_text.lower() in extracted_text if extracted_text else False,
            }

        except Exception as e:
            logger.error("Verify action failed", error=str(e), target=action.target_field)
            return {
                "success": False,
                "action": "verify",
                "verified": False,
                "error": str(e),
            }

    async def execute_action_with_retry(
        self,
        action: RPAAction,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_ms: int = DEFAULT_RETRY_DELAY_MS,
    ) -> Dict[str, Any]:
        """
        Execute an action with automatic retry on failure.

        Args:
            action: RPAAction to execute
            max_retries: Maximum retry attempts
            retry_delay_ms: Delay between retries in milliseconds

        Returns:
            Dict with execution result including retry info
        """
        last_error = None
        attempts = 0

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            try:
                result = await self.execute_action(action)

                if result.get("success"):
                    result["attempts"] = attempts
                    result["retried"] = attempt > 0
                    return result

                # Action returned success=False but no exception
                last_error = result.get("error", "Action failed")

                if attempt < max_retries:
                    logger.warning(
                        "Action failed, retrying",
                        action=action.action_type.value,
                        attempt=attempts,
                        max_retries=max_retries,
                        error=last_error,
                    )
                    await asyncio.sleep(retry_delay_ms / 1000)

            except Exception as e:
                last_error = str(e)

                if attempt < max_retries:
                    logger.warning(
                        "Action exception, retrying",
                        action=action.action_type.value,
                        attempt=attempts,
                        max_retries=max_retries,
                        error=last_error,
                    )
                    await asyncio.sleep(retry_delay_ms / 1000)

        # All retries exhausted
        logger.error(
            "Action failed after all retries",
            action=action.action_type.value,
            attempts=attempts,
            error=last_error,
        )

        return {
            "success": False,
            "action": action.action_type.value,
            "error": last_error,
            "attempts": attempts,
            "retried": True,
            "exhausted_retries": True,
        }

    async def execute_sequence(
        self,
        actions: List[RPAAction],
        stop_on_error: bool = True,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_ms: int = DEFAULT_RETRY_DELAY_MS,
        verify_after_type: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a sequence of RPA actions with retry logic.

        Args:
            actions: List of actions to execute
            stop_on_error: Stop execution on first error
            max_retries: Maximum retry attempts per action
            retry_delay_ms: Delay between retries in milliseconds
            verify_after_type: Auto-verify after TYPE actions

        Returns:
            Dict with execution results
        """
        results = []
        success_count = 0
        error_count = 0
        verification_failures = []

        for i, action in enumerate(actions):
            try:
                # Execute with retry
                result = await self.execute_action_with_retry(
                    action,
                    max_retries=max_retries,
                    retry_delay_ms=retry_delay_ms,
                )

                results.append({
                    "index": i,
                    "action": action.action_type.value,
                    "target": action.target_field,
                    **result,
                })

                if result.get("success"):
                    success_count += 1

                    # Auto-verify after TYPE actions if enabled
                    if verify_after_type and action.action_type == RPAActionType.TYPE and action.value:
                        # Create verification action for what we just typed
                        verify_action = RPAAction(
                            action_type=RPAActionType.VERIFY,
                            target_field=f"{action.target_field}_verify",
                            value=action.value[:50],  # First 50 chars for verification
                            coordinates=action.coordinates,
                        )
                        verify_result = await self.execute_action(verify_action)

                        if not verify_result.get("verified", False):
                            verification_failures.append({
                                "index": i,
                                "target": action.target_field,
                                "expected": action.value[:50],
                                "verification_result": verify_result,
                            })
                            logger.warning(
                                "Post-type verification failed",
                                target=action.target_field,
                                result=verify_result,
                            )
                else:
                    error_count += 1
                    if stop_on_error:
                        break

                # Small delay between actions
                await asyncio.sleep(0.1)

            except Exception as e:
                error_count += 1
                results.append({
                    "index": i,
                    "action": action.action_type.value,
                    "target": action.target_field,
                    "success": False,
                    "error": str(e),
                })
                if stop_on_error:
                    break

        return {
            "total": len(actions),
            "executed": success_count + error_count,
            "success_count": success_count,
            "error_count": error_count,
            "results": results,
            "completed": error_count == 0,
            "verification_failures": verification_failures,
            "all_verified": len(verification_failures) == 0,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_rpa_executor: Optional[RPAExecutor] = None


def get_rpa_executor() -> RPAExecutor:
    """Get singleton RPA executor."""
    global _rpa_executor
    if _rpa_executor is None:
        _rpa_executor = RPAExecutor()
    return _rpa_executor
