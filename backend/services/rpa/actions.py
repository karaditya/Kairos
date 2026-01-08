"""
RPA Action Executor for CAE System.

Executes atomic RPA actions using pyautogui.
Cross-platform support (Windows, Mac, Linux).
"""

from typing import Optional, List, Dict, Any
import asyncio
import time

from config import RPA_ENABLED, RPA_TYPING_INTERVAL, RPA_CLICK_PAUSE, RPA_FAILSAFE
from models.rpa_action import RPAAction, RPAActionType
from utils.logging import get_logger
from utils.exceptions import RPAExecutionError, RPAError

logger = get_logger(__name__)


class RPAExecutor:
    """
    Executes RPA actions using pyautogui.

    All actions require prior human approval (enforced by RPASafetyGate).
    """

    def __init__(self):
        self._initialized = False
        self._pyautogui = None

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
        Verify screen state (placeholder for future OCR verification).

        Currently just returns success - actual verification would
        capture screen and verify expected content.
        """
        logger.debug("Verify executed (placeholder)", target=action.target_field)
        return {"success": True, "action": "verify", "verified": True}

    async def execute_sequence(
        self,
        actions: List[RPAAction],
        stop_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a sequence of RPA actions.

        Args:
            actions: List of actions to execute
            stop_on_error: Stop execution on first error

        Returns:
            Dict with execution results
        """
        results = []
        success_count = 0
        error_count = 0

        for i, action in enumerate(actions):
            try:
                result = await self.execute_action(action)
                results.append({
                    "index": i,
                    "action": action.action_type.value,
                    "target": action.target_field,
                    **result,
                })

                if result.get("success"):
                    success_count += 1
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
