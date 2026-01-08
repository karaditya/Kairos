"""
RPA Coordinator for CAE System.

Orchestrates the complete RPA flow:
1. Build actions from compte rendu
2. Request human verification
3. Execute approved actions
"""

from typing import Optional, List, Dict, Any

from models.rpa_action import RPAAction, RPAActionType, RPAVerification
from models.compte_rendu import CompteRendu
from services.rpa.actions import RPAExecutor, get_rpa_executor
from services.rpa.coordinate_map import CoordinateMapManager, CoordinateMap, get_coordinate_manager
from services.rpa.safety import RPASafetyGate, get_safety_gate
from config import COMPTE_RENDU_SECTIONS
from utils.logging import get_logger
from utils.exceptions import RPAError, CoordinateMapError

logger = get_logger(__name__)


class RPACoordinator:
    """
    Coordinates RPA operations for EHR sync.

    Full flow:
    1. prepare_sync() - Build actions from CR + coordinate map
    2. request_sync() - Create verification request (requires approval)
    3. execute_sync() - Execute approved actions
    """

    def __init__(
        self,
        executor: RPAExecutor,
        coord_manager: CoordinateMapManager,
        safety_gate: RPASafetyGate,
    ):
        self.executor = executor
        self.coord_manager = coord_manager
        self.safety_gate = safety_gate

    async def prepare_sync(
        self,
        compte_rendu: CompteRendu,
        coordinate_map: CoordinateMap,
        target_fields: Optional[List[str]] = None,
    ) -> List[RPAAction]:
        """
        Prepare RPA actions from compte rendu.

        Args:
            compte_rendu: Compte rendu to sync
            coordinate_map: EHR coordinate mapping
            target_fields: Specific fields to sync (defaults to all)

        Returns:
            List of RPAAction objects
        """
        if target_fields is None:
            target_fields = list(compte_rendu.sections.keys())

        actions = []

        for field_name in target_fields:
            # Get field content
            content = compte_rendu.sections.get(field_name, "")
            if not content:
                continue

            # Get coordinates
            field_coord = coordinate_map.get_field(field_name)
            if not field_coord:
                logger.warning("No coordinates for field", field=field_name)
                continue

            # Create actions: click, clear, type
            click_point = field_coord.center

            # Click to focus
            actions.append(RPAAction(
                action_type=RPAActionType.CLICK,
                target_field=field_name,
                coordinates=list(click_point),
            ))

            # Clear existing content
            actions.append(RPAAction(
                action_type=RPAActionType.CLEAR,
                target_field=field_name,
                coordinates=list(click_point),
            ))

            # Type new content
            actions.append(RPAAction(
                action_type=RPAActionType.TYPE,
                target_field=field_name,
                value=content,
                coordinates=list(click_point),
            ))

        # Add save button click if mapped
        save_coord = coordinate_map.get_field("save_button")
        if save_coord:
            actions.append(RPAAction(
                action_type=RPAActionType.CLICK,
                target_field="save_button",
                coordinates=list(save_coord.center),
            ))

        logger.info(
            "Sync actions prepared",
            compte_rendu_id=compte_rendu.id,
            action_count=len(actions),
            target_fields=target_fields,
        )

        return actions

    async def request_sync(
        self,
        session_id: str,
        compte_rendu: CompteRendu,
        ehr_type: str,
        map_name: str,
        target_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Request sync with human verification.

        Args:
            session_id: Session ID
            compte_rendu: Compte rendu to sync
            ehr_type: EHR type for coordinate lookup
            map_name: Coordinate map name
            target_fields: Specific fields to sync

        Returns:
            Dict with verification_id and preview
        """
        # Load coordinate map
        coord_map = await self.coord_manager.load_map(ehr_type, map_name)
        if not coord_map:
            raise CoordinateMapError(
                map_name,
                f"Coordinate map not found: {ehr_type}/{map_name}",
            )

        # Prepare actions
        actions = await self.prepare_sync(compte_rendu, coord_map, target_fields)

        if not actions:
            raise RPAError("No actions to execute - no fields mapped")

        # Create verification request
        verification = await self.safety_gate.request_verification(
            session_id=session_id,
            compte_rendu=compte_rendu,
            actions=actions,
            target_fields=target_fields or list(compte_rendu.sections.keys()),
        )

        # Generate preview
        preview = self.safety_gate.generate_preview(compte_rendu, actions)

        return {
            "status": "pending_approval",
            "verification_id": verification.id,
            "preview": preview,
        }

    async def execute_sync(
        self,
        verification_id: str,
    ) -> Dict[str, Any]:
        """
        Execute approved sync.

        Args:
            verification_id: Approved verification ID

        Returns:
            Execution result
        """
        # Validate approval
        verification = await self.safety_gate.validate_for_execution(verification_id)

        # Apply modifications if any
        actions = [
            RPAAction.from_dict(a) for a in
            (await self._get_verification_actions(verification_id))
        ]

        # Execute
        result = await self.executor.execute_sequence(actions)

        # Mark executed
        await self.safety_gate.mark_executed(
            verification_id=verification_id,
            success=result["completed"],
            result=str(result),
        )

        logger.info(
            "Sync executed",
            verification_id=verification_id,
            success=result["completed"],
            actions_executed=result["executed"],
        )

        return {
            "status": "completed" if result["completed"] else "failed",
            "actions_executed": result["executed"],
            "success_count": result["success_count"],
            "error_count": result["error_count"],
            "timestamp": verification.executed_at.isoformat() if verification.executed_at else None,
            "errors": [
                r["error"] for r in result["results"]
                if not r.get("success") and r.get("error")
            ],
        }

    async def _get_verification_actions(
        self,
        verification_id: str,
    ) -> List[Dict[str, Any]]:
        """Get actions from verification record."""
        from db.database import RPAVerificationRepository

        verification_data = await RPAVerificationRepository.get(verification_id)
        if not verification_data:
            return []

        return verification_data.get("actions", [])

    async def get_pending(
        self,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get pending verification requests."""
        verifications = await self.safety_gate.get_pending_verifications(session_id)
        return [v.to_dict() for v in verifications]


# =============================================================================
# FACTORY
# =============================================================================

_rpa_coordinator: Optional[RPACoordinator] = None


def get_rpa_coordinator() -> RPACoordinator:
    """Get singleton RPA coordinator."""
    global _rpa_coordinator
    if _rpa_coordinator is None:
        executor = get_rpa_executor()
        coord_manager = get_coordinate_manager()
        safety_gate = get_safety_gate()
        _rpa_coordinator = RPACoordinator(executor, coord_manager, safety_gate)
    return _rpa_coordinator
