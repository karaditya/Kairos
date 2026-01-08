"""
RPA Safety Gate for CAE System.

Enforces human-in-the-loop verification for all RPA actions.
This is a NON-NEGOTIABLE safety requirement.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from models.rpa_action import RPAAction, RPAVerification, RPAVerificationStatus
from models.compte_rendu import CompteRendu
from db.database import RPAVerificationRepository
from config import REQUIRE_HUMAN_VERIFICATION
from utils.logging import get_logger
from utils.exceptions import RPANotApprovedError, RPAError

logger = get_logger(__name__)


class RPASafetyGate:
    """
    Safety gate for RPA operations.

    ALL RPA sync operations MUST go through this gate.
    Human verification is ALWAYS required (non-configurable).
    """

    def __init__(self):
        # Safety flag is hardcoded - cannot be disabled
        self._require_verification = True

        # Log safety status on init
        logger.info(
            "RPA Safety Gate initialized",
            require_verification=self._require_verification,
        )

    async def request_verification(
        self,
        session_id: str,
        compte_rendu: CompteRendu,
        actions: List[RPAAction],
        target_fields: List[str],
    ) -> RPAVerification:
        """
        Create verification request for RPA actions.

        This creates a pending verification that MUST be approved
        before actions can be executed.

        Args:
            session_id: Session ID
            compte_rendu: Compte rendu to sync
            actions: List of RPA actions to execute
            target_fields: Target field names

        Returns:
            RPAVerification in pending state
        """
        # Create verification record
        verification_data = await RPAVerificationRepository.create(
            session_id=session_id,
            compte_rendu_id=compte_rendu.id,
            actions=[a.to_dict() for a in actions],
            target_fields=target_fields,
        )

        verification = RPAVerification.from_dict(verification_data)

        logger.info(
            "Verification request created",
            verification_id=verification.id,
            session_id=session_id,
            action_count=len(actions),
        )

        return verification

    async def check_approval(self, verification_id: str) -> bool:
        """
        Check if verification has been approved.

        Args:
            verification_id: Verification ID to check

        Returns:
            True if approved, False otherwise
        """
        verification_data = await RPAVerificationRepository.get(verification_id)
        if not verification_data:
            return False

        return verification_data.get("status") == "approved"

    async def approve(
        self,
        verification_id: str,
        user_id: str,
        modifications: Optional[Dict[str, str]] = None,
    ) -> RPAVerification:
        """
        Approve a verification request.

        Args:
            verification_id: Verification ID to approve
            user_id: ID of approving user
            modifications: Optional field value modifications

        Returns:
            Updated RPAVerification
        """
        # Get current verification
        verification_data = await RPAVerificationRepository.get(verification_id)
        if not verification_data:
            raise RPAError(f"Verification not found: {verification_id}")

        if verification_data.get("status") != "pending":
            raise RPAError(f"Verification not pending: {verification_id}")

        # Approve
        await RPAVerificationRepository.approve(
            verification_id=verification_id,
            approved_by=user_id,
            modifications=modifications,
        )

        # Get updated
        updated_data = await RPAVerificationRepository.get(verification_id)

        logger.info(
            "Verification approved",
            verification_id=verification_id,
            approved_by=user_id,
        )

        return RPAVerification.from_dict(updated_data)

    async def reject(
        self,
        verification_id: str,
        user_id: str,
    ) -> RPAVerification:
        """
        Reject a verification request.

        Args:
            verification_id: Verification ID to reject
            user_id: ID of rejecting user

        Returns:
            Updated RPAVerification
        """
        await RPAVerificationRepository.reject(
            verification_id=verification_id,
            rejected_by=user_id,
        )

        updated_data = await RPAVerificationRepository.get(verification_id)

        logger.info(
            "Verification rejected",
            verification_id=verification_id,
            rejected_by=user_id,
        )

        return RPAVerification.from_dict(updated_data)

    async def validate_for_execution(
        self,
        verification_id: str,
    ) -> RPAVerification:
        """
        Validate verification is ready for execution.

        Args:
            verification_id: Verification ID to validate

        Returns:
            RPAVerification if approved

        Raises:
            RPANotApprovedError if not approved
        """
        verification_data = await RPAVerificationRepository.get(verification_id)
        if not verification_data:
            raise RPAError(f"Verification not found: {verification_id}")

        verification = RPAVerification.from_dict(verification_data)

        if not verification.is_approved:
            raise RPANotApprovedError(verification_id)

        return verification

    async def mark_executed(
        self,
        verification_id: str,
        success: bool,
        result: Optional[str] = None,
    ) -> RPAVerification:
        """
        Mark verification as executed.

        Args:
            verification_id: Verification ID
            success: Whether execution succeeded
            result: Optional result message

        Returns:
            Updated RPAVerification
        """
        await RPAVerificationRepository.mark_executed(
            verification_id=verification_id,
            success=success,
            result=result,
        )

        updated_data = await RPAVerificationRepository.get(verification_id)

        logger.info(
            "Verification marked executed",
            verification_id=verification_id,
            success=success,
        )

        return RPAVerification.from_dict(updated_data)

    async def get_pending_verifications(
        self,
        session_id: Optional[str] = None,
    ) -> List[RPAVerification]:
        """
        Get pending verification requests.

        Args:
            session_id: Optional filter by session

        Returns:
            List of pending verifications
        """
        pending_data = await RPAVerificationRepository.get_pending(session_id)
        return [RPAVerification.from_dict(v) for v in pending_data]

    def generate_preview(
        self,
        compte_rendu: CompteRendu,
        actions: List[RPAAction],
    ) -> Dict[str, Any]:
        """
        Generate preview of RPA actions for UI display.

        Args:
            compte_rendu: Compte rendu being synced
            actions: Actions to execute

        Returns:
            Preview dict for UI
        """
        action_previews = []
        for action in actions:
            preview = {
                "action": action.action_type.value,
                "target": action.target_field,
            }

            if action.value:
                # Truncate long values for preview
                value = action.value
                if len(value) > 100:
                    value = value[:100] + "..."
                preview["value"] = value

            if action.coordinates:
                preview["coordinates"] = action.coordinates

            action_previews.append(preview)

        return {
            "compte_rendu_id": compte_rendu.id,
            "patient_name": compte_rendu.patient_name,
            "sections_to_sync": list(compte_rendu.sections.keys()),
            "action_count": len(actions),
            "actions": action_previews,
            "requires_approval": True,
            "warning": "All fields will be typed into the EHR. Verify data before approval.",
        }


# =============================================================================
# SINGLETON
# =============================================================================

_safety_gate: Optional[RPASafetyGate] = None


def get_safety_gate() -> RPASafetyGate:
    """Get singleton RPA safety gate."""
    global _safety_gate
    if _safety_gate is None:
        _safety_gate = RPASafetyGate()
    return _safety_gate
