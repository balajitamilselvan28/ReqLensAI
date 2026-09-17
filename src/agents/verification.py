"""Verification Agent — Version 6.

Independently evaluates a generated Recommendation against its associated Finding
to detect unsupported claims, validate traceability, and flag items requiring
human analyst review.

Design principles:
- Conservative: ambiguous cases default to needs_review, never auto-approved.
- Never replaces the human analyst.
- Never modifies requirement text.
- Produces structured VerificationResult objects for the HITL layer.
"""
import logging
from typing import List

from src.schemas.data_models import (
    Finding, Recommendation, VerificationResult, VerificationStatus, ContradictionType
)

logger = logging.getLogger(__name__)

# Threshold above which a recommendation can be APPROVED automatically
_APPROVE_THRESHOLD = 0.70
# Threshold below which a recommendation is REJECTED
_REJECT_THRESHOLD = 0.25


class VerificationAgent:
    """
    Independently verifies recommendations produced by the
    RecommendationGenerationAgent before they are presented to the
    human analyst.

    IMPORTANT: Even a fully APPROVED recommendation still carries
    requires_human_review=True when appropriate. The human remains
    the final decision-maker.
    """

    def verify(
        self,
        recommendations: List[Recommendation],
        findings: List[Finding],
    ) -> List[VerificationResult]:
        """
        Verify a list of recommendations against their source findings.

        Args:
            recommendations: Output of RecommendationGenerationAgent.
            findings:        Source findings (used for traceability checks).

        Returns:
            List of VerificationResult objects.
        """
        finding_map = {f.finding_id: f for f in findings}
        results: List[VerificationResult] = []

        for rec in recommendations:
            finding = finding_map.get(rec.finding_id)
            result = self._verify_one(rec, finding)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _verify_one(
        self,
        rec: Recommendation,
        finding: Finding | None,
    ) -> VerificationResult:
        """Run all verification checks on a single recommendation."""
        verified_claims: List[str] = []
        warnings: List[str] = []
        score = 1.0  # Start at full confidence; deduct for each issue

        # --- Check 1: Finding exists (traceability) ---
        if finding is None:
            warnings.append(
                f"No source finding found for finding_id='{rec.finding_id}'. "
                "Traceability cannot be established."
            )
            score -= 0.50
        else:
            verified_claims.append("Source finding is traceable.")

        # --- Check 2: Requirement IDs are present ---
        if rec.requirement_id_1 and rec.requirement_id_2:
            verified_claims.append("Both requirement IDs are present.")
        else:
            warnings.append("One or both requirement IDs are missing.")
            score -= 0.20

        # --- Check 3: Contradiction type is a valid enum value ---
        try:
            ContradictionType(rec.contradiction_type)
            verified_claims.append("Contradiction type is a recognised category.")
        except ValueError:
            warnings.append(f"Contradiction type '{rec.contradiction_type}' is not a recognised category.")
            score -= 0.15

        # --- Check 4: Confidence score is in [0, 1] ---
        if 0.0 <= rec.confidence_score <= 1.0:
            verified_claims.append("Confidence score is within valid bounds [0, 1].")
        else:
            warnings.append(f"Confidence score {rec.confidence_score} is out of bounds.")
            score -= 0.15

        # --- Check 5: Recommendation text exists and is non-trivial ---
        if rec.recommendation_text and len(rec.recommendation_text.strip()) > 20:
            verified_claims.append("Recommendation text is present and non-trivial.")
        else:
            warnings.append("Recommendation text is missing or too short.")
            score -= 0.20

        # --- Check 6: Recommendation does not appear to modify requirements ---
        modification_keywords = [
            "change requirement", "rewrite requirement", "delete requirement",
            "remove requirement", "modify requirement", "replace requirement",
            "automatically update", "overwrite"
        ]
        rec_lower = rec.recommendation_text.lower() if rec.recommendation_text else ""
        if any(kw in rec_lower for kw in modification_keywords):
            warnings.append(
                "Recommendation text appears to suggest automatic SRS modification, "
                "which is not permitted."
            )
            score -= 0.30
        else:
            verified_claims.append("Recommendation does not propose automatic SRS modification.")

        # --- Check 7: Evidence is present ---
        if rec.evidence and len(rec.evidence.strip()) > 5:
            verified_claims.append("Evidence string is present.")
        else:
            warnings.append("Evidence field is empty or too short; traceability is reduced.")
            score -= 0.10

        # --- Check 8: Recommendation references the source finding context ---
        has_context = (
            (rec.requirement_id_1 in rec.recommendation_text or
             rec.requirement_id_2 in rec.recommendation_text)
            if rec.recommendation_text else False
        )
        if has_context:
            verified_claims.append("Recommendation references requirement IDs from the finding.")
        else:
            # Not a hard failure — templates use IDs by default, but log it
            warnings.append(
                "Recommendation text does not explicitly reference the requirement IDs. "
                "Consider adding context for the analyst."
            )

        # Clamp score
        score = max(round(score, 3), 0.0)

        # Determine status
        if score >= _APPROVE_THRESHOLD and not warnings:
            status = VerificationStatus.APPROVED
            requires_human = False
        elif score < _REJECT_THRESHOLD:
            status = VerificationStatus.REJECTED
            requires_human = True
        else:
            status = VerificationStatus.NEEDS_REVIEW
            requires_human = True

        # High-severity findings always require human review
        from src.schemas.data_models import Severity
        if finding and finding.severity == Severity.HIGH:
            requires_human = True

        return VerificationResult(
            finding_id=rec.finding_id,
            verification_status=status,
            verification_confidence=score,
            verified_claims=verified_claims,
            warnings=warnings,
            final_recommendation=rec.recommendation_text or "",
            requires_human_review=requires_human,
        )
