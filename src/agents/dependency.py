import logging
from typing import List, Optional

from src.schemas.data_models import (
    Finding, FindingStatus, DependencyResult, DependencyStatus, Severity
)

logger = logging.getLogger(__name__)

# Confidence adjustment constants
_BOOST_SUPPORTED = 0.20
_PENALTY_CONDITIONAL = 0.10
_PENALTY_UNRELATED = 0.40
_VERIFY_THRESHOLD = 0.60
_REJECT_THRESHOLD = 0.20


class DependencyVerificationAgent:
    """
    Examines potential findings from the Contradiction Detection Agent and
    validates them against structural dependency evidence.

    Purpose: Reduce false positives. A finding is only promoted to 'verified'
    when there is sufficient structural evidence. Findings can be 'rejected'
    when dependency evidence explains away the apparent conflict.

    IMPORTANT: This agent does NOT generate recommendations. It only
    adjusts finding status and adds dependency context.
    """

    def verify(
        self,
        findings: List[Finding],
        all_findings: Optional[List[Finding]] = None,
    ) -> tuple[List[Finding], List[DependencyResult]]:
        """
        Verify findings using dependency analysis.

        Args:
            findings:     Potential findings from ContradictionDetectionAgent.
            all_findings: Full finding list (for cross-reference). Defaults to findings.

        Returns:
            Tuple of (updated findings list, dependency results list).
        """
        if all_findings is None:
            all_findings = findings

        dep_results: List[DependencyResult] = []
        updated_findings: List[Finding] = []

        for finding in findings:
            dep = self._analyze_dependency(finding, all_findings)
            dep_results.append(dep)
            updated = self._apply_dependency(finding, dep)
            updated_findings.append(updated)

        return updated_findings, dep_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyze_dependency(
        self,
        finding: Finding,
        all_findings: List[Finding],
    ) -> DependencyResult:
        """
        Derive dependency status deterministically from the finding's
        contradiction_type, severity, and confidence.
        """
        from src.schemas.data_models import ContradictionType

        id1, id2 = finding.requirement_id_1, finding.requirement_id_2
        evidence_parts: List[str] = [finding.evidence]

        # --- Rule A: High-confidence structured conflicts are supported ---
        if (
            finding.confidence_score >= _VERIFY_THRESHOLD
            and finding.contradiction_type
            in (
                ContradictionType.PERMISSION_CONFLICT,
                ContradictionType.NUMERICAL_CONFLICT,
                ContradictionType.DIRECT_CONFLICT,
            )
        ):
            status = DependencyStatus.SUPPORTED
            dep_confidence = min(finding.confidence_score + 0.10, 1.0)
            explanation = (
                f"Structural evidence sufficiently supports this {finding.contradiction_type.value}. "
                "The finding is backed by deterministic reasoning."
            )

        # --- Rule B: Conditional conflicts — scope may limit the conflict ---
        elif finding.contradiction_type == ContradictionType.CONDITIONAL_CONFLICT:
            status = DependencyStatus.CONDITIONAL
            dep_confidence = max(finding.confidence_score - _PENALTY_CONDITIONAL, 0.0)
            explanation = (
                "The conflict appears conditional. One or both requirements may apply only under "
                "specific conditions, which could limit or negate the conflict in practice."
            )

        # --- Rule C: Redundancy is inherently a weak dependency ---
        elif finding.contradiction_type == ContradictionType.REDUNDANCY:
            status = DependencyStatus.CONDITIONAL
            dep_confidence = finding.confidence_score
            explanation = (
                "The requirements appear redundant. They may intentionally overlap in scope "
                "or be duplicated across sections."
            )

        # --- Rule D: Unknown / low-confidence — insufficient information ---
        elif (
            finding.contradiction_type == ContradictionType.UNKNOWN
            or finding.confidence_score < 0.25
        ):
            status = DependencyStatus.INSUFFICIENT_INFORMATION
            dep_confidence = max(finding.confidence_score - 0.05, 0.0)
            explanation = (
                "Insufficient structural evidence to confirm or reject this finding. "
                "Semantic similarity alone does not establish a dependency."
            )

        # --- Rule E: Unrelated (low confidence, no structural match) ---
        else:
            status = DependencyStatus.UNRELATED
            dep_confidence = max(finding.confidence_score - _PENALTY_UNRELATED, 0.0)
            explanation = (
                "No clear dependency or conflict relationship was found between these requirements."
            )

        return DependencyResult(
            requirement_id_1=id1,
            requirement_id_2=id2,
            dependency_status=status,
            explanation=explanation,
            supporting_evidence="\n".join(evidence_parts),
            confidence_score=round(dep_confidence, 3),
        )

    def _apply_dependency(self, finding: Finding, dep: DependencyResult) -> Finding:
        """
        Apply dependency result to update the finding's status and confidence.
        """
        new_confidence = finding.confidence_score

        if dep.dependency_status == DependencyStatus.SUPPORTED:
            new_confidence = min(new_confidence + _BOOST_SUPPORTED, 1.0)
            new_status = FindingStatus.VERIFIED

        elif dep.dependency_status == DependencyStatus.CONDITIONAL:
            new_confidence = max(new_confidence - _PENALTY_CONDITIONAL, 0.0)
            new_status = (
                FindingStatus.POTENTIAL
                if new_confidence >= _REJECT_THRESHOLD
                else FindingStatus.REJECTED
            )

        elif dep.dependency_status == DependencyStatus.UNRELATED:
            new_confidence = max(new_confidence - _PENALTY_UNRELATED, 0.0)
            new_status = FindingStatus.REJECTED

        else:  # INSUFFICIENT_INFORMATION
            # Preserve finding but keep it potential (never auto-verify)
            new_status = FindingStatus.POTENTIAL

        # Return a copy with updated fields (Pydantic model_copy in v2)
        return finding.model_copy(
            update={
                "confidence_score": round(new_confidence, 3),
                "status": new_status,
            }
        )
