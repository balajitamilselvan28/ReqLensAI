import uuid
import logging
from typing import List, Dict, Optional, Set, Tuple

from src.schemas.data_models import (
    Requirement, CandidatePair, KnowledgeRepresentation, ReasoningResult,
    Finding, FindingStatus, ContradictionType, Severity
)

logger = logging.getLogger(__name__)

# Confidence score weights — each piece of deterministic evidence adds weight
_WEIGHT_ACTOR_CONFLICT = 0.50
_WEIGHT_ACTION_OVERLAP = 0.25
_WEIGHT_NUMERICAL_OVERLAP = 0.30
_WEIGHT_CONDITIONAL = 0.15
_WEIGHT_HIGH_SEMANTIC_SIM = 0.10
_WEIGHT_SAME_ACTION_SAME_OBJECT = 0.20

# Thresholds
_VERIFY_THRESHOLD = 0.60


class ContradictionDetectionAgent:
    """
    Consumes outputs from the Semantic Retrieval, Knowledge Representation,
    and Logical Reasoning agents to detect potential inconsistencies.

    IMPORTANT: This agent classifies *potential* and *verified* conflicts.
    It does NOT automatically modify requirements or generate recommendations.
    Final human judgment is reserved for the Human Verification stage.
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: Optional LangChain-compatible LLM for additional semantic
                 classification. If None the agent uses deterministic rules only.
        """
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        requirements: List[Requirement],
        candidate_pairs: List[CandidatePair],
        reasoning_results: List[ReasoningResult],
        knowledge_map: Optional[Dict[str, KnowledgeRepresentation]] = None,
    ) -> List[Finding]:
        """
        Detect potential contradictions from structured evidence.

        Args:
            requirements:     Structured Requirement objects (Version 2).
            candidate_pairs:  Semantically similar pairs (Version 3).
            reasoning_results:Logical reasoning evidence (Version 4).
            knowledge_map:    Optional {req_id: KnowledgeRepresentation} (Version 4).

        Returns:
            List of Finding objects (status = potential or verified).
        """
        req_map: Dict[str, Requirement] = {
            r.requirement_id: r for r in requirements if r.requirement_id
        }
        reasoning_map: Dict[Tuple[str, str], ReasoningResult] = {}
        for rr in reasoning_results:
            key = tuple(sorted([rr.requirement_id_1, rr.requirement_id_2]))
            reasoning_map[key] = rr

        findings: List[Finding] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for pair in candidate_pairs:
            id1, id2 = pair.requirement_id_1, pair.requirement_id_2
            canonical = tuple(sorted([id1, id2]))
            if canonical in seen_pairs:
                continue
            seen_pairs.add(canonical)

            req1 = req_map.get(id1)
            req2 = req_map.get(id2)
            if not req1 or not req2:
                continue

            rr = reasoning_map.get(canonical)
            finding = self._classify(req1, req2, pair, rr)
            if finding:
                findings.append(finding)

        # Deduplicate by (canonical_pair, contradiction_type)
        return self._deduplicate(findings)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        req1: Requirement,
        req2: Requirement,
        pair: CandidatePair,
        rr: Optional[ReasoningResult],
    ) -> Optional[Finding]:
        """Apply deterministic classification rules and compute confidence."""

        id1, id2 = req1.requirement_id, req2.requirement_id
        canonical_id1, canonical_id2 = (
            (id1, id2) if id1 < id2 else (id2, id1)  # type: ignore[operator]
        )

        confidence = 0.0
        contradiction_type = ContradictionType.UNKNOWN
        detail_parts: List[str] = []

        # --- Rule 1: Permission conflict ---
        if rr and rr.has_overlapping_actions and rr.has_different_actors:
            contradiction_type = ContradictionType.PERMISSION_CONFLICT
            confidence += _WEIGHT_ACTION_OVERLAP + _WEIGHT_ACTOR_CONFLICT
            detail_parts.append(
                f"Both requirements govern the same action but specify different authorized actors "
                f"('{req1.actor}' vs '{req2.actor}')."
            )

        # --- Rule 2: Numerical conflict ---
        elif rr and rr.has_numerical_overlap and rr.has_overlapping_actions:
            contradiction_type = ContradictionType.NUMERICAL_CONFLICT
            confidence += _WEIGHT_NUMERICAL_OVERLAP + _WEIGHT_ACTION_OVERLAP
            detail_parts.append(
                f"Both requirements constrain the same action with different numerical values: "
                f"{rr.numerical_comparison}."
            )

        # --- Rule 3: Conditional conflict ---
        elif rr and rr.has_overlapping_actions and rr.has_conditional_implication:
            contradiction_type = ContradictionType.CONDITIONAL_CONFLICT
            confidence += _WEIGHT_ACTION_OVERLAP + _WEIGHT_CONDITIONAL
            detail_parts.append(
                "Both requirements share the same action but differ in conditional triggers."
            )

        # --- Rule 4: Redundancy (high similarity, same action+object, same actor) ---
        elif (
            rr
            and rr.has_overlapping_actions
            and not rr.has_different_actors
            and not rr.has_numerical_overlap
            and pair.similarity_score >= 0.85
        ):
            contradiction_type = ContradictionType.REDUNDANCY
            confidence += pair.similarity_score * 0.5 + _WEIGHT_SAME_ACTION_SAME_OBJECT
            detail_parts.append(
                f"The requirements appear substantially equivalent "
                f"(semantic similarity: {pair.similarity_score:.2f})."
            )

        # --- Rule 5: Semantic-only — insufficient deterministic evidence ---
        elif pair.similarity_score >= 0.70:
            contradiction_type = ContradictionType.UNKNOWN
            confidence += _WEIGHT_HIGH_SEMANTIC_SIM
            detail_parts.append(
                f"High semantic similarity ({pair.similarity_score:.2f}) detected but no "
                "deterministic structural conflict was identified."
            )

        # Not enough signal — skip
        if contradiction_type == ContradictionType.UNKNOWN and confidence < 0.15:
            return None

        # Clamp confidence
        confidence = min(confidence, 1.0)

        # Determine severity
        severity = self._assign_severity(contradiction_type, confidence)

        # Determine status
        status = (
            FindingStatus.VERIFIED if confidence >= _VERIFY_THRESHOLD
            else FindingStatus.POTENTIAL
        )

        explanation = self._build_explanation(req1, req2, contradiction_type, detail_parts)
        evidence = (
            f"[{id1}]: \"{req1.original_text}\"\n"
            f"[{id2}]: \"{req2.original_text}\"\n"
            f"Reasoning: {rr.reasoning_details if rr else 'No structured reasoning available.'}"
        )

        return Finding(
            finding_id=f"F-{uuid.uuid4().hex[:8].upper()}",
            requirement_id_1=canonical_id1,
            requirement_id_2=canonical_id2,
            contradiction_type=contradiction_type,
            severity=severity,
            confidence_score=round(confidence, 3),
            explanation=explanation,
            evidence=evidence,
            status=status,
        )

    def _assign_severity(self, ctype: ContradictionType, confidence: float) -> Severity:
        """Deterministically assign severity based on type and confidence."""
        if ctype in (ContradictionType.PERMISSION_CONFLICT, ContradictionType.DIRECT_CONFLICT):
            return Severity.HIGH
        if ctype == ContradictionType.NUMERICAL_CONFLICT:
            return Severity.HIGH if confidence >= 0.5 else Severity.MEDIUM
        if ctype == ContradictionType.CONDITIONAL_CONFLICT:
            return Severity.MEDIUM
        if ctype == ContradictionType.REDUNDANCY:
            return Severity.LOW
        return Severity.LOW

    def _build_explanation(
        self,
        req1: Requirement,
        req2: Requirement,
        ctype: ContradictionType,
        detail_parts: List[str],
    ) -> str:
        prefix_map = {
            ContradictionType.PERMISSION_CONFLICT: (
                f"{req1.requirement_id} and {req2.requirement_id} govern the same action "
                "but specify incompatible authorization conditions."
            ),
            ContradictionType.NUMERICAL_CONFLICT: (
                f"{req1.requirement_id} and {req2.requirement_id} both impose numerical "
                "constraints on the same action, with differing values."
            ),
            ContradictionType.CONDITIONAL_CONFLICT: (
                f"{req1.requirement_id} and {req2.requirement_id} address the same action "
                "under conditions that may be logically incompatible."
            ),
            ContradictionType.REDUNDANCY: (
                f"{req1.requirement_id} and {req2.requirement_id} appear to describe "
                "substantially the same requirement."
            ),
            ContradictionType.UNKNOWN: (
                f"{req1.requirement_id} and {req2.requirement_id} are semantically related "
                "but the nature of their relationship could not be fully determined."
            ),
        }
        base = prefix_map.get(ctype, "A potential inconsistency was detected.")
        details = " ".join(detail_parts)
        return f"{base} {details}".strip()

    def _deduplicate(self, findings: List[Finding]) -> List[Finding]:
        """Remove duplicate findings for the same (canonical pair, contradiction_type)."""
        seen: Set[Tuple[str, str, str]] = set()
        unique: List[Finding] = []
        for f in findings:
            key = (f.requirement_id_1, f.requirement_id_2, f.contradiction_type.value)
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique
