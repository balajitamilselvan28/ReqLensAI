"""Recommendation Generation Agent — Version 6.

Consumes verified findings from the Dependency Verification Agent (Version 5)
and produces explainable, context-aware recommendations for a requirements engineer.

Design principles:
- Never automatically modifies any requirement.
- Never invents actors, constraints, or system behavior.
- Provides deterministic fallback for all supported contradiction types.
- Optionally uses an LLM for richer language; falls back gracefully on failure.
"""
import logging
from typing import List, Optional, Dict

from src.schemas.data_models import (
    Finding, ContradictionType, Severity, Recommendation, DependencyResult
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic recommendation templates (keyed by contradiction type)
# ---------------------------------------------------------------------------
_TEMPLATES: Dict[ContradictionType, str] = {
    ContradictionType.PERMISSION_CONFLICT: (
        "Review the authorization rules governing this action and clarify which roles "
        "({req1} or {req2}) are permitted, restricted, or both. Confirm whether the two "
        "requirements address different scopes or represent an incompatible overlap."
    ),
    ContradictionType.NUMERICAL_CONFLICT: (
        "Review the numerical constraints specified by {req1} and {req2} and clarify the "
        "applicable operating conditions. Determine whether one limit supersedes the other, "
        "whether both can be satisfied simultaneously, or whether they apply to different contexts."
    ),
    ContradictionType.CONDITIONAL_CONFLICT: (
        "Review the conditions under which {req1} and {req2} each apply. Clarify whether "
        "the requirements govern different system states or whether their conditional triggers "
        "can overlap, potentially causing conflicting behavior."
    ),
    ContradictionType.REDUNDANCY: (
        "Review {req1} and {req2} for duplicate intent. If they express the same behavior, "
        "consider consolidating them into a single authoritative requirement and removing the "
        "duplicate to reduce maintenance burden."
    ),
    ContradictionType.DIRECT_CONFLICT: (
        "Review {req1} and {req2}, which appear to directly conflict. Clarify the intended "
        "system behavior and reconcile the requirements so they are logically consistent."
    ),
    ContradictionType.LOGICAL_CONFLICT: (
        "Review {req1} and {req2} for logical inconsistencies. Examine the actors, actions, "
        "conditions, and constraints in each requirement and clarify how they should interact."
    ),
    ContradictionType.UNKNOWN: (
        "Review {req1} and {req2}, which are semantically related but whose exact relationship "
        "could not be determined with certainty. Clarify whether they address the same concern "
        "or are intentionally distinct."
    ),
}

# Severity-specific preambles
_SEVERITY_PREFIX: Dict[Severity, str] = {
    Severity.HIGH:   "[HIGH PRIORITY] ",
    Severity.MEDIUM: "[MEDIUM PRIORITY] ",
    Severity.LOW:    "[LOW PRIORITY] ",
}


class RecommendationGenerationAgent:
    """
    Generates context-aware, explainable recommendations from verified findings.

    The agent operates in two modes:
    1. Deterministic (default) — uses templates keyed by contradiction type.
    2. LLM-augmented (optional) — uses a provider-independent LangChain
       BaseChatModel for richer language; falls back to deterministic on failure.

    IMPORTANT: This agent never modifies requirements and never chooses a
    "winner" between conflicting requirements.
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: Optional LangChain BaseChatModel. If None the agent uses
                 deterministic templates only.
        """
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        findings: List[Finding],
        dependency_results: Optional[List[DependencyResult]] = None,
    ) -> List[Recommendation]:
        """
        Generate recommendations for a list of verified or potential findings.

        Args:
            findings:           Findings from ContradictionDetectionAgent /
                                DependencyVerificationAgent.
            dependency_results: Optional dependency context indexed by pair.

        Returns:
            List of Recommendation objects.
        """
        dep_map: Dict[tuple, DependencyResult] = {}
        if dependency_results:
            for dr in dependency_results:
                key = tuple(sorted([dr.requirement_id_1, dr.requirement_id_2]))
                dep_map[key] = dr

        recommendations: List[Recommendation] = []
        for finding in findings:
            dep = dep_map.get(
                tuple(sorted([finding.requirement_id_1, finding.requirement_id_2]))
            )
            rec = self._make_recommendation(finding, dep)
            recommendations.append(rec)

        return recommendations

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_recommendation(
        self,
        finding: Finding,
        dep: Optional[DependencyResult],
    ) -> Recommendation:
        """Build a single Recommendation, with optional LLM enhancement."""
        rec_text = None
        used_llm = False

        if self.llm is not None:
            rec_text, used_llm = self._try_llm(finding, dep)

        if rec_text is None:
            rec_text = self._deterministic_text(finding, dep)

        return Recommendation(
            finding_id=finding.finding_id,
            requirement_id_1=finding.requirement_id_1,
            requirement_id_2=finding.requirement_id_2,
            contradiction_type=finding.contradiction_type,
            severity=finding.severity,
            confidence_score=finding.confidence_score,
            explanation=finding.explanation,
            recommendation_text=rec_text,
            evidence=finding.evidence,
            used_llm=used_llm,
        )

    def _deterministic_text(
        self,
        finding: Finding,
        dep: Optional[DependencyResult],
    ) -> str:
        """Generate a deterministic recommendation from templates."""
        template = _TEMPLATES.get(finding.contradiction_type, _TEMPLATES[ContradictionType.UNKNOWN])
        severity_prefix = _SEVERITY_PREFIX.get(finding.severity, "")

        text = template.format(
            req1=finding.requirement_id_1,
            req2=finding.requirement_id_2,
        )

        # Append dependency context when available
        if dep and dep.explanation:
            text += f" Note: {dep.explanation}"

        # Warn when evidence is insufficient
        if finding.confidence_score < 0.30:
            text += (
                " However, the available evidence is limited; further clarification "
                "from stakeholders is strongly recommended before acting on this finding."
            )

        return f"{severity_prefix}{text}"

    def _try_llm(
        self,
        finding: Finding,
        dep: Optional[DependencyResult],
    ):
        """
        Attempt LLM-based recommendation generation.

        Returns:
            (recommendation_text, used_llm): tuple. used_llm=False on failure.
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate

            dep_context = dep.explanation if dep else "No dependency context available."
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are an expert requirements engineering consultant. "
                 "Your task is to generate a concise, actionable recommendation for a "
                 "requirements analyst reviewing a potential inconsistency in an SRS document.\n\n"
                 "RULES:\n"
                 "- Do NOT modify, delete, or rewrite any requirement.\n"
                 "- Do NOT automatically choose which requirement is correct.\n"
                 "- Do NOT invent actors, constraints, or behaviors not present in the evidence.\n"
                 "- If evidence is insufficient, explicitly state that clarification is required.\n"
                 "- Keep the recommendation concise (2–4 sentences).\n\n"
                 "Finding ID: {finding_id}\n"
                 "Contradiction Type: {contradiction_type}\n"
                 "Severity: {severity}\n"
                 "Confidence: {confidence}\n"
                 "Explanation: {explanation}\n"
                 "Evidence:\n{evidence}\n"
                 "Dependency context: {dep_context}\n\n"
                 "Generate a single recommendation paragraph."
                 )
            ])
            messages = prompt.format_messages(
                finding_id=finding.finding_id,
                contradiction_type=finding.contradiction_type.value,
                severity=finding.severity.value,
                confidence=finding.confidence_score,
                explanation=finding.explanation,
                evidence=finding.evidence,
                dep_context=dep_context,
            )
            response = self.llm.invoke(messages)
            text = getattr(response, "content", None)
            # Only accept valid, non-empty string content
            if not isinstance(text, str) or not text.strip():
                return None, False
            if len(text.strip()) > 10:
                return text.strip(), True
        except Exception as exc:
            logger.warning("LLM recommendation generation failed: %s. Using deterministic fallback.", exc)
        return None, False
