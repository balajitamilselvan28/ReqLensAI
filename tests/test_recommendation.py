"""Tests for the Recommendation Generation Agent (Version 6)."""
import pytest
from unittest.mock import MagicMock

from src.schemas.data_models import (
    Finding, ContradictionType, Severity, FindingStatus,
    DependencyResult, DependencyStatus
)
from src.agents.recommendation import RecommendationGenerationAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_finding(
    id1="REQ-001", id2="REQ-002",
    ctype=ContradictionType.PERMISSION_CONFLICT,
    severity=Severity.HIGH,
    confidence=0.75,
    explanation="Conflict explanation.",
    evidence="[REQ-001]: text1\n[REQ-002]: text2",
    status=FindingStatus.VERIFIED,
):
    return Finding(
        finding_id="F-ABCD1234",
        requirement_id_1=id1,
        requirement_id_2=id2,
        contradiction_type=ctype,
        severity=severity,
        confidence_score=confidence,
        explanation=explanation,
        evidence=evidence,
        status=status,
    )

def make_dep(id1="REQ-001", id2="REQ-002", status=DependencyStatus.SUPPORTED, explanation="Supported."):
    return DependencyResult(
        requirement_id_1=id1,
        requirement_id_2=id2,
        dependency_status=status,
        explanation=explanation,
        supporting_evidence="[REQ-001]: text1",
        confidence_score=0.80,
    )


# ---------------------------------------------------------------------------
# Tests: Deterministic recommendations per contradiction type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctype", [
    ContradictionType.PERMISSION_CONFLICT,
    ContradictionType.NUMERICAL_CONFLICT,
    ContradictionType.CONDITIONAL_CONFLICT,
    ContradictionType.REDUNDANCY,
    ContradictionType.DIRECT_CONFLICT,
    ContradictionType.LOGICAL_CONFLICT,
    ContradictionType.UNKNOWN,
])
def test_deterministic_recommendation_generated_for_all_types(ctype):
    finding = make_finding(ctype=ctype)
    agent = RecommendationGenerationAgent()
    recs = agent.generate([finding])

    assert len(recs) == 1
    assert len(recs[0].recommendation_text.strip()) > 20

def test_permission_conflict_recommendation_content():
    finding = make_finding(ctype=ContradictionType.PERMISSION_CONFLICT)
    agent = RecommendationGenerationAgent()
    recs = agent.generate([finding])

    text = recs[0].recommendation_text
    assert "authorization" in text.lower() or "role" in text.lower() or "permitted" in text.lower()

def test_numerical_conflict_recommendation_content():
    finding = make_finding(ctype=ContradictionType.NUMERICAL_CONFLICT)
    agent = RecommendationGenerationAgent()
    recs = agent.generate([finding])

    text = recs[0].recommendation_text
    assert "numerical" in text.lower() or "constraint" in text.lower() or "limit" in text.lower()

def test_redundancy_recommendation_mentions_consolidation():
    finding = make_finding(ctype=ContradictionType.REDUNDANCY, severity=Severity.LOW)
    agent = RecommendationGenerationAgent()
    recs = agent.generate([finding])

    text = recs[0].recommendation_text
    assert "consolidat" in text.lower() or "duplicate" in text.lower() or "redundant" in text.lower()

def test_unknown_finding_recommendation_requests_clarification():
    finding = make_finding(ctype=ContradictionType.UNKNOWN, confidence=0.12)
    agent = RecommendationGenerationAgent()
    recs = agent.generate([finding])

    text = recs[0].recommendation_text
    assert "clarif" in text.lower() or "evidence" in text.lower() or "stakeholder" in text.lower()


# ---------------------------------------------------------------------------
# Tests: Context awareness and evidence
# ---------------------------------------------------------------------------

def test_context_aware_uses_dependency_explanation():
    finding = make_finding()
    dep = make_dep(explanation="One requirement may apply only under specific conditions.")
    agent = RecommendationGenerationAgent()
    recs = agent.generate([finding], dependency_results=[dep])

    assert "specific conditions" in recs[0].recommendation_text

def test_original_evidence_preserved():
    finding = make_finding(evidence="[REQ-001]: EXACT ORIGINAL TEXT")
    agent = RecommendationGenerationAgent()
    recs = agent.generate([finding])

    assert recs[0].evidence == "[REQ-001]: EXACT ORIGINAL TEXT"

def test_recommendation_does_not_modify_requirements():
    finding = make_finding()
    agent = RecommendationGenerationAgent()
    recs = agent.generate([finding])

    text = recs[0].recommendation_text.lower()
    forbidden = ["rewrite requirement", "delete requirement", "remove requirement", "automatically update"]
    for kw in forbidden:
        assert kw not in text, f"Recommendation must not contain '{kw}'"

def test_severity_prefix_present_in_recommendation():
    finding = make_finding(severity=Severity.HIGH)
    agent = RecommendationGenerationAgent()
    recs = agent.generate([finding])

    assert "[HIGH PRIORITY]" in recs[0].recommendation_text

def test_low_confidence_adds_clarification_note():
    finding = make_finding(confidence=0.15, ctype=ContradictionType.UNKNOWN)
    agent = RecommendationGenerationAgent()
    recs = agent.generate([finding])

    assert "clarification" in recs[0].recommendation_text.lower() or "evidence is limited" in recs[0].recommendation_text.lower()


# ---------------------------------------------------------------------------
# Tests: LLM fallback
# ---------------------------------------------------------------------------

def test_llm_failure_falls_back_to_deterministic():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("API Timeout")

    finding = make_finding()
    agent = RecommendationGenerationAgent(llm=mock_llm)
    recs = agent.generate([finding])

    # Should still produce a recommendation
    assert len(recs) == 1
    assert recs[0].used_llm is False
    assert len(recs[0].recommendation_text) > 20

def test_malformed_llm_output_falls_back_to_deterministic():
    mock_llm = MagicMock()
    # Return an object with no usable content
    response = MagicMock()
    response.content = ""
    mock_llm.invoke.return_value = response

    finding = make_finding()
    agent = RecommendationGenerationAgent(llm=mock_llm)
    recs = agent.generate([finding])

    assert recs[0].used_llm is False
    assert len(recs[0].recommendation_text) > 20

def test_llm_success_sets_used_llm_true():
    mock_llm = MagicMock()
    response = MagicMock()
    response.content = "The analyst should review the authorization policy for event registration carefully."
    mock_llm.invoke.return_value = response

    finding = make_finding()
    agent = RecommendationGenerationAgent(llm=mock_llm)
    recs = agent.generate([finding])

    assert recs[0].used_llm is True
    assert "authorization policy" in recs[0].recommendation_text
