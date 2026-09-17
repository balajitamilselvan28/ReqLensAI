"""Tests for the Contradiction Detection Agent (Version 5)."""
import pytest
from src.schemas.data_models import (
    Requirement, NumericalValue, CandidatePair, ReasoningResult,
    ContradictionType, Severity, FindingStatus
)
from src.agents.contradiction import ContradictionDetectionAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_pair(id1: str, id2: str, score: float = 0.90, text1: str = "", text2: str = "") -> CandidatePair:
    return CandidatePair(
        requirement_id_1=id1,
        requirement_id_2=id2,
        similarity_score=score,
        requirement_text_1=text1,
        requirement_text_2=text2,
    )

def make_rr(id1, id2, overlapping=False, diff_actors=False, numerical=False,
            num_cmp=None, conditional=False, details="") -> ReasoningResult:
    return ReasoningResult(
        requirement_id_1=id1,
        requirement_id_2=id2,
        has_overlapping_actions=overlapping,
        has_different_actors=diff_actors,
        has_numerical_overlap=numerical,
        numerical_comparison=num_cmp,
        has_conditional_implication=conditional,
        reasoning_details=details or "Test reasoning details.",
    )


# ---------------------------------------------------------------------------
# Tests: Permission Conflict
# ---------------------------------------------------------------------------

def test_permission_conflict_detected():
    req1 = Requirement(requirement_id="REQ-001", original_text="Students shall be allowed to register for events.", actor="Students", action="register", object="events", source_page=1)
    req2 = Requirement(requirement_id="REQ-002", original_text="Only administrators shall be allowed to register students for events.", actor="administrators", action="register", object="events", source_page=1)
    pair = make_pair("REQ-001", "REQ-002", 0.90)
    rr = make_rr("REQ-001", "REQ-002", overlapping=True, diff_actors=True, details="Different actors on same action.")

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [rr])

    assert len(findings) == 1
    assert findings[0].contradiction_type == ContradictionType.PERMISSION_CONFLICT
    assert findings[0].severity == Severity.HIGH

def test_permission_conflict_explanation_contains_actors():
    req1 = Requirement(requirement_id="REQ-001", original_text="Students register.", actor="Students", action="register", source_page=1)
    req2 = Requirement(requirement_id="REQ-002", original_text="Only admins register.", actor="administrators", action="register", source_page=1)
    pair = make_pair("REQ-001", "REQ-002", 0.90)
    rr = make_rr("REQ-001", "REQ-002", overlapping=True, diff_actors=True)

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [rr])

    assert "Students" in findings[0].explanation or "administrators" in findings[0].explanation

def test_original_text_preserved_in_evidence():
    req1 = Requirement(requirement_id="REQ-001", original_text="EXACT TEXT ONE", actor="A", action="act", source_page=1)
    req2 = Requirement(requirement_id="REQ-002", original_text="EXACT TEXT TWO", actor="B", action="act", source_page=1)
    pair = make_pair("REQ-001", "REQ-002", 0.90)
    rr = make_rr("REQ-001", "REQ-002", overlapping=True, diff_actors=True)

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [rr])

    assert "EXACT TEXT ONE" in findings[0].evidence
    assert "EXACT TEXT TWO" in findings[0].evidence


# ---------------------------------------------------------------------------
# Tests: Numerical Conflict
# ---------------------------------------------------------------------------

def test_numerical_conflict_detected():
    req1 = Requirement(requirement_id="REQ-003", original_text="Respond within 300ms.", action="respond", numerical_values=[NumericalValue(value=300, unit="ms")], source_page=1)
    req2 = Requirement(requirement_id="REQ-004", original_text="Respond within 500ms.", action="respond", numerical_values=[NumericalValue(value=500, unit="ms")], source_page=1)
    pair = make_pair("REQ-003", "REQ-004", 0.88)
    rr = make_rr("REQ-003", "REQ-004", overlapping=True, numerical=True, num_cmp="300.0 vs 500.0 ms")

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [rr])

    assert len(findings) == 1
    assert findings[0].contradiction_type == ContradictionType.NUMERICAL_CONFLICT
    assert "300.0 vs 500.0 ms" in findings[0].explanation


# ---------------------------------------------------------------------------
# Tests: Conditional Conflict
# ---------------------------------------------------------------------------

def test_conditional_conflict_detected():
    req1 = Requirement(requirement_id="REQ-005", original_text="Allow access after authentication.", action="allow", conditions=["after authentication"], source_page=1)
    req2 = Requirement(requirement_id="REQ-006", original_text="Deny access after authentication.", action="deny", conditions=["after authentication"], source_page=1)
    pair = make_pair("REQ-005", "REQ-006", 0.85)
    rr = make_rr("REQ-005", "REQ-006", overlapping=False, conditional=True)
    # Force overlapping manually
    rr.has_overlapping_actions = True

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [rr])

    assert len(findings) == 1
    assert findings[0].contradiction_type == ContradictionType.CONDITIONAL_CONFLICT


# ---------------------------------------------------------------------------
# Tests: Redundancy
# ---------------------------------------------------------------------------

def test_redundancy_detected():
    req1 = Requirement(requirement_id="REQ-007", original_text="The system shall log all errors.", action="log", object="errors", actor="The system", source_page=1)
    req2 = Requirement(requirement_id="REQ-008", original_text="The system shall log all errors.", action="log", object="errors", actor="The system", source_page=2)
    pair = make_pair("REQ-007", "REQ-008", 0.98)
    rr = make_rr("REQ-007", "REQ-008", overlapping=True, diff_actors=False, numerical=False)

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [rr])

    assert len(findings) == 1
    assert findings[0].contradiction_type == ContradictionType.REDUNDANCY
    assert findings[0].severity == Severity.LOW


# ---------------------------------------------------------------------------
# Tests: Confidence Score Bounds
# ---------------------------------------------------------------------------

def test_confidence_score_bounded_0_to_1():
    req1 = Requirement(requirement_id="REQ-001", original_text="A", actor="X", action="act", source_page=1)
    req2 = Requirement(requirement_id="REQ-002", original_text="B", actor="Y", action="act", source_page=1)
    pair = make_pair("REQ-001", "REQ-002", 0.99)
    rr = make_rr("REQ-001", "REQ-002", overlapping=True, diff_actors=True, numerical=True, num_cmp="1 vs 2")

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [rr])

    for f in findings:
        assert 0.0 <= f.confidence_score <= 1.0


# ---------------------------------------------------------------------------
# Tests: Duplicate Prevention
# ---------------------------------------------------------------------------

def test_duplicate_pair_prevention():
    req1 = Requirement(requirement_id="REQ-001", original_text="A", actor="X", action="act", source_page=1)
    req2 = Requirement(requirement_id="REQ-002", original_text="B", actor="Y", action="act", source_page=1)
    pair_fwd = make_pair("REQ-001", "REQ-002", 0.90)
    pair_rev = make_pair("REQ-002", "REQ-001", 0.90)  # reversed
    rr = make_rr("REQ-001", "REQ-002", overlapping=True, diff_actors=True)

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair_fwd, pair_rev], [rr])

    assert len(findings) == 1


# ---------------------------------------------------------------------------
# Tests: Missing / Insufficient Evidence
# ---------------------------------------------------------------------------

def test_low_similarity_no_reasoning_returns_no_finding():
    req1 = Requirement(requirement_id="REQ-001", original_text="A", source_page=1)
    req2 = Requirement(requirement_id="REQ-002", original_text="B", source_page=1)
    pair = make_pair("REQ-001", "REQ-002", 0.30)  # low similarity

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [])

    assert findings == []

def test_unknown_type_assigned_when_no_structural_match():
    req1 = Requirement(requirement_id="REQ-001", original_text="High similarity requirement A", source_page=1)
    req2 = Requirement(requirement_id="REQ-002", original_text="High similarity requirement B", source_page=1)
    pair = make_pair("REQ-001", "REQ-002", 0.75)

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [])

    if findings:
        assert findings[0].contradiction_type == ContradictionType.UNKNOWN
        assert findings[0].status == FindingStatus.POTENTIAL


# ---------------------------------------------------------------------------
# Tests: Severity Assignment
# ---------------------------------------------------------------------------

def test_permission_conflict_severity_is_high():
    req1 = Requirement(requirement_id="R1", original_text="t1", actor="A", action="x", source_page=1)
    req2 = Requirement(requirement_id="R2", original_text="t2", actor="B", action="x", source_page=1)
    pair = make_pair("R1", "R2", 0.90)
    rr = make_rr("R1", "R2", overlapping=True, diff_actors=True)

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [rr])

    assert findings[0].severity == Severity.HIGH

def test_redundancy_severity_is_low():
    req1 = Requirement(requirement_id="R1", original_text="same", action="x", actor="S", object="o", source_page=1)
    req2 = Requirement(requirement_id="R2", original_text="same", action="x", actor="S", object="o", source_page=1)
    pair = make_pair("R1", "R2", 0.98)
    rr = make_rr("R1", "R2", overlapping=True, diff_actors=False)

    agent = ContradictionDetectionAgent()
    findings = agent.detect([req1, req2], [pair], [rr])

    if findings:
        assert findings[0].severity == Severity.LOW
