"""Tests for the Dependency Verification Agent (Version 5)."""
import pytest
from src.schemas.data_models import (
    Requirement, NumericalValue, CandidatePair, ReasoningResult,
    ContradictionType, Severity, FindingStatus, Finding,
    DependencyStatus
)
from src.agents.contradiction import ContradictionDetectionAgent
from src.agents.dependency import DependencyVerificationAgent
from src.agents.representation import KnowledgeRepresentationAgent
from src.agents.reasoning import LogicalReasoningAgent


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_finding(id1, id2, ctype, severity, confidence, explanation="", evidence="", status=FindingStatus.POTENTIAL) -> Finding:
    return Finding(
        finding_id=f"F-{id1}{id2}",
        requirement_id_1=id1,
        requirement_id_2=id2,
        contradiction_type=ctype,
        severity=severity,
        confidence_score=confidence,
        explanation=explanation or f"Conflict between {id1} and {id2}",
        evidence=evidence or f"[{id1}]: text1\n[{id2}]: text2",
        status=status,
    )


# ---------------------------------------------------------------------------
# Tests: Dependency Status Classification
# ---------------------------------------------------------------------------

def test_supported_dependency_on_high_confidence_permission_conflict():
    finding = make_finding("R1", "R2", ContradictionType.PERMISSION_CONFLICT, Severity.HIGH, 0.75)
    agent = DependencyVerificationAgent()
    updated, dep_results = agent.verify([finding])

    assert dep_results[0].dependency_status == DependencyStatus.SUPPORTED
    assert updated[0].status == FindingStatus.VERIFIED

def test_conditional_dependency_on_conditional_conflict():
    finding = make_finding("R1", "R2", ContradictionType.CONDITIONAL_CONFLICT, Severity.MEDIUM, 0.40)
    agent = DependencyVerificationAgent()
    updated, dep_results = agent.verify([finding])

    assert dep_results[0].dependency_status == DependencyStatus.CONDITIONAL

def test_insufficient_info_on_unknown_type():
    finding = make_finding("R1", "R2", ContradictionType.UNKNOWN, Severity.LOW, 0.15)
    agent = DependencyVerificationAgent()
    updated, dep_results = agent.verify([finding])

    assert dep_results[0].dependency_status == DependencyStatus.INSUFFICIENT_INFORMATION
    # Should stay potential, never auto-verified
    assert updated[0].status == FindingStatus.POTENTIAL

def test_unrelated_dependency_rejects_finding():
    # LOGICAL_CONFLICT with moderate confidence (above 0.25 threshold) but low enough
    # to fall through to the "else → UNRELATED → REJECTED" branch.
    # The SUPPORTED rule needs >= 0.60 *and* specific types (not LOGICAL_CONFLICT).
    # So LOGICAL_CONFLICT at 0.35 (>= 0.25, < 0.60, not conditional/redundancy/unknown)
    # → UNRELATED → penalty brings confidence to 0.00 → REJECTED.
    finding = make_finding("R1", "R2", ContradictionType.LOGICAL_CONFLICT, Severity.LOW, 0.35)
    agent = DependencyVerificationAgent()
    updated, dep_results = agent.verify([finding])

    assert dep_results[0].dependency_status == DependencyStatus.UNRELATED
    assert updated[0].status == FindingStatus.REJECTED

def test_evidence_preserved_in_dependency_result():
    finding = make_finding("R1", "R2", ContradictionType.PERMISSION_CONFLICT, Severity.HIGH, 0.75, evidence="[R1]: EXACT TEXT")
    agent = DependencyVerificationAgent()
    _, dep_results = agent.verify([finding])

    assert "[R1]: EXACT TEXT" in dep_results[0].supporting_evidence

def test_confidence_score_bounded_in_dependency():
    finding = make_finding("R1", "R2", ContradictionType.PERMISSION_CONFLICT, Severity.HIGH, 0.95)
    agent = DependencyVerificationAgent()
    updated, dep_results = agent.verify([finding])

    assert 0.0 <= updated[0].confidence_score <= 1.0
    assert 0.0 <= dep_results[0].confidence_score <= 1.0

def test_false_positive_reduction_conditional():
    """Conditional conflict with moderate confidence should be downgraded to potential or rejected."""
    finding = make_finding("R1", "R2", ContradictionType.CONDITIONAL_CONFLICT, Severity.MEDIUM, 0.30)
    agent = DependencyVerificationAgent()
    updated, _ = agent.verify([finding])

    assert updated[0].status in (FindingStatus.POTENTIAL, FindingStatus.REJECTED)
    # Must not be auto-verified
    assert updated[0].status != FindingStatus.VERIFIED

def test_redundancy_stays_conditional():
    finding = make_finding("R1", "R2", ContradictionType.REDUNDANCY, Severity.LOW, 0.45)
    agent = DependencyVerificationAgent()
    updated, dep_results = agent.verify([finding])

    assert dep_results[0].dependency_status == DependencyStatus.CONDITIONAL

def test_numerical_conflict_verified_when_high_confidence():
    finding = make_finding("R1", "R2", ContradictionType.NUMERICAL_CONFLICT, Severity.HIGH, 0.70)
    agent = DependencyVerificationAgent()
    updated, dep_results = agent.verify([finding])

    assert dep_results[0].dependency_status == DependencyStatus.SUPPORTED
    assert updated[0].status == FindingStatus.VERIFIED

def test_explanation_present_in_dependency_result():
    finding = make_finding("R1", "R2", ContradictionType.PERMISSION_CONFLICT, Severity.HIGH, 0.75)
    agent = DependencyVerificationAgent()
    _, dep_results = agent.verify([finding])

    assert len(dep_results[0].explanation) > 0


# ---------------------------------------------------------------------------
# Integration Test: Full pipeline V2 → V4 → V5
# ---------------------------------------------------------------------------

def test_full_pipeline_integration():
    """
    Integration test:
    V2 Requirement objects
        ↓ V4 Knowledge Representation
        ↓ V4 Logical Reasoning
        ↓ V5 Contradiction Detection
        ↓ V5 Dependency Verification
        ↓ Verified Findings
    """
    # V2 structured requirements
    req1 = Requirement(
        requirement_id="SYS-01",
        original_text="Students shall be allowed to register for university events.",
        actor="Students",
        action="register",
        object="events",
        source_page=1,
    )
    req2 = Requirement(
        requirement_id="SYS-02",
        original_text="Only administrators shall be allowed to register students for university events.",
        actor="administrators",
        action="register",
        object="events",
        source_page=2,
    )

    # V4 Knowledge Representation
    kg_agent = KnowledgeRepresentationAgent()
    reps = kg_agent.transform([req1, req2])
    assert len(reps) == 2

    # Simulate V3 candidate pair
    pair = CandidatePair(
        requirement_id_1="SYS-01",
        requirement_id_2="SYS-02",
        similarity_score=0.91,
        requirement_text_1=req1.original_text,
        requirement_text_2=req2.original_text,
    )

    # V4 Logical Reasoning
    reasoning_agent = LogicalReasoningAgent()
    reasoning_results = reasoning_agent.evaluate_candidates([req1, req2], [pair])
    assert len(reasoning_results) == 1
    assert reasoning_results[0].has_overlapping_actions is True
    assert reasoning_results[0].has_different_actors is True

    # V5 Contradiction Detection
    contradiction_agent = ContradictionDetectionAgent()
    findings = contradiction_agent.detect(
        requirements=[req1, req2],
        candidate_pairs=[pair],
        reasoning_results=reasoning_results,
    )
    assert len(findings) == 1
    assert findings[0].contradiction_type == ContradictionType.PERMISSION_CONFLICT
    assert findings[0].status == FindingStatus.VERIFIED  # high confidence

    # V5 Dependency Verification
    dep_agent = DependencyVerificationAgent()
    verified_findings, dep_results = dep_agent.verify(findings)

    assert len(verified_findings) == 1
    assert len(dep_results) == 1
    assert verified_findings[0].status == FindingStatus.VERIFIED
    assert dep_results[0].dependency_status == DependencyStatus.SUPPORTED
    # Evidence contains original text
    assert "Students shall be allowed" in dep_results[0].supporting_evidence or \
           "administrators" in dep_results[0].supporting_evidence
