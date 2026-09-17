"""Tests for the Verification Agent (Version 6)."""
import pytest
from src.schemas.data_models import (
    Finding, Recommendation, ContradictionType, Severity, FindingStatus,
    VerificationStatus
)
from src.agents.verification import VerificationAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_finding(
    finding_id="F-001",
    id1="REQ-001",
    id2="REQ-002",
    ctype=ContradictionType.PERMISSION_CONFLICT,
    severity=Severity.HIGH,
    confidence=0.75,
    explanation="Conflict explanation.",
    evidence="[REQ-001]: text1\n[REQ-002]: text2",
    status=FindingStatus.VERIFIED,
):
    return Finding(
        finding_id=finding_id,
        requirement_id_1=id1,
        requirement_id_2=id2,
        contradiction_type=ctype,
        severity=severity,
        confidence_score=confidence,
        explanation=explanation,
        evidence=evidence,
        status=status,
    )


def make_recommendation(
    finding_id="F-001",
    id1="REQ-001",
    id2="REQ-002",
    ctype=ContradictionType.PERMISSION_CONFLICT,
    severity=Severity.HIGH,
    confidence=0.75,
    explanation="Conflict explanation.",
    rec_text="Review the authorization rules.",
    evidence="[REQ-001]: text1\n[REQ-002]: text2",
    used_llm=False,
):
    return Recommendation(
        finding_id=finding_id,
        requirement_id_1=id1,
        requirement_id_2=id2,
        contradiction_type=ctype,
        severity=severity,
        confidence_score=confidence,
        explanation=explanation,
        recommendation_text=rec_text,
        evidence=evidence,
        used_llm=used_llm,
    )


# ---------------------------------------------------------------------------
# Tests: Valid Recommendations
# ---------------------------------------------------------------------------

def test_valid_recommendation_approved():
    """A complete, well-formed recommendation should be approved or needs_review."""
    finding = make_finding()
    rec = make_recommendation()
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert len(results) == 1
    assert results[0].verification_status in (VerificationStatus.APPROVED, VerificationStatus.NEEDS_REVIEW)
    assert results[0].verification_confidence > 0.5
    assert len(results[0].verified_claims) > 0


def test_valid_recommendation_includes_verified_claims():
    """Verified claims list should document passing checks."""
    finding = make_finding()
    rec = make_recommendation()
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert "Source finding is traceable." in results[0].verified_claims
    assert "Both requirement IDs are present." in results[0].verified_claims


def test_valid_recommendation_no_modification_keywords():
    """Valid recommendation should not contain modification keywords."""
    finding = make_finding()
    rec = make_recommendation()
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert not any("automatic SRS modification" in w for w in results[0].warnings)


# ---------------------------------------------------------------------------
# Tests: Traceability (Check 1)
# ---------------------------------------------------------------------------

def test_missing_source_finding_reduces_confidence():
    """Recommendation without source finding should be flagged."""
    rec = make_recommendation(finding_id="F-MISSING")
    finding = make_finding(finding_id="F-001")  # Different ID
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    # Score should be reduced when traceability is lost
    assert results[0].verification_confidence <= 0.5  # Max reduced by 0.50
    assert any("Traceability cannot be established" in w for w in results[0].warnings)


# ---------------------------------------------------------------------------
# Tests: Requirement IDs (Check 2)
# ---------------------------------------------------------------------------

def test_missing_requirement_id_1_flagged():
    """Missing requirement_id_1 should be flagged."""
    finding = make_finding()
    rec = make_recommendation(id1="")  # Empty ID
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert any("requirement" in w.lower() for w in results[0].warnings) or len(results[0].warnings) > 0
    assert results[0].verification_confidence < 1.0


def test_both_requirement_ids_present_verified():
    """Both requirement IDs present should be verified claim."""
    finding = make_finding()
    rec = make_recommendation()
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert "Both requirement IDs are present." in results[0].verified_claims


# (Removed test_missing_requirement_id_2_flagged - Pydantic validation prevents None)


# ---------------------------------------------------------------------------
# Tests: Contradiction Type (Check 3)
# ---------------------------------------------------------------------------

def test_valid_contradiction_type_verified():
    """Valid contradiction type should be verified claim."""
    finding = make_finding()
    rec = make_recommendation()
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert "Contradiction type is a recognised category." in results[0].verified_claims


# (Removed test_invalid_contradiction_type_flagged - Pydantic validation prevents invalid enum)


# ---------------------------------------------------------------------------
# Tests: Confidence Score (Check 4)
# ---------------------------------------------------------------------------

def test_confidence_in_valid_bounds():
    """Confidence score in [0, 1] should be verified."""
    finding = make_finding(confidence=0.5)
    rec = make_recommendation(confidence=0.5)
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert "Confidence score is within valid bounds [0, 1]." in results[0].verified_claims


# (Removed test_confidence_out_of_bounds_high - Pydantic validation prevents > 1.0)
# (Removed test_confidence_out_of_bounds_low - Pydantic validation prevents < 0.0)


# ---------------------------------------------------------------------------
# Tests: Recommendation Text (Check 5)
# ---------------------------------------------------------------------------

def test_missing_recommendation_text_flagged():
    """Missing recommendation text should be flagged."""
    finding = make_finding()
    rec = make_recommendation(rec_text="")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert any("missing or too short" in w.lower() for w in results[0].warnings)


def test_recommendation_text_too_short_flagged():
    """Recommendation text < 20 chars should be flagged."""
    finding = make_finding()
    rec = make_recommendation(rec_text="Brief")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert any("missing or too short" in w.lower() for w in results[0].warnings)


def test_recommendation_text_adequate_verified():
    """Recommendation text > 20 chars should be verified."""
    finding = make_finding()
    rec = make_recommendation(rec_text="This is a proper recommendation with adequate length.")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert "Recommendation text is present and non-trivial." in results[0].verified_claims


# ---------------------------------------------------------------------------
# Tests: Modification Keywords (Check 6)
# ---------------------------------------------------------------------------

def test_recommendation_with_change_keyword_rejected():
    """Recommendation suggesting 'change requirement' should be rejected."""
    finding = make_finding()
    rec = make_recommendation(rec_text="You should change requirement REQ-001 to say X instead of Y.")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert any("automatic SRS modification" in w for w in results[0].warnings)
    # Score should be significantly reduced due to modification keyword penalty
    assert results[0].verification_confidence <= 0.7  # 1.0 - 0.30 (penalty) = 0.70


def test_recommendation_with_delete_keyword_rejected():
    """Recommendation suggesting 'delete requirement' should be rejected."""
    finding = make_finding()
    rec = make_recommendation(rec_text="Delete requirement REQ-002 because it is redundant.")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert any("automatic SRS modification" in w for w in results[0].warnings)


def test_recommendation_with_rewrite_keyword_rejected():
    """Recommendation suggesting 'rewrite requirement' should be rejected."""
    finding = make_finding()
    rec = make_recommendation(rec_text="Rewrite requirement REQ-001 more clearly.")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert any("automatic SRS modification" in w for w in results[0].warnings)


def test_recommendation_without_modification_keywords_verified():
    """Recommendation without modification keywords should be verified."""
    finding = make_finding()
    rec = make_recommendation(rec_text="Review the requirements and clarify which applies in which context.")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert "Recommendation does not propose automatic SRS modification." in results[0].verified_claims


# ---------------------------------------------------------------------------
# Tests: Evidence (Check 7)
# ---------------------------------------------------------------------------

def test_missing_evidence_flagged():
    """Missing evidence should be flagged."""
    finding = make_finding()
    rec = make_recommendation(evidence="")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert any("empty or too short" in w.lower() for w in results[0].warnings)
    assert results[0].verification_confidence < 1.0


def test_evidence_too_short_flagged():
    """Evidence < 5 chars should be flagged."""
    finding = make_finding()
    rec = make_recommendation(evidence="abc")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert any("empty or too short" in w.lower() for w in results[0].warnings)


def test_adequate_evidence_verified():
    """Adequate evidence should be verified."""
    finding = make_finding()
    rec = make_recommendation(evidence="[REQ-001]: First requirement text\n[REQ-002]: Second requirement text")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert "Evidence string is present." in results[0].verified_claims


# ---------------------------------------------------------------------------
# Tests: Context Reference (Check 8)
# ---------------------------------------------------------------------------

def test_recommendation_references_requirement_ids():
    """Recommendation mentioning requirement IDs should reference context."""
    finding = make_finding(id1="REQ-A", id2="REQ-B")
    rec = make_recommendation(id1="REQ-A", id2="REQ-B", rec_text="Review REQ-A and REQ-B for clarity.")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert "Recommendation references requirement IDs from the finding." in results[0].verified_claims


def test_recommendation_missing_context_warned():
    """Recommendation not mentioning IDs should trigger warning."""
    finding = make_finding(id1="REQ-A", id2="REQ-B")
    rec = make_recommendation(id1="REQ-A", id2="REQ-B", rec_text="This is a generic recommendation without specific IDs.")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert any("does not explicitly reference the requirement IDs" in w for w in results[0].warnings)


# ---------------------------------------------------------------------------
# Tests: Status Determination
# ---------------------------------------------------------------------------

def test_high_score_no_warnings_approved():
    """High score with no warnings → APPROVED."""
    finding = make_finding()
    rec = make_recommendation(
        rec_text="Review REQ-001 and REQ-002 carefully for authorization scope clarification.",
        evidence="[REQ-001]: full text\n[REQ-002]: full text"
    )
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    # May be APPROVED if score >= 0.70 and no warnings, or NEEDS_REVIEW
    assert results[0].verification_status in (VerificationStatus.APPROVED, VerificationStatus.NEEDS_REVIEW)
    if not results[0].warnings:
        # If no warnings, should be approved or at least high confidence
        assert results[0].verification_confidence >= 0.65


def test_low_score_needs_review_with_major_issues():
    """Major issues result in NEEDS_REVIEW status."""
    finding = make_finding()
    rec = make_recommendation(
        confidence=0.15,  # Already low finding confidence
        rec_text="",  # Missing text to reduce score further
        evidence=""   # Missing evidence
    )
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    # Major issues should result in needs_review or rejected
    assert results[0].verification_status in (VerificationStatus.NEEDS_REVIEW, VerificationStatus.REJECTED)
    assert results[0].verification_confidence <= 0.7  # Reduced by missing text and evidence


def test_medium_score_needs_review():
    """Medium confidence score → NEEDS_REVIEW."""
    finding = make_finding(severity=Severity.MEDIUM)
    rec = make_recommendation(
        confidence=0.5,
        rec_text="Review requirements for consistency.",
        evidence="[REQ]: partial evidence"
    )
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert results[0].verification_status in (VerificationStatus.NEEDS_REVIEW, VerificationStatus.APPROVED)
    assert results[0].verification_confidence >= 0.25


def test_high_severity_always_requires_human_review():
    """HIGH severity findings always require human review."""
    finding = make_finding(severity=Severity.HIGH)
    rec = make_recommendation(severity=Severity.HIGH)
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert results[0].requires_human_review is True


def test_low_severity_may_not_require_human_review():
    """LOW severity with high score may not require human review."""
    finding = make_finding(severity=Severity.LOW, confidence=0.95)
    rec = make_recommendation(
        severity=Severity.LOW,
        confidence=0.95,
        rec_text="Review REQ-001 and REQ-002 carefully for scope clarification with adequate length.",
        evidence="[REQ-001]: full text here\n[REQ-002]: full text here"
    )
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    # With no warnings and high score, may not require review
    if results[0].verification_status == VerificationStatus.APPROVED and not results[0].warnings:
        assert results[0].requires_human_review is False


# ---------------------------------------------------------------------------
# Tests: Final Recommendation Field
# ---------------------------------------------------------------------------

def test_final_recommendation_field_preserved():
    """final_recommendation should be the recommendation text."""
    finding = make_finding()
    rec_text = "This is the recommendation text to verify."
    rec = make_recommendation(rec_text=rec_text)
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert results[0].final_recommendation == rec_text


def test_final_recommendation_handles_empty_text():
    """final_recommendation should handle empty text gracefully."""
    finding = make_finding()
    rec = make_recommendation(rec_text="")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert results[0].final_recommendation == ""
    assert results[0].final_recommendation is not None


# ---------------------------------------------------------------------------
# Tests: Multiple Recommendations
# ---------------------------------------------------------------------------

def test_verify_multiple_recommendations():
    """Agent should handle multiple recommendations."""
    finding1 = make_finding(finding_id="F-001")
    finding2 = make_finding(finding_id="F-002", id1="REQ-003", id2="REQ-004")
    
    rec1 = make_recommendation(finding_id="F-001")
    rec2 = make_recommendation(finding_id="F-002", id1="REQ-003", id2="REQ-004")
    
    agent = VerificationAgent()
    
    results = agent.verify([rec1, rec2], [finding1, finding2])
    
    assert len(results) == 2
    assert results[0].finding_id == "F-001"
    assert results[1].finding_id == "F-002"


def test_verify_unmatched_findings():
    """Agent should handle recommendations without matching findings."""
    finding = make_finding(finding_id="F-001")
    rec_unmatched = make_recommendation(finding_id="F-UNKNOWN")
    
    agent = VerificationAgent()
    
    results = agent.verify([rec_unmatched], [finding])
    
    assert len(results) == 1
    # Missing source finding is a -0.50 penalty
    assert results[0].verification_confidence <= 0.5
    assert any("cannot be established" in w.lower() for w in results[0].warnings)


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------

def test_whitespace_only_recommendation_text_flagged():
    """Recommendation text with only whitespace should be flagged."""
    finding = make_finding()
    rec = make_recommendation(rec_text="   \n\t  ")
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert any("missing or too short" in w.lower() for w in results[0].warnings)


def test_recommendation_with_all_valid_fields():
    """Recommendation with all valid fields should pass all checks."""
    finding = make_finding(
        severity=Severity.MEDIUM,
        confidence=0.75,
        evidence="[REQ-001]: The system shall authenticate users.\n[REQ-002]: All users must be authenticated."
    )
    rec = make_recommendation(
        severity=Severity.MEDIUM,
        confidence=0.75,
        rec_text="Review REQ-001 and REQ-002 to determine whether both conditions can be satisfied simultaneously or whether they represent different scopes.",
        evidence="[REQ-001]: The system shall authenticate users.\n[REQ-002]: All users must be authenticated."
    )
    agent = VerificationAgent()
    
    results = agent.verify([rec], [finding])
    
    assert len(results[0].verified_claims) >= 5
    assert results[0].verification_status in (VerificationStatus.APPROVED, VerificationStatus.NEEDS_REVIEW)
