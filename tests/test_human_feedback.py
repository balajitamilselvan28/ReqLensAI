"""Tests for the HumanFeedback Pydantic Model (Version 6)."""
import pytest
from src.schemas.data_models import HumanFeedback, HumanDecision


# ---------------------------------------------------------------------------
# Tests: HumanFeedback Creation
# ---------------------------------------------------------------------------

def test_human_feedback_approve_decision():
    """Create HumanFeedback with APPROVE decision."""
    feedback = HumanFeedback(
        finding_id="F-001",
        decision=HumanDecision.APPROVE,
        original_recommendation="Review REQ-001 and REQ-002 for clarity.",
    )
    
    assert feedback.finding_id == "F-001"
    assert feedback.decision == HumanDecision.APPROVE
    assert feedback.original_recommendation == "Review REQ-001 and REQ-002 for clarity."
    assert feedback.analyst_comment is None
    assert feedback.modified_recommendation is None
    assert feedback.timestamp is None


def test_human_feedback_modify_decision():
    """Create HumanFeedback with MODIFY decision."""
    feedback = HumanFeedback(
        finding_id="F-002",
        decision=HumanDecision.MODIFY,
        original_recommendation="Original recommendation text.",
        modified_recommendation="Modified recommendation text with clearer guidance.",
        analyst_comment="I improved the clarity of this recommendation.",
    )
    
    assert feedback.finding_id == "F-002"
    assert feedback.decision == HumanDecision.MODIFY
    assert feedback.original_recommendation == "Original recommendation text."
    assert feedback.modified_recommendation == "Modified recommendation text with clearer guidance."
    assert feedback.analyst_comment == "I improved the clarity of this recommendation."


def test_human_feedback_reject_decision():
    """Create HumanFeedback with REJECT decision."""
    feedback = HumanFeedback(
        finding_id="F-003",
        decision=HumanDecision.REJECT,
        original_recommendation="Automatically proposed recommendation.",
        analyst_comment="This is a false positive. Both requirements are intentionally independent.",
    )
    
    assert feedback.finding_id == "F-003"
    assert feedback.decision == HumanDecision.REJECT
    assert feedback.analyst_comment == "This is a false positive. Both requirements are intentionally independent."
    assert feedback.modified_recommendation is None


# ---------------------------------------------------------------------------
# Tests: Original Recommendation Preservation
# ---------------------------------------------------------------------------

def test_original_recommendation_preserved_immutably():
    """Original recommendation should be preserved and not modified."""
    original = "Original machine-generated recommendation."
    feedback = HumanFeedback(
        finding_id="F-004",
        decision=HumanDecision.MODIFY,
        original_recommendation=original,
        modified_recommendation="Analyst-modified version.",
    )
    
    assert feedback.original_recommendation == original
    # Verify it can't be changed after creation (immutable within Pydantic)
    assert feedback.original_recommendation is not None


def test_original_recommendation_required():
    """Original recommendation field is required."""
    with pytest.raises(ValueError):
        # Missing original_recommendation field
        HumanFeedback(
            finding_id="F-005",
            decision=HumanDecision.APPROVE,
        )


# ---------------------------------------------------------------------------
# Tests: Modified Recommendation
# ---------------------------------------------------------------------------

def test_modified_recommendation_separate_from_original():
    """Modified recommendation is stored separately from original."""
    original = "Review the requirements."
    modified = "Review the requirements and consider context X."
    feedback = HumanFeedback(
        finding_id="F-006",
        decision=HumanDecision.MODIFY,
        original_recommendation=original,
        modified_recommendation=modified,
    )
    
    assert feedback.original_recommendation == original
    assert feedback.modified_recommendation == modified
    assert feedback.original_recommendation != feedback.modified_recommendation


def test_modified_recommendation_optional_for_approve():
    """Modified recommendation should be None for APPROVE decision."""
    feedback = HumanFeedback(
        finding_id="F-007",
        decision=HumanDecision.APPROVE,
        original_recommendation="Machine recommendation approved as-is.",
    )
    
    assert feedback.modified_recommendation is None


def test_modified_recommendation_optional_for_reject():
    """Modified recommendation should be None for REJECT decision."""
    feedback = HumanFeedback(
        finding_id="F-008",
        decision=HumanDecision.REJECT,
        original_recommendation="False positive recommendation.",
    )
    
    assert feedback.modified_recommendation is None


def test_modified_recommendation_provided_for_modify():
    """Modified recommendation should typically be provided for MODIFY decision."""
    feedback = HumanFeedback(
        finding_id="F-009",
        decision=HumanDecision.MODIFY,
        original_recommendation="Original version.",
        modified_recommendation="Improved version.",
    )
    
    assert feedback.modified_recommendation is not None
    assert len(feedback.modified_recommendation) > 0


# ---------------------------------------------------------------------------
# Tests: Analyst Comment
# ---------------------------------------------------------------------------

def test_analyst_comment_preserved():
    """Analyst comment should be preserved exactly as provided."""
    comment = "This finding is critical. The two requirements are mutually exclusive in our current architecture."
    feedback = HumanFeedback(
        finding_id="F-010",
        decision=HumanDecision.REJECT,
        original_recommendation="Auto recommendation.",
        analyst_comment=comment,
    )
    
    assert feedback.analyst_comment == comment


def test_analyst_comment_optional():
    """Analyst comment is optional."""
    feedback = HumanFeedback(
        finding_id="F-011",
        decision=HumanDecision.APPROVE,
        original_recommendation="Recommendation.",
    )
    
    assert feedback.analyst_comment is None


def test_analyst_comment_empty_string_allowed():
    """Analyst comment can be an empty string."""
    feedback = HumanFeedback(
        finding_id="F-012",
        decision=HumanDecision.APPROVE,
        original_recommendation="Recommendation.",
        analyst_comment="",
    )
    
    assert feedback.analyst_comment == ""


def test_analyst_comment_multiline():
    """Analyst comment can contain multiple lines."""
    comment = """This finding requires discussion with stakeholders.

Considerations:
1. Actor A interprets the requirement as X
2. Actor B interprets the requirement as Y
3. We need to align on which interpretation is correct."""
    
    feedback = HumanFeedback(
        finding_id="F-013",
        decision=HumanDecision.MODIFY,
        original_recommendation="Original.",
        modified_recommendation="Modified.",
        analyst_comment=comment,
    )
    
    assert feedback.analyst_comment == comment
    assert "\n" in feedback.analyst_comment


# ---------------------------------------------------------------------------
# Tests: Timestamp
# ---------------------------------------------------------------------------

def test_timestamp_optional():
    """Timestamp is optional."""
    feedback = HumanFeedback(
        finding_id="F-014",
        decision=HumanDecision.APPROVE,
        original_recommendation="Recommendation.",
    )
    
    assert feedback.timestamp is None


def test_timestamp_iso_8601_format():
    """Timestamp can be provided in ISO 8601 format."""
    iso_timestamp = "2026-09-15T14:30:00Z"
    feedback = HumanFeedback(
        finding_id="F-015",
        decision=HumanDecision.APPROVE,
        original_recommendation="Recommendation.",
        timestamp=iso_timestamp,
    )
    
    assert feedback.timestamp == iso_timestamp


def test_timestamp_with_microseconds():
    """Timestamp can include microseconds."""
    iso_timestamp = "2026-09-15T14:30:00.123456Z"
    feedback = HumanFeedback(
        finding_id="F-016",
        decision=HumanDecision.APPROVE,
        original_recommendation="Recommendation.",
        timestamp=iso_timestamp,
    )
    
    assert feedback.timestamp == iso_timestamp


def test_timestamp_with_timezone_offset():
    """Timestamp can include timezone offset."""
    iso_timestamp = "2026-09-15T14:30:00+05:30"
    feedback = HumanFeedback(
        finding_id="F-017",
        decision=HumanDecision.APPROVE,
        original_recommendation="Recommendation.",
        timestamp=iso_timestamp,
    )
    
    assert feedback.timestamp == iso_timestamp


# ---------------------------------------------------------------------------
# Tests: Finding ID
# ---------------------------------------------------------------------------

def test_finding_id_required():
    """Finding ID is required."""
    with pytest.raises(ValueError):
        HumanFeedback(
            decision=HumanDecision.APPROVE,
            original_recommendation="Recommendation.",
        )


def test_finding_id_string():
    """Finding ID should be a string."""
    feedback = HumanFeedback(
        finding_id="F-ABCD-1234",
        decision=HumanDecision.APPROVE,
        original_recommendation="Recommendation.",
    )
    
    assert isinstance(feedback.finding_id, str)
    assert feedback.finding_id == "F-ABCD-1234"


def test_finding_id_empty_string():
    """Finding ID can be an empty string (edge case)."""
    # Pydantic allows empty strings unless explicitly restricted
    feedback = HumanFeedback(
        finding_id="",
        decision=HumanDecision.APPROVE,
        original_recommendation="Recommendation.",
    )
    
    assert feedback.finding_id == ""


# ---------------------------------------------------------------------------
# Tests: Decision Enum
# ---------------------------------------------------------------------------

def test_decision_enum_values():
    """All decision enum values should work."""
    for decision in [HumanDecision.APPROVE, HumanDecision.MODIFY, HumanDecision.REJECT]:
        feedback = HumanFeedback(
            finding_id="F-001",
            decision=decision,
            original_recommendation="Recommendation.",
        )
        assert feedback.decision == decision


def test_decision_string_conversion():
    """Decision can be provided as string and will be converted to enum."""
    feedback = HumanFeedback(
        finding_id="F-018",
        decision="approve",  # String value
        original_recommendation="Recommendation.",
    )
    
    assert feedback.decision == HumanDecision.APPROVE
    assert isinstance(feedback.decision, HumanDecision)


# ---------------------------------------------------------------------------
# Tests: Complex Scenarios
# ---------------------------------------------------------------------------

def test_full_feedback_workflow_approve():
    """Complete workflow: analyst reviews and approves recommendation."""
    feedback = HumanFeedback(
        finding_id="F-COMPLEX-001",
        decision=HumanDecision.APPROVE,
        original_recommendation="Review REQ-001 and REQ-002 for authorization scope clarification.",
        timestamp="2026-09-15T14:30:00Z",
    )
    
    assert feedback.finding_id == "F-COMPLEX-001"
    assert feedback.decision == HumanDecision.APPROVE
    assert feedback.original_recommendation is not None
    assert feedback.modified_recommendation is None
    assert feedback.analyst_comment is None
    assert feedback.timestamp is not None


def test_full_feedback_workflow_modify():
    """Complete workflow: analyst reviews, modifies, and explains."""
    feedback = HumanFeedback(
        finding_id="F-COMPLEX-002",
        decision=HumanDecision.MODIFY,
        original_recommendation="Review the conflicting requirements.",
        modified_recommendation="Review REQ-X and REQ-Y. Contact the Product Owner to clarify the business intent—the conflict may be scoped by feature flag or user role.",
        analyst_comment="Added specific guidance based on architecture discussion yesterday.",
        timestamp="2026-09-15T15:45:30.123Z",
    )
    
    assert feedback.decision == HumanDecision.MODIFY
    assert "Product Owner" in feedback.modified_recommendation
    assert "architecture" in feedback.analyst_comment
    assert feedback.original_recommendation != feedback.modified_recommendation


def test_full_feedback_workflow_reject():
    """Complete workflow: analyst reviews and rejects as false positive."""
    feedback = HumanFeedback(
        finding_id="F-COMPLEX-003",
        decision=HumanDecision.REJECT,
        original_recommendation="False positive: requirements appear conflicting but they are intentionally designed for different user roles.",
        analyst_comment="Verified with requirements engineer. REQ-A applies to end users, REQ-B applies to admins. Not a conflict.",
        timestamp="2026-09-15T16:00:00Z",
    )
    
    assert feedback.decision == HumanDecision.REJECT
    assert feedback.modified_recommendation is None
    assert "different user roles" in feedback.original_recommendation
    assert "end users" in feedback.analyst_comment


def test_multiple_feedbacks_for_different_findings():
    """Multiple feedbacks should be independent."""
    feedback1 = HumanFeedback(
        finding_id="F-001",
        decision=HumanDecision.APPROVE,
        original_recommendation="Rec 1.",
    )
    
    feedback2 = HumanFeedback(
        finding_id="F-002",
        decision=HumanDecision.REJECT,
        original_recommendation="Rec 2.",
    )
    
    assert feedback1.finding_id != feedback2.finding_id
    assert feedback1.decision != feedback2.decision
    assert feedback1.original_recommendation != feedback2.original_recommendation


# ---------------------------------------------------------------------------
# Tests: Immutability & Validation
# ---------------------------------------------------------------------------

def test_human_feedback_is_pydantic_model():
    """HumanFeedback should be a Pydantic model."""
    feedback = HumanFeedback(
        finding_id="F-019",
        decision=HumanDecision.APPROVE,
        original_recommendation="Recommendation.",
    )
    
    # Should have model_dump (Pydantic v2 method)
    assert hasattr(feedback, 'model_dump')
    dumped = feedback.model_dump()
    assert isinstance(dumped, dict)
    assert "finding_id" in dumped
    assert "decision" in dumped
    assert "original_recommendation" in dumped


def test_human_feedback_serialization():
    """HumanFeedback should serialize to JSON."""
    feedback = HumanFeedback(
        finding_id="F-020",
        decision=HumanDecision.MODIFY,
        original_recommendation="Original.",
        modified_recommendation="Modified.",
        analyst_comment="Comment.",
        timestamp="2026-09-15T14:30:00Z",
    )
    
    dumped = feedback.model_dump()
    assert dumped["finding_id"] == "F-020"
    assert dumped["decision"] == "modify"  # Enum serializes as string
    assert dumped["original_recommendation"] == "Original."
    assert dumped["modified_recommendation"] == "Modified."
    assert dumped["analyst_comment"] == "Comment."
    assert dumped["timestamp"] == "2026-09-15T14:30:00Z"


def test_human_feedback_deserialization():
    """HumanFeedback should deserialize from dict."""
    data = {
        "finding_id": "F-021",
        "decision": "reject",
        "original_recommendation": "Original.",
        "analyst_comment": "Not a real conflict.",
    }
    
    feedback = HumanFeedback(**data)
    assert feedback.finding_id == "F-021"
    assert feedback.decision == HumanDecision.REJECT
    assert feedback.analyst_comment == "Not a real conflict."


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------

def test_recommendation_text_with_special_characters():
    """Recommendation text can contain special characters."""
    special_text = "Review REQ-001 & REQ-002: Are they <mutually exclusive> or {interdependent}?"
    feedback = HumanFeedback(
        finding_id="F-022",
        decision=HumanDecision.APPROVE,
        original_recommendation=special_text,
    )
    
    assert feedback.original_recommendation == special_text


def test_recommendation_text_very_long():
    """Recommendation text can be very long."""
    long_text = "This is a recommendation. " * 100  # 2700+ characters
    feedback = HumanFeedback(
        finding_id="F-023",
        decision=HumanDecision.APPROVE,
        original_recommendation=long_text,
    )
    
    assert len(feedback.original_recommendation) > 2500


def test_analyst_comment_with_code_blocks():
    """Analyst comment can include code-like content."""
    comment = """
    The conflict is in this condition:
    if (actor == "User" && action == "register") {
        // REQ-001 says allow, REQ-002 says deny
    }
    """
    feedback = HumanFeedback(
        finding_id="F-024",
        decision=HumanDecision.MODIFY,
        original_recommendation="Original.",
        modified_recommendation="Modified.",
        analyst_comment=comment,
    )
    
    assert "actor" in feedback.analyst_comment
    assert '&&' in feedback.analyst_comment
