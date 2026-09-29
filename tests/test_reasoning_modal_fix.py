"""
Regression test for modal-verb normalization in LogicalReasoningAgent.

Root cause: Gemini inconsistently includes/excludes modal verbs ('shall', 'must', etc.)
in the extracted action field. Without normalization, 'shall register' != 'register'
causing has_overlapping_actions=False and findings=0 in the real pipeline.

This test verifies that action matching is modal-verb-agnostic.
"""
import pytest
from src.schemas.data_models import Requirement, CandidatePair
from src.agents.reasoning import LogicalReasoningAgent, _normalize_action
from src.agents.contradiction import ContradictionDetectionAgent


# ---------------------------------------------------------------------------
# Unit tests: _normalize_action helper
# ---------------------------------------------------------------------------

def test_normalize_strips_shall():
    assert _normalize_action("shall register") == "register"

def test_normalize_strips_must():
    assert _normalize_action("must respond") == "respond"

def test_normalize_strips_should():
    assert _normalize_action("should allow") == "allow"

def test_normalize_strips_multiple_modals():
    assert _normalize_action("shall be allowed") == "be allowed"

def test_normalize_no_modal_unchanged():
    assert _normalize_action("register") == "register"

def test_normalize_empty_string():
    assert _normalize_action("") == ""

def test_normalize_is_case_insensitive():
    assert _normalize_action("SHALL Register") == "register"


# ---------------------------------------------------------------------------
# Integration: modal mismatch no longer kills permission conflict detection
# ---------------------------------------------------------------------------

def make_req(rid, actor, action, obj="events"):
    return Requirement(
        requirement_id=rid,
        original_text=f"{rid}: {actor} {action} {obj}.",
        actor=actor, action=action, object=obj,
        source_page=1,
    )

def make_pair(id1, id2, sim=0.90):
    return CandidatePair(
        requirement_id_1=id1, requirement_id_2=id2,
        similarity_score=sim,
        requirement_text_1=f"{id1} text", requirement_text_2=f"{id2} text",
    )


@pytest.mark.parametrize("a1,a2", [
    ("shall register", "register"),       # Gemini includes modal for one only
    ("register", "shall register"),       # reversed
    ("shall register", "shall register"), # both include modal
    ("must respond", "respond"),          # different modal verb
    ("should allow", "allow"),
])
def test_action_match_modal_agnostic(a1, a2):
    """Overlapping actions detected regardless of modal verb inclusion."""
    req1 = make_req("REQ-001", "Students", a1)
    req2 = make_req("REQ-002", "administrators", a2)
    agent = LogicalReasoningAgent()
    result = agent.evaluate_pair(req1, req2)
    assert result.has_overlapping_actions is True, (
        f"Expected has_overlapping_actions=True for '{a1}' vs '{a2}'"
    )


def test_permission_conflict_detected_with_modal_mismatch():
    """
    Full pipeline regression: REQ-001/REQ-002 with modal mismatch must produce
    a permission conflict Finding and not be swallowed at reasoning stage.
    """
    req1 = make_req("REQ-001", "Students", "shall register", "public university events")
    req2 = make_req("REQ-002", "administrators", "register", "public university events")
    pair = make_pair("REQ-001", "REQ-002", sim=0.90)

    agent_reason = LogicalReasoningAgent()
    rr = agent_reason.evaluate_pair(req1, req2)
    assert rr.has_overlapping_actions is True, "Reasoning must detect overlapping actions"
    assert rr.has_different_actors is True, "Reasoning must detect different actors"

    agent_contra = ContradictionDetectionAgent()
    findings = agent_contra.detect(
        requirements=[req1, req2],
        candidate_pairs=[pair],
        reasoning_results=[rr],
    )
    assert len(findings) == 1, "Must produce exactly one finding"
    from src.schemas.data_models import ContradictionType, Severity
    assert findings[0].contradiction_type == ContradictionType.PERMISSION_CONFLICT
    assert findings[0].severity == Severity.HIGH
    assert findings[0].confidence_score >= 0.60
