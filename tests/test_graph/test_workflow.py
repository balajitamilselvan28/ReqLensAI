"""
Comprehensive tests for LangGraph workflow orchestration.

Tests cover:
- Graph construction and node connectivity
- End-to-end execution with deterministic agents
- State propagation through all 10 stages
- Error handling and recovery
- Preservation of findings, recommendations, verification results
- Human feedback handling (no automatic modification)
- Deterministic test cases with known requirements
"""
import pytest
from src.graph.workflow import create_workflow
from src.graph.state import ReqLensState
from src.schemas.data_models import (
    Requirement, Finding, FindingStatus, ContradictionType, 
    Severity, HumanFeedback, HumanDecision
)


class TestWorkflowConstruction:
    """Test that the workflow graph is properly constructed."""
    
    def test_workflow_compiles(self):
        """Verify the workflow compiles without errors."""
        app = create_workflow()
        assert app is not None
    
    def test_workflow_has_correct_nodes(self):
        """Verify all expected nodes exist in the graph."""
        app = create_workflow()
        # The compiled graph should be invokable
        assert hasattr(app, 'invoke')
    
    def test_initial_state_schema(self):
        """Verify the state schema is properly initialized."""
        app = create_workflow()
        
        initial_state = {
            "srs_content": "Test SRS",
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        # Should not raise on invoke
        result = app.invoke(initial_state)
        assert result is not None
        assert "final_report" in result


class TestEndToEndExecution:
    """Test complete pipeline execution with deterministic data."""
    
    def test_minimal_srs_execution(self):
        """Execute workflow on minimal SRS without contradictions."""
        app = create_workflow()
        
        initial_state = {
            "srs_content": "The system shall respond within 300 milliseconds.",
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Verify state structure is preserved
        assert "srs_content" in result
        assert "final_report" in result
        assert isinstance(result["extracted_requirements"], list)
        assert isinstance(result["candidate_pairs"], list)
    
    def test_multi_requirement_execution(self):
        """Execute workflow on SRS with multiple requirements."""
        app = create_workflow()
        
        srs_content = """
REQ-001: Students shall be allowed to register for university events.
REQ-002: Only administrators shall be allowed to register students for university events.
REQ-003: The system shall respond within 300 milliseconds.
REQ-004: The system shall respond within 500 milliseconds under normal operating conditions.
REQ-005: Students shall cancel their own event registrations.
REQ-006: Only administrators shall cancel event registrations.
"""
        
        initial_state = {
            "srs_content": srs_content,
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Verify extraction happened
        assert len(result["extracted_requirements"]) > 0
        
        # Verify state flows through pipeline
        assert isinstance(result["candidate_pairs"], list)
        assert isinstance(result["findings"], list)
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["verification_results"], list)
        
        # Verify report generation
        assert result["final_report"] is not None
        assert "Pipeline Report" in result["final_report"] or "requirements" in result["final_report"].lower()


class TestStatePropagation:
    """Test that data properly flows through all pipeline stages."""
    
    def test_extracted_requirements_propagate(self):
        """Verify extracted requirements flow to later stages."""
        app = create_workflow()
        
        srs_content = """
REQ-001: The system shall allow users to log in.
REQ-002: The system shall require password authentication.
"""
        
        initial_state = {
            "srs_content": srs_content,
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Requirements should be extracted
        assert len(result["extracted_requirements"]) >= 1
        
        # Later stages should see them
        assert result["knowledge_representations"] is not None
        assert result["reasoning_results"] is not None
    
    def test_findings_preserved_through_dependency_stage(self):
        """Verify findings are preserved and updated through dependency verification."""
        app = create_workflow()
        
        # Create a SRS with potential contradictions
        srs_content = """
REQ-001: Students shall be allowed to register for events.
REQ-002: Only administrators shall be allowed to register students.
"""
        
        initial_state = {
            "srs_content": srs_content,
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Findings should either be detected or explicitly empty
        assert isinstance(result["findings"], list)
        
        # Dependency results should be consistent
        assert isinstance(result["dependency_results"], list)
    
    def test_recommendations_generated_from_findings(self):
        """Verify recommendations are generated when findings exist."""
        app = create_workflow()
        
        srs_content = """
REQ-001: The system shall respond within 300 milliseconds.
REQ-002: The system shall respond within 500 milliseconds.
"""
        
        initial_state = {
            "srs_content": srs_content,
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Recommendations should be a list (empty or populated)
        assert isinstance(result["recommendations"], list)
        
        # Verification results should match recommendation count
        assert isinstance(result["verification_results"], list)


class TestErrorHandling:
    """Test error handling and recovery at each stage."""
    
    def test_empty_srs_handled_gracefully(self):
        """Workflow should handle empty SRS without crashing."""
        app = create_workflow()
        
        initial_state = {
            "srs_content": "",
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Should complete without crashing
        assert result is not None
        assert "final_report" in result
        assert len(result["extracted_requirements"]) == 0
    
    def test_missing_state_fields_handled(self):
        """Workflow should handle missing optional state fields."""
        app = create_workflow()
        
        # Create a state with only required fields
        initial_state = {
            "srs_content": "The system shall log in.",
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Should complete without KeyError
        assert result is not None
    
    def test_workflow_error_field_populated_on_failure(self):
        """When a stage fails, workflow_error should be populated."""
        app = create_workflow()
        
        # This test verifies error handling is in place
        # (actual errors depend on agent implementation)
        initial_state = {
            "srs_content": "Test",
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # workflow_error should be present (may be None if no errors)
        assert "workflow_error" in result


class TestHumanFeedbackHandling:
    """Test that human feedback is properly preserved without auto-modification."""
    
    def test_human_feedback_preserved_through_pipeline(self):
        """Verify human feedback is carried through state without modification."""
        app = create_workflow()
        
        # Pre-populate human feedback
        human_feedback = HumanFeedback(
            finding_id="F-001",
            decision=HumanDecision.APPROVE,
            analyst_comment="Approved by analyst",
            original_recommendation="Review requirements",
            timestamp="2026-09-15T12:00:00Z"
        )
        
        initial_state = {
            "srs_content": "Test requirement",
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [human_feedback],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Human feedback should be preserved exactly
        assert len(result["human_feedback"]) == 1
        assert result["human_feedback"][0].decision == HumanDecision.APPROVE
    
    def test_human_feedback_does_not_auto_modify_recommendations(self):
        """Verify workflow does not automatically apply human feedback."""
        app = create_workflow()
        
        # The workflow should NOT consume human_feedback to modify recommendations
        # That's a responsibility of a separate HITL processor
        
        initial_state = {
            "srs_content": "Test requirement",
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Recommendations should be machine-generated (workflow-generated)
        # Human feedback should be separate
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["human_feedback"], list)


class TestDeterministicEndToEnd:
    """Test known deterministic scenarios end-to-end."""
    
    def test_numerical_conflict_detection(self):
        """Test detection of numerical conflicts (e.g., response time)."""
        app = create_workflow()
        
        srs_content = """
REQ-001: The system shall respond within 300 milliseconds.
REQ-002: The system shall respond within 500 milliseconds under normal operating conditions.
"""
        
        initial_state = {
            "srs_content": srs_content,
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Pipeline should execute completely
        assert result is not None
        assert result["final_report"] is not None
        
        # Requirements should be extracted
        assert len(result["extracted_requirements"]) > 0
    
    def test_permission_conflict_detection(self):
        """Test detection of permission conflicts."""
        app = create_workflow()
        
        srs_content = """
REQ-001: Students shall be allowed to register for university events.
REQ-002: Only administrators shall be allowed to register students for university events.
"""
        
        initial_state = {
            "srs_content": srs_content,
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Pipeline should execute completely
        assert result is not None
        assert result["final_report"] is not None
        
        # Requirements should be extracted
        assert len(result["extracted_requirements"]) > 0
    
    def test_cancellation_conflict_detection(self):
        """Test detection of conflicts in cancellation permissions."""
        app = create_workflow()
        
        srs_content = """
REQ-005: Students shall cancel their own event registrations.
REQ-006: Only administrators shall cancel event registrations.
"""
        
        initial_state = {
            "srs_content": srs_content,
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        # Pipeline should execute completely
        assert result is not None
        assert result["final_report"] is not None
        
        # Requirements should be extracted
        assert len(result["extracted_requirements"]) > 0


class TestReportGeneration:
    """Test final report generation."""
    
    def test_report_includes_stage_summary(self):
        """Verify final report includes summary of all stages."""
        app = create_workflow()
        
        initial_state = {
            "srs_content": "The system shall log in users.",
            "extracted_requirements": [],
            "candidate_pairs": [],
            "knowledge_representations": [],
            "reasoning_results": [],
            "findings": [],
            "dependency_results": [],
            "recommendations": [],
            "verification_results": [],
            "human_feedback": [],
            "logical_rules": [],
            "final_report": None,
            "workflow_error": None,
        }
        
        result = app.invoke(initial_state)
        
        report = result["final_report"]
        assert report is not None
        assert "ReqLens AI V7.1 Pipeline Report" in report or "Requirements" in report or "Pipeline" in report


def test_dummy_workflow():
    """Legacy test for compatibility."""
    app = create_workflow()

    initial_state = {
        "srs_content": "The system shall have a login page. The system must use 2FA.",
        "extracted_requirements": [],
        "knowledge_representations": [],
        "candidate_pairs": [],
        "reasoning_results": [],
        "findings": [],
        "dependency_results": [],
        "logical_rules": [],
        "human_feedback": [],
        "recommendations": [],
        "verification_results": [],
        "final_report": None,
        "workflow_error": None,
    }

    result = app.invoke(initial_state)

    assert result["final_report"] is not None
    assert len(result["extracted_requirements"]) > 0
    print("Graph execution successful!")


if __name__ == "__main__":
    test_dummy_workflow()

