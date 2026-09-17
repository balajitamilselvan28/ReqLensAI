"""
ReqLens AI V7.2 Streamlit User Interface

A lightweight web interface for the ReqLens AI requirements engineering pipeline.

Features:
- PDF upload and parsing
- LangGraph workflow execution
- Interactive result viewing
- Human-in-the-loop feedback capture
- No automatic SRS modification

Architecture:
- Reuses existing V1-V6 agents via LangGraph workflow
- Preserves all machine-generated recommendations
- Stores human feedback separately (no overwriting)
- Maintains original SRS immutability
"""

import sys
import os

# Add project root to path so we can import src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import logging
import tempfile
from typing import Optional, List, Dict, Any

from src.utils.pdf_parser import parse_pdf, PDFParsingError
from src.graph.workflow import create_workflow
from src.graph.state import ReqLensState
from src.schemas.data_models import (
    HumanFeedback, HumanDecision, Recommendation, Finding, 
    VerificationResult, DependencyResult, Requirement
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================================
# STREAMLIT PAGE CONFIG
# =========================================================================

st.set_page_config(
    page_title="ReqLens AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# SESSION STATE INITIALIZATION
# =========================================================================

if "workflow_result" not in st.session_state:
    st.session_state.workflow_result = None

if "uploaded_pdf_text" not in st.session_state:
    st.session_state.uploaded_pdf_text = ""

if "human_feedback_store" not in st.session_state:
    st.session_state.human_feedback_store = {}  # {finding_id: HumanFeedback}

if "workflow_executed" not in st.session_state:
    st.session_state.workflow_executed = False

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def extract_srs_from_pdf(uploaded_file) -> Optional[str]:
    """
    Extract SRS text from uploaded PDF using the existing PDF parser.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Combined SRS text or None on failure
    """
    if uploaded_file is None:
        st.error("No file uploaded")
        return None
    
    if uploaded_file.type != "application/pdf":
        st.error("Please upload a PDF file")
        return None
    
    try:
        # Temporarily save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        # Parse using existing PDF parser
        pages = parse_pdf(tmp_path)
        
        if not pages:
            st.error("PDF is empty or contains no extractable text")
            return None
        
        # Combine all pages into single SRS text
        srs_text = "\n\n".join([page["text"] for page in pages if page["text"]])
        
        if not srs_text or not srs_text.strip():
            st.error("No text could be extracted from the PDF")
            return None
        
        return srs_text
        
    except PDFParsingError as e:
        st.error(f"PDF parsing error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error while processing PDF: {str(e)}")
        logger.exception("PDF processing failed")
        return None
    finally:
        # Clean up temporary file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


def run_workflow(srs_content: str) -> Optional[ReqLensState]:
    """
    Execute the LangGraph workflow on the provided SRS content.
    
    Args:
        srs_content: The SRS text to analyze
        
    Returns:
        Workflow result state or None on failure
    """
    try:
        with st.spinner("Running ReqLens AI pipeline..."):
            workflow = create_workflow()
            
            # Initialize state
            initial_state: ReqLensState = {
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
            
            # Execute workflow
            result = workflow.invoke(initial_state)
            return result
            
    except Exception as e:
        st.error(f"Workflow execution failed: {str(e)}")
        logger.exception("Workflow execution error")
        return None


def display_document_info(srs_text: str, filename: str):
    """Display basic document information."""
    pages = srs_text.count("\n\n") + 1  # Approximate page count
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filename", filename)
    with col2:
        st.metric("Extracted Pages", pages)
    with col3:
        st.metric("Text Length", f"{len(srs_text)} chars")


def display_requirements(requirements: List[Requirement]):
    """Display extracted requirements."""
    if not requirements:
        st.info("No requirements extracted")
        return
    
    st.write(f"**{len(requirements)} requirements extracted**")
    
    for i, req in enumerate(requirements, 1):
        with st.expander(f"**REQ-{i}**: {req.original_text[:60]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**ID**: {req.requirement_id or 'None'}")
                st.write(f"**Original Text**: {req.original_text}")
                st.write(f"**Actor**: {req.actor or 'None'}")
                st.write(f"**Action**: {req.action or 'None'}")
                st.write(f"**Object**: {req.object or 'None'}")
            
            with col2:
                st.write(f"**Constraints**: {', '.join(req.constraints) if req.constraints else 'None'}")
                st.write(f"**Conditions**: {', '.join(req.conditions) if req.conditions else 'None'}")
                st.write(f"**Priority**: {req.priority or 'None'}")
            
            numerical_values = [
                f"{nv.value} {nv.unit}".strip() for nv in req.numerical_values
            ]
            st.write(f"**Numerical Values**: {', '.join(numerical_values) if numerical_values else 'None'}")
            
            st.write(f"**Source Page**: {req.source_page}")


def display_analysis_summary(result: ReqLensState):
    """Display high-level analysis summary."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Requirements", len(result.get("extracted_requirements", [])))
    with col2:
        st.metric("Candidate Pairs", len(result.get("candidate_pairs", [])))
    with col3:
        findings = result.get("findings", [])
        st.metric("Findings", len(findings))
    with col4:
        recs = result.get("recommendations", [])
        st.metric("Recommendations", len(recs))
    
    # Status breakdown
    findings = result.get("findings", [])
    if findings:
        verified = len([f for f in findings if f.status.value == "verified"])
        potential = len([f for f in findings if f.status.value == "potential"])
        rejected = len([f for f in findings if f.status.value == "rejected"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Verified", verified)
        with col2:
            st.metric("Potential", potential)
        with col3:
            st.metric("Rejected", rejected)


def display_findings(findings: List[Finding], dependency_results: List[DependencyResult]):
    """Display detected inconsistencies with dependency information."""
    if not findings:
        st.info("No inconsistencies detected")
        return
    
    st.write(f"**{len(findings)} inconsistencies detected**")
    
    # Build dependency map
    dep_map = {}
    for dep in dependency_results:
        key = (dep.requirement_id_1, dep.requirement_id_2)
        dep_map[key] = dep
    
    for finding in findings:
        with st.expander(
            f"**{finding.contradiction_type.value}** ({finding.severity.value}) "
            f"→ {finding.requirement_id_1} ↔ {finding.requirement_id_2}"
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Finding ID**: {finding.finding_id}")
                st.write(f"**Type**: {finding.contradiction_type.value}")
                st.write(f"**Severity**: {finding.severity.value}")
                st.write(f"**Status**: {finding.status.value}")
                st.write(f"**Confidence**: {finding.confidence_score:.2f}")
            
            with col2:
                st.write(f"**Requirement 1**: {finding.requirement_id_1}")
                st.write(f"**Requirement 2**: {finding.requirement_id_2}")
                st.write(f"**Explanation**: {finding.explanation}")
            
            st.write(f"**Evidence**: {finding.evidence}")
            
            # Display dependency info if available
            key = (finding.requirement_id_1, finding.requirement_id_2)
            if key in dep_map:
                dep = dep_map[key]
                st.write(f"**Dependency Status**: {dep.dependency_status.value}")
                st.write(f"**Dependency Explanation**: {dep.explanation}")


def display_recommendations(recommendations: List[Recommendation]):
    """Display actionable recommendations."""
    if not recommendations:
        st.info("No recommendations generated")
        return
    
    st.write(f"**{len(recommendations)} recommendations generated**")
    
    for rec in recommendations:
        with st.expander(
            f"**{rec.contradiction_type.value}** → {rec.requirement_id_1} ↔ {rec.requirement_id_2}"
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Finding ID**: {rec.finding_id}")
                st.write(f"**Type**: {rec.contradiction_type.value}")
                st.write(f"**Severity**: {rec.severity.value}")
                st.write(f"**Confidence**: {rec.confidence_score:.2f}")
                st.write(f"**Used LLM**: {rec.used_llm}")
            
            with col2:
                st.write(f"**Requirement 1**: {rec.requirement_id_1}")
                st.write(f"**Requirement 2**: {rec.requirement_id_2}")
            
            st.write("**Explanation**:")
            st.write(rec.explanation)
            
            st.write("**Recommendation**:")
            st.write(rec.recommendation_text)
            
            st.write(f"**Evidence**: {rec.evidence}")


def display_verification(verification_results: List[VerificationResult]):
    """Display verification results."""
    if not verification_results:
        st.info("No verification results")
        return
    
    st.write(f"**{len(verification_results)} recommendations verified**")
    
    for vr in verification_results:
        status_color = {
            "approved": "🟢",
            "needs_review": "🟡",
            "rejected": "🔴"
        }.get(vr.verification_status.value, "⚪")
        
        with st.expander(
            f"{status_color} **{vr.verification_status.value.replace('_', ' ').title()}** "
            f"→ {vr.finding_id}"
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Finding ID**: {vr.finding_id}")
                st.write(f"**Status**: {vr.verification_status.value}")
                st.write(f"**Confidence**: {vr.verification_confidence:.2f}")
                st.write(f"**Requires Review**: {vr.requires_human_review}")
            
            with col2:
                if vr.verified_claims:
                    st.write("**Verified Claims**:")
                    for claim in vr.verified_claims:
                        st.write(f"  ✓ {claim}")
                
                if vr.warnings:
                    st.write("**Warnings**:")
                    for warning in vr.warnings:
                        st.write(f"  ⚠️ {warning}")
            
            st.write("**Final Recommendation**:")
            st.write(vr.final_recommendation)


def display_human_feedback_form(recommendations: List[Recommendation], findings: List[Finding]):
    """Display human-in-the-loop feedback form."""
    if not recommendations:
        st.info("No recommendations to review")
        return
    
    st.write(f"**Review {len(recommendations)} recommendation(s)**")
    
    for rec in recommendations:
        st.write("---")
        
        # Find corresponding finding
        finding = next((f for f in findings if f.finding_id == rec.finding_id), None)
        
        with st.expander(f"Review: {rec.contradiction_type.value} (Confidence: {rec.confidence_score:.2f})"):
            # Display machine recommendation
            st.write("### Machine-Generated Recommendation")
            st.write(f"**Original**: {rec.recommendation_text}")
            
            # Decision buttons
            st.write("### Your Decision")
            decision = st.radio(
                label="Decision",
                options=["Approve", "Modify", "Reject"],
                key=f"decision_{rec.finding_id}"
            )
            
            # Modified recommendation (if modifying)
            modified_rec = None
            if decision == "Modify":
                modified_rec = st.text_area(
                    "Modified recommendation",
                    value="",
                    key=f"modified_{rec.finding_id}"
                )
            
            # Analyst comment
            comment = st.text_area(
                "Analyst comment (optional)",
                value="",
                key=f"comment_{rec.finding_id}"
            )
            
            # Store feedback
            if st.button("Submit", key=f"submit_{rec.finding_id}"):
                decision_map = {
                    "Approve": HumanDecision.APPROVE,
                    "Modify": HumanDecision.MODIFY,
                    "Reject": HumanDecision.REJECT
                }
                
                feedback = HumanFeedback(
                    finding_id=rec.finding_id,
                    decision=decision_map[decision],
                    original_recommendation=rec.recommendation_text,
                    modified_recommendation=modified_rec if decision == "Modify" else None,
                    analyst_comment=comment if comment else None
                )
                
                st.session_state.human_feedback_store[rec.finding_id] = feedback
                st.success(f"Feedback saved for {rec.finding_id}")


def display_original_srs(srs_content: str):
    """Display original SRS text."""
    with st.expander("📄 Original SRS Text"):
        st.text_area(
            "Original SRS (read-only)",
            value=srs_content,
            height=300,
            disabled=True,
            key="original_srs_display"
        )


# =========================================================================
# MAIN APPLICATION
# =========================================================================

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("⚙️ ReqLens AI")
    st.markdown("**V7.2 Streamlit User Interface**")
    st.markdown("Intelligent requirements analysis and verification")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Pipeline Control")
        
        # PDF Upload
        uploaded_file = st.file_uploader(
            "Upload SRS PDF",
            type="pdf",
            help="Select a PDF file containing your Software Requirements Specification"
        )
        
        if uploaded_file is not None:
            st.success(f"✓ Uploaded: {uploaded_file.name}")
            
            # Extract PDF text
            if st.button("Parse PDF", type="primary"):
                srs_text = extract_srs_from_pdf(uploaded_file)
                
                if srs_text:
                    st.session_state.uploaded_pdf_text = srs_text
                    st.success("✓ PDF parsed successfully")
                    st.rerun()
        
        # Analyze button
        if st.session_state.uploaded_pdf_text:
            st.divider()
            
            if st.button("🚀 Analyze Requirements", type="primary"):
                result = run_workflow(st.session_state.uploaded_pdf_text)
                
                if result:
                    st.session_state.workflow_result = result
                    st.session_state.workflow_executed = True
                    st.success("✓ Analysis complete")
                    st.rerun()
                else:
                    st.error("Analysis failed")
        
        # Reset
        st.divider()
        if st.button("🔄 Reset"):
            st.session_state.clear()
            st.rerun()
    
    # Main content area
    if not st.session_state.uploaded_pdf_text:
        st.info("👈 Upload a PDF to get started")
        return
    
    # Document info
    with st.container():
        st.header("📄 Document Information")
        display_document_info(st.session_state.uploaded_pdf_text, "uploaded_srs.pdf")
    
    if not st.session_state.workflow_executed or not st.session_state.workflow_result:
        st.info("👈 Click 'Analyze Requirements' to start the pipeline")
        return
    
    result = st.session_state.workflow_result
    
    # Check for workflow errors
    if result.get("workflow_error"):
        st.error(f"Workflow error: {result['workflow_error']}")
        return
    
    # Analysis Summary
    with st.container():
        st.header("📊 Analysis Summary")
        display_analysis_summary(result)
    
    # Extracted Requirements
    with st.container():
        st.header("📝 Extracted Requirements")
        display_requirements(result.get("extracted_requirements", []))
    
    # Detected Inconsistencies
    with st.container():
        st.header("🔴 Detected Inconsistencies")
        display_findings(
            result.get("findings", []),
            result.get("dependency_results", [])
        )
    
    # Recommendations
    with st.container():
        st.header("💡 Recommendations")
        display_recommendations(result.get("recommendations", []))
    
    # Verification Results
    with st.container():
        st.header("✅ Verification Results")
        display_verification(result.get("verification_results", []))
    
    # Human-in-the-Loop
    with st.container():
        st.header("👤 Human Review")
        st.write("Provide feedback on recommendations below")
        display_human_feedback_form(
            result.get("recommendations", []),
            result.get("findings", [])
        )
        
        # Display stored feedback
        if st.session_state.human_feedback_store:
            st.write("---")
            st.write("**Stored Feedback**")
            for finding_id, feedback in st.session_state.human_feedback_store.items():
                st.write(f"- {finding_id}: {feedback.decision.value}")
    
    # Original SRS
    with st.container():
        display_original_srs(st.session_state.uploaded_pdf_text)
    
    # Final Report
    if result.get("final_report"):
        with st.expander("📋 Pipeline Report"):
            st.text(result["final_report"])


if __name__ == "__main__":
    main()
