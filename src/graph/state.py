from typing import TypedDict, List, Any, Optional
from src.schemas.data_models import (
    Requirement, Rule, Finding, HumanFeedback, CandidatePair,
    KnowledgeRepresentation, ReasoningResult, DependencyResult,
    Recommendation, VerificationResult
)

class ReqLensState(TypedDict):
    """The state dictionary for the ReqLens AI LangGraph.
    
    Carries data through all 10 pipeline stages:
    1. Ingestion (PDF parsing)
    2. Extraction (Requirements)
    3. Retrieval (Candidate pairs)
    4. Representation (Knowledge graph)
    5. Reasoning (Logical relationships)
    6. Detection (Contradiction findings)
    7. Dependency (Structural verification)
    8. Recommendation (Actionable insights)
    9. Verification (Automated QA)
    10. Human Feedback (HITL decisions)
    """
    # Stage 1: Ingestion
    srs_content: str
    
    # Stage 2: Extraction
    extracted_requirements: List[Requirement]
    
    # Stage 3: Retrieval
    candidate_pairs: List[CandidatePair]
    
    # Stage 4: Representation
    knowledge_representations: List[KnowledgeRepresentation]
    
    # Stage 5: Reasoning
    reasoning_results: List[ReasoningResult]
    
    # Stage 6: Detection
    findings: List[Finding]
    
    # Stage 7: Dependency
    dependency_results: List[DependencyResult]
    
    # Stage 8: Recommendation
    recommendations: List[Recommendation]
    
    # Stage 9: Verification
    verification_results: List[VerificationResult]
    
    # Stage 10: Human Feedback
    human_feedback: List[HumanFeedback]
    
    # Legacy/metadata
    logical_rules: List[Rule]
    final_report: Optional[str]
    
    # Error tracking
    workflow_error: Optional[str]
