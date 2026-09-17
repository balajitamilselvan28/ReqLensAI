"""Real LangGraph StateGraph for ReqLens AI V7.1 orchestration.

Integrates V1-V6 agents into a complete pipeline:
1. Ingestion → PDF parsing
2. Extraction → Requirement extraction
3. Retrieval → Candidate pair generation
4. Representation → Knowledge graph transformation
5. Reasoning → Logical relationship evaluation
6. Detection → Contradiction identification
7. Dependency → Structural verification
8. Recommendation → Actionable insight generation
9. Verification → Automated QA before HITL
10. HITL → Human feedback capture (no automatic decisions)

This is NOT a UI. It is purely orchestration logic.
No real LLM calls unless explicitly configured.
No automatic SRS modification.
No automatic requirement selection.
"""
import logging
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from src.graph.state import ReqLensState
from src.agents.extractor import RequirementExtractionAgent
from src.agents.retrieval import SemanticRetrievalAgent
from src.agents.representation import KnowledgeRepresentationAgent
from src.agents.reasoning import LogicalReasoningAgent
from src.agents.contradiction import ContradictionDetectionAgent
from src.agents.dependency import DependencyVerificationAgent
from src.agents.recommendation import RecommendationGenerationAgent
from src.agents.verification import VerificationAgent
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import RequirementVectorStore

logger = logging.getLogger(__name__)


# =========================================================================
# NODE IMPLEMENTATIONS
# =========================================================================

def node_ingest(state: ReqLensState) -> ReqLensState:
    """
    Stage 1: Ingestion
    
    No-op in this pipeline as SRS content is pre-loaded.
    In production, this would call parse_pdf() and populate srs_content.
    """
    logger.info("Stage 1: Ingestion (srs_content already populated)")
    return {}


def _deterministic_extract(srs_content: str):
    """Extract requirement lines when no LLM-backed extractor is available."""
    from src.schemas.data_models import Requirement

    requirements = []
    generated_id_counter = 1
    for line in srs_content.split("\n"):
        line = line.strip()
        if line and ("shall" in line.lower() or "must" in line.lower() or "should" in line.lower()):
            requirements.append(
                Requirement(
                    requirement_id=f"REQ-{generated_id_counter:03d}",
                    original_text=line,
                    actor=None,
                    action=None,
                    object=None,
                    source_page=1,
                )
            )
            generated_id_counter += 1
    return requirements


def _srs_pages(srs_content: str):
    """Adapt the UI's combined page text to the V2 extractor page contract."""
    return [
        {"page_number": page_number, "text": text}
        for page_number, text in enumerate(srs_content.split("\n\n"), 1)
        if text.strip()
    ]


def node_extract(state: ReqLensState) -> ReqLensState:
    """
    Stage 2: Extraction

    Uses Gemini through the existing V2 RequirementExtractionAgent when configured,
    with the deterministic fallback retained for unavailable or failed LLM calls.
    """
    logger.info("Stage 2: Extraction")
    
    if not state.get("srs_content"):
        logger.warning("No SRS content to extract from")
        return {"extracted_requirements": []}
    
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model="gemini-3.8-flash",
                temperature=0,
                google_api_key=api_key,
            )
            requirements = RequirementExtractionAgent(llm=llm).extract(_srs_pages(state["srs_content"]))
            if requirements:
                logger.info("Extracted %d requirements with Gemini", len(requirements))
                return {"extracted_requirements": requirements}
            logger.warning("Gemini extraction returned no requirements; using deterministic fallback")
        else:
            logger.info("GOOGLE_API_KEY is unavailable; using deterministic extraction fallback")
    except Exception as e:
        logger.warning("Gemini extraction unavailable; using deterministic fallback: %s", e)

    requirements = _deterministic_extract(state["srs_content"])
    logger.info("Extracted %d requirements with deterministic fallback", len(requirements))
    return {"extracted_requirements": requirements}


def node_retrieve(state: ReqLensState) -> ReqLensState:
    """
    Stage 3: Retrieval
    
    Uses SemanticRetrievalAgent to find semantically related requirement pairs.
    """
    logger.info("Stage 3: Retrieval")
    
    if not state.get("extracted_requirements") or len(state["extracted_requirements"]) < 2:
        logger.info("Insufficient requirements for semantic retrieval")
        return {"candidate_pairs": []}
    
    try:
        # Initialize the CPU-friendly Sentence Transformers embedding model.
        vector_store = RequirementVectorStore(
            embedding_model=EmbeddingModel("all-MiniLM-L6-v2")
        )
        
        retriever = SemanticRetrievalAgent(
            vector_store=vector_store,
            top_k=3,
            threshold=0.7
        )
        
        candidate_pairs = retriever.generate_candidate_pairs(
            state["extracted_requirements"]
        )
        
        logger.info(f"Generated {len(candidate_pairs)} candidate pairs")
        return {"candidate_pairs": candidate_pairs}
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            "candidate_pairs": [],
            "workflow_error": f"Retrieval stage failed: {str(e)}"
        }


def node_represent(state: ReqLensState) -> ReqLensState:
    """
    Stage 4: Representation
    
    Uses KnowledgeRepresentationAgent to transform requirements into knowledge graphs.
    """
    logger.info("Stage 4: Representation")
    
    if not state.get("extracted_requirements"):
        logger.info("No requirements to represent")
        return {"knowledge_representations": []}
    
    try:
        representer = KnowledgeRepresentationAgent()
        knowledge_reps = representer.transform(state["extracted_requirements"])
        
        logger.info(f"Created {len(knowledge_reps)} knowledge representations")
        return {"knowledge_representations": knowledge_reps}
        
    except Exception as e:
        logger.error(f"Representation failed: {e}")
        return {
            "knowledge_representations": [],
            "workflow_error": f"Representation stage failed: {str(e)}"
        }


def node_reason(state: ReqLensState) -> ReqLensState:
    """
    Stage 5: Reasoning
    
    Uses LogicalReasoningAgent to evaluate logical relationships between candidate pairs.
    """
    logger.info("Stage 5: Reasoning")
    
    if not state.get("candidate_pairs"):
        logger.info("No candidate pairs to reason about")
        return {"reasoning_results": []}
    
    try:
        reasoner = LogicalReasoningAgent(llm=None)  # Deterministic by default
        reasoning_results = reasoner.evaluate_candidates(
            state["extracted_requirements"],
            state["candidate_pairs"]
        )
        
        logger.info(f"Generated {len(reasoning_results)} reasoning results")
        return {"reasoning_results": reasoning_results}
        
    except Exception as e:
        logger.error(f"Reasoning failed: {e}")
        return {
            "reasoning_results": [],
            "workflow_error": f"Reasoning stage failed: {str(e)}"
        }


def node_detect(state: ReqLensState) -> ReqLensState:
    """
    Stage 6: Detection
    
    Uses ContradictionDetectionAgent to identify potential contradictions.
    """
    logger.info("Stage 6: Detection")
    
    if not state.get("candidate_pairs"):
        logger.info("No candidate pairs to detect contradictions in")
        return {"findings": []}
    
    try:
        # Build knowledge map for detector
        knowledge_map = {
            kr.requirement_id: kr 
            for kr in state.get("knowledge_representations", [])
        }
        
        detector = ContradictionDetectionAgent(llm=None)  # Deterministic by default
        findings = detector.detect(
            requirements=state["extracted_requirements"],
            candidate_pairs=state["candidate_pairs"],
            reasoning_results=state.get("reasoning_results", []),
            knowledge_map=knowledge_map if knowledge_map else None
        )
        
        logger.info(f"Detected {len(findings)} potential findings")
        return {"findings": findings}
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return {
            "findings": [],
            "workflow_error": f"Detection stage failed: {str(e)}"
        }


def node_dependency(state: ReqLensState) -> ReqLensState:
    """
    Stage 7: Dependency Verification
    
    Uses DependencyVerificationAgent to validate findings against structural evidence.
    """
    logger.info("Stage 7: Dependency Verification")
    
    if not state.get("findings"):
        logger.info("No findings to verify dependencies")
        return {"findings": [], "dependency_results": []}
    
    try:
        verifier = DependencyVerificationAgent()
        updated_findings, dep_results = verifier.verify(
            findings=state["findings"],
            all_findings=state.get("findings", [])
        )
        
        logger.info(f"Verified {len(updated_findings)} findings, {len(dep_results)} dependency results")
        return {
            "findings": updated_findings,
            "dependency_results": dep_results
        }
        
    except Exception as e:
        logger.error(f"Dependency verification failed: {e}")
        return {
            "findings": state.get("findings", []),
            "dependency_results": [],
            "workflow_error": f"Dependency stage failed: {str(e)}"
        }


def node_recommend(state: ReqLensState) -> ReqLensState:
    """
    Stage 8: Recommendation Generation
    
    Uses RecommendationGenerationAgent to produce actionable insights.
    """
    logger.info("Stage 8: Recommendation Generation")
    
    if not state.get("findings"):
        logger.info("No findings to generate recommendations for")
        return {"recommendations": []}
    
    try:
        recommender = RecommendationGenerationAgent(llm=None)  # Deterministic by default
        recommendations = recommender.generate(
            findings=state["findings"],
            dependency_results=state.get("dependency_results", [])
        )
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return {"recommendations": recommendations}
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        return {
            "recommendations": [],
            "workflow_error": f"Recommendation stage failed: {str(e)}"
        }


def node_verify(state: ReqLensState) -> ReqLensState:
    """
    Stage 9: Verification
    
    Uses VerificationAgent to QA recommendations before human review.
    """
    logger.info("Stage 9: Verification")
    
    if not state.get("recommendations"):
        logger.info("No recommendations to verify")
        return {"verification_results": []}
    
    try:
        verifier = VerificationAgent()
        verification_results = verifier.verify(
            recommendations=state["recommendations"],
            findings=state.get("findings", [])
        )
        
        logger.info(f"Verified {len(verification_results)} recommendations")
        return {"verification_results": verification_results}
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return {
            "verification_results": [],
            "workflow_error": f"Verification stage failed: {str(e)}"
        }


def node_human_feedback(state: ReqLensState) -> ReqLensState:
    """
    Stage 10: Human Feedback (HITL)
    
    No automatic decisions. Human feedback is captured separately.
    This node is a placeholder for the HITL interface.
    It does NOT consume or modify human_feedback autonomously.
    """
    logger.info("Stage 10: Human Feedback (awaiting analyst input)")
    
    # In production: This would be integrated with a UI/API that captures
    # HumanFeedback objects submitted by the analyst.
    # For now: Just acknowledge and pass through.
    
    return {
        "final_report": _generate_final_report(state)
    }


def _generate_final_report(state: ReqLensState) -> str:
    """Generate a summary report of the workflow execution."""
    findings_count = len(state.get("findings", []))
    recommendations_count = len(state.get("recommendations", []))
    verification_count = len(state.get("verification_results", []))
    human_feedback_count = len(state.get("human_feedback", []))
    
    error_msg = state.get("workflow_error")
    
    report = f"""
ReqLens AI V7.1 Pipeline Report
================================

Stage Summary:
- Extracted Requirements: {len(state.get("extracted_requirements", []))}
- Candidate Pairs: {len(state.get("candidate_pairs", []))}
- Knowledge Representations: {len(state.get("knowledge_representations", []))}
- Reasoning Results: {len(state.get("reasoning_results", []))}
- Findings (Contradictions): {findings_count}
- Dependency Results: {len(state.get("dependency_results", []))}
- Recommendations: {recommendations_count}
- Verification Results: {verification_count}
- Human Feedback Decisions: {human_feedback_count}

Status:
- {error_msg if error_msg else 'No errors'}

Next Steps:
- Review {findings_count} finding(s) with human analyst
- Implement {recommendations_count} recommendation(s) as approved by analyst
- Track {human_feedback_count} feedback decision(s)
"""
    return report.strip()


# =========================================================================
# GRAPH CONSTRUCTION
# =========================================================================

def create_workflow() -> StateGraph:
    """
    Creates and compiles the complete LangGraph StateGraph for ReqLens AI.
    
    Returns:
        StateGraph: The compiled workflow ready for invocation.
    """
    workflow = StateGraph(ReqLensState)

    # Add all nodes
    workflow.add_node("ingest", node_ingest)
    workflow.add_node("extract", node_extract)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("represent", node_represent)
    workflow.add_node("reason", node_reason)
    workflow.add_node("detect", node_detect)
    workflow.add_node("dependency", node_dependency)
    workflow.add_node("recommend", node_recommend)
    workflow.add_node("verify", node_verify)
    workflow.add_node("human_feedback", node_human_feedback)

    # Set entry point
    workflow.set_entry_point("ingest")

    # Add linear edges through the pipeline
    workflow.add_edge("ingest", "extract")
    workflow.add_edge("extract", "retrieve")
    workflow.add_edge("retrieve", "represent")
    workflow.add_edge("represent", "reason")
    workflow.add_edge("reason", "detect")
    workflow.add_edge("detect", "dependency")
    workflow.add_edge("dependency", "recommend")
    workflow.add_edge("recommend", "verify")
    workflow.add_edge("verify", "human_feedback")
    workflow.add_edge("human_feedback", END)

    logger.info("LangGraph workflow constructed successfully")
    return workflow.compile()

