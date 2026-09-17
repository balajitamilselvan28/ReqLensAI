import logging
from typing import List
from src.schemas.data_models import Requirement, ReasoningResult, CandidatePair

logger = logging.getLogger(__name__)

class LogicalReasoningAgent:
    """
    Evaluates candidate pairs of requirements to infer logical relationships
    such as numerical overlaps, permission models, and condition triggers.
    Does NOT declare contradictions. Output serves as evidence for V5.
    """
    def __init__(self, llm=None):
        """
        Args:
            llm: Optional LLM for complex semantic reasoning. 
                 If None, the agent falls back to deterministic rule-based reasoning.
        """
        self.llm = llm

    def evaluate_pair(self, req1: Requirement, req2: Requirement) -> ReasoningResult:
        """
        Evaluates a single pair of requirements for logical relationships.
        """
        result = ReasoningResult(
            requirement_id_1=req1.requirement_id or "unknown",
            requirement_id_2=req2.requirement_id or "unknown",
            reasoning_details=""
        )
        
        details = []
        
        # 1. Action Overlap
        action_match = False
        if req1.action and req2.action and req1.action.lower() == req2.action.lower():
            action_match = True
                
        # Object Overlap
        object_match = False
        if req1.object and req2.object and req1.object.lower() == req2.object.lower():
            object_match = True
            
        if action_match:
            result.has_overlapping_actions = True
            
        if action_match and object_match:
            details.append(f"Both requirements target the action '{req1.action}' on '{req1.object}'.")
        elif action_match:
            details.append(f"Both requirements share the action '{req1.action}'.")
            
        # 2. Permission / Actor Differences
        if action_match:
            if req1.actor and req2.actor and req1.actor.lower() != req2.actor.lower():
                result.has_different_actors = True
                details.append(f"Different actors are specified ('{req1.actor}' vs '{req2.actor}').")
                
        # 3. Numerical Overlap
        if req1.numerical_values and req2.numerical_values:
            for n1 in req1.numerical_values:
                for n2 in req2.numerical_values:
                    # Match if units are identical or both are None
                    if n1.unit == n2.unit:
                        result.has_numerical_overlap = True
                        unit_str = n1.unit if n1.unit else "units"
                        result.numerical_comparison = f"{n1.value} vs {n2.value} {unit_str}"
                        details.append(f"Found related numerical constraints: {n1.value} and {n2.value} ({unit_str}).")
                        break
                        
        # 4. Conditional Implications
        if req1.conditions or req2.conditions:
            result.has_conditional_implication = True
            details.append("Conditional triggers are present.")
            
        if not details:
            result.reasoning_details = "No explicit logical intersections (action, actor, or numerical) found deterministically."
        else:
            result.reasoning_details = " ".join(details)
            
        return result
        
    def evaluate_candidates(self, requirements: List[Requirement], candidate_pairs: List[CandidatePair]) -> List[ReasoningResult]:
        """
        Evaluates a list of CandidatePair objects.
        """
        req_map = {r.requirement_id: r for r in requirements if r.requirement_id}
        results = []
        
        for pair in candidate_pairs:
            r1 = req_map.get(pair.requirement_id_1)
            r2 = req_map.get(pair.requirement_id_2)
            
            if r1 and r2:
                results.append(self.evaluate_pair(r1, r2))
                
        return results
