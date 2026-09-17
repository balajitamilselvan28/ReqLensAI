import logging
from typing import List, Set
from src.schemas.data_models import Requirement, CandidatePair
from src.retrieval.vector_store import RequirementVectorStore

logger = logging.getLogger(__name__)

class SemanticRetrievalAgent:
    """
    Agent responsible for generating semantically related candidate pairs of requirements.
    This is NOT a contradiction detection agent. High similarity only implies relevance.
    """
    def __init__(self, vector_store: RequirementVectorStore, top_k: int = 3, threshold: float = 0.7):
        """
        Initializes the Semantic Retrieval Agent.
        
        Args:
            vector_store: The configured vector store containing the requirements.
            top_k: The number of top similar requirements to fetch per requirement.
            threshold: The minimum cosine similarity score to consider them a pair.
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.threshold = threshold

    def generate_candidate_pairs(self, requirements: List[Requirement]) -> List[CandidatePair]:
        """
        Processes a list of requirements, stores them, and finds candidate pairs.
        
        Args:
            requirements: List of parsed and structured Requirement objects.
            
        Returns:
            List of CandidatePair objects representing related requirements.
        """
        if not requirements or len(requirements) < 2:
            return []
            
        # 1. Populate the vector store with all requirements
        self.vector_store.add_requirements(requirements)
        
        candidate_pairs = []
        seen_pairs: Set[tuple] = set()
        
        for req in requirements:
            if not req.original_text or not req.original_text.strip() or not req.requirement_id:
                continue
                
            # 2. Retrieve similar items for each requirement
            similar_items = self.vector_store.search_similar(
                query_text=req.original_text,
                top_k=self.top_k,
                threshold=self.threshold
            )
            
            for item in similar_items:
                matched_id = item["metadata"].get("requirement_id")
                if not matched_id or matched_id == "unknown":
                    continue
                    
                # Exclude self-matches
                if matched_id == req.requirement_id:
                    continue
                    
                # Create a canonical tuple key to avoid duplicate pairs (A-B and B-A are the same)
                pair_key = tuple(sorted([req.requirement_id, matched_id]))
                if pair_key in seen_pairs:
                    continue
                    
                seen_pairs.add(pair_key)
                
                pair = CandidatePair(
                    requirement_id_1=req.requirement_id,
                    requirement_id_2=matched_id,
                    similarity_score=item["similarity"],
                    requirement_text_1=req.original_text,
                    requirement_text_2=item["metadata"].get("original_text", "")
                )
                candidate_pairs.append(pair)
                
        return candidate_pairs
