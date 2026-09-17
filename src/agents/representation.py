import uuid
import logging
from typing import List
from src.schemas.data_models import Requirement, KnowledgeRepresentation, Entity, Relation

logger = logging.getLogger(__name__)

class KnowledgeRepresentationAgent:
    """
    Transforms structured requirements into a machine-readable knowledge graph format (nodes and edges).
    This deterministic transformation does not require an LLM.
    """
    
    def transform(self, requirements: List[Requirement]) -> List[KnowledgeRepresentation]:
        """
        Processes a list of requirements and converts them into KnowledgeRepresentation objects.
        """
        representations = []
        for req in requirements:
            if not req.requirement_id:
                continue
                
            entities = []
            relations = []
            
            # Root node for the requirement itself
            req_entity_id = f"req-{req.requirement_id}"
            entities.append(Entity(id=req_entity_id, label="Requirement", name=req.requirement_id))
            
            # Actor
            actor_id = None
            if req.actor:
                actor_id = f"actor-{uuid.uuid4().hex[:8]}"
                entities.append(Entity(id=actor_id, label="Actor", name=req.actor))
                relations.append(Relation(source_id=req_entity_id, target_id=actor_id, relation_type="has_actor"))
                
            # Action
            action_id = None
            if req.action:
                action_id = f"action-{uuid.uuid4().hex[:8]}"
                entities.append(Entity(id=action_id, label="Action", name=req.action))
                if actor_id:
                    relations.append(Relation(source_id=actor_id, target_id=action_id, relation_type="performs"))
                else:
                    relations.append(Relation(source_id=req_entity_id, target_id=action_id, relation_type="has_action"))
                    
            # Object
            if req.object:
                object_id = f"object-{uuid.uuid4().hex[:8]}"
                entities.append(Entity(id=object_id, label="Object", name=req.object))
                if action_id:
                    relations.append(Relation(source_id=action_id, target_id=object_id, relation_type="targets"))
                else:
                    relations.append(Relation(source_id=req_entity_id, target_id=object_id, relation_type="has_object"))
                    
            # Conditions
            for cond in req.conditions:
                cond_id = f"cond-{uuid.uuid4().hex[:8]}"
                entities.append(Entity(id=cond_id, label="Condition", name=cond))
                relations.append(Relation(source_id=req_entity_id, target_id=cond_id, relation_type="has_condition"))
                
            # Constraints
            for constr in req.constraints:
                constr_id = f"constr-{uuid.uuid4().hex[:8]}"
                entities.append(Entity(id=constr_id, label="Constraint", name=constr))
                relations.append(Relation(source_id=req_entity_id, target_id=constr_id, relation_type="has_constraint"))
                
            # Numerical Values
            for num in req.numerical_values:
                num_id = f"num-{uuid.uuid4().hex[:8]}"
                name = f"{num.value} {num.unit}" if num.unit else str(num.value)
                entities.append(Entity(id=num_id, label="NumericalValue", name=name))
                relations.append(Relation(source_id=req_entity_id, target_id=num_id, relation_type="has_numerical_value"))
                
            representations.append(KnowledgeRepresentation(
                requirement_id=req.requirement_id,
                original_text=req.original_text,
                entities=entities,
                relations=relations
            ))
            
        return representations
