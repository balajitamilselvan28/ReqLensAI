import pytest
from src.schemas.data_models import Requirement, NumericalValue
from src.agents.representation import KnowledgeRepresentationAgent

def test_representation_transformation():
    req = Requirement(
        requirement_id="REQ-001",
        original_text="The system shall encrypt data within 300 ms.",
        actor="The system",
        action="encrypt",
        object="data",
        conditions=["when at rest"],
        constraints=["AES-256"],
        numerical_values=[NumericalValue(value=300, unit="ms")],
        source_page=1
    )
    
    agent = KnowledgeRepresentationAgent()
    reps = agent.transform([req])
    
    assert len(reps) == 1
    rep = reps[0]
    assert rep.requirement_id == "REQ-001"
    assert rep.original_text == "The system shall encrypt data within 300 ms."
    
    # Check entities
    labels = [e.label for e in rep.entities]
    assert "Requirement" in labels
    assert "Actor" in labels
    assert "Action" in labels
    assert "Object" in labels
    assert "Condition" in labels
    assert "Constraint" in labels
    assert "NumericalValue" in labels
    
    # Check specific entity content
    num_entities = [e for e in rep.entities if e.label == "NumericalValue"]
    assert num_entities[0].name == "300.0 ms"
    
    # Check relations
    rel_types = [r.relation_type for r in rep.relations]
    assert "has_actor" in rel_types
    assert "performs" in rel_types
    assert "targets" in rel_types
    assert "has_condition" in rel_types
    assert "has_constraint" in rel_types
    assert "has_numerical_value" in rel_types

def test_missing_fields_representation():
    req = Requirement(
        requirement_id="REQ-002",
        original_text="Simple text",
        source_page=1
    )
    agent = KnowledgeRepresentationAgent()
    reps = agent.transform([req])
    
    assert len(reps) == 1
    rep = reps[0]
    assert len(rep.entities) == 1
    assert rep.entities[0].label == "Requirement"
    assert len(rep.relations) == 0

def test_missing_requirement_id_ignored():
    req = Requirement(
        requirement_id=None,
        original_text="No ID",
        source_page=1
    )
    agent = KnowledgeRepresentationAgent()
    reps = agent.transform([req])
    assert len(reps) == 0
