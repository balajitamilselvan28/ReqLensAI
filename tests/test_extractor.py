import pytest
from unittest.mock import MagicMock
from src.schemas.data_models import Requirement, ExtractedRequirements, NumericalValue
from src.agents.extractor import RequirementExtractionAgent
from src.utils.pdf_parser import parse_pdf
import pymupdf as fitz

def get_mock_llm(return_value=None, side_effect=None):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    if side_effect:
        mock_structured.invoke.side_effect = side_effect
    else:
        mock_structured.invoke.return_value = return_value
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm

def test_successful_extraction_and_id_preservation():
    mock_llm = get_mock_llm(return_value=ExtractedRequirements(
        requirements=[
            Requirement(
                requirement_id="SYS-001",
                original_text="The system shall encrypt data.",
                actor="The system",
                action="encrypt",
                object="data",
                source_page=1
            )
        ]
    ))
    
    agent = RequirementExtractionAgent(llm=mock_llm)
    pages = [{"page_number": 1, "text": "The system shall encrypt data. [SYS-001]"}]
    
    results = agent.extract(pages)
    
    assert len(results) == 1
    assert results[0].requirement_id == "SYS-001"
    assert results[0].actor == "The system"
    assert results[0].action == "encrypt"
    assert results[0].object == "data"

def test_generated_ids_when_missing():
    mock_llm = get_mock_llm(return_value=ExtractedRequirements(
        requirements=[
            Requirement(original_text="Requirement 1", source_page=1),
            Requirement(original_text="Requirement 2", source_page=1)
        ]
    ))
    
    agent = RequirementExtractionAgent(llm=mock_llm)
    pages = [{"page_number": 1, "text": "Requirement 1. Requirement 2."}]
    
    results = agent.extract(pages)
    
    assert len(results) == 2
    assert results[0].requirement_id == "REQ-001"
    assert results[1].requirement_id == "REQ-002"

def test_numerical_value_and_condition_extraction():
    mock_llm = get_mock_llm(return_value=ExtractedRequirements(
        requirements=[
            Requirement(
                original_text="The system shall lock the account after 5 failed login attempts.",
                conditions=["after 5 failed login attempts"],
                numerical_values=[NumericalValue(value=5, unit="attempts")],
                source_page=1
            )
        ]
    ))
    
    agent = RequirementExtractionAgent(llm=mock_llm)
    pages = [{"page_number": 1, "text": "..."}]
    
    results = agent.extract(pages)
    
    assert len(results) == 1
    assert len(results[0].conditions) == 1
    assert "failed login attempts" in results[0].conditions[0]
    assert len(results[0].numerical_values) == 1
    assert results[0].numerical_values[0].value == 5
    assert results[0].numerical_values[0].unit == "attempts"


def test_semantic_fields_are_preserved_from_structured_output():
    mock_llm = get_mock_llm(return_value=ExtractedRequirements(
        requirements=[
            Requirement(
                requirement_id="REQ-001",
                original_text="Students shall register for public university events.",
                actor="Students",
                action="register",
                object="public university events",
                constraints=["Students are allowed to register"],
                source_page=1,
            ),
            Requirement(
                requirement_id="REQ-002",
                original_text="Only administrators shall register students for public university events.",
                actor="administrators",
                action="register",
                object="students for public university events",
                constraints=["Only administrators are allowed to register"],
                source_page=1,
            ),
            Requirement(
                requirement_id="REQ-031",
                original_text="The system shall respond to user requests within 300 ms.",
                actor="The system",
                action="respond",
                object="user requests",
                numerical_values=[NumericalValue(value=300, unit="ms")],
                source_page=1,
            ),
        ]
    ))

    results = RequirementExtractionAgent(llm=mock_llm).extract([
        {"page_number": 1, "text": "semantic examples"}
    ])

    assert results[0].actor == "Students"
    assert results[0].action == "register"
    assert results[0].object == "public university events"
    assert results[1].actor == "administrators"
    assert results[1].action == "register"
    assert results[2].actor == "The system"
    assert results[2].action == "respond"
    assert results[2].numerical_values[0].value == 300
    assert results[2].numerical_values[0].unit == "ms"

def test_missing_optional_fields():
    mock_llm = get_mock_llm(return_value=ExtractedRequirements(
        requirements=[
            Requirement(
                original_text="Simple requirement.",
                source_page=1
            )
        ]
    ))
    
    agent = RequirementExtractionAgent(llm=mock_llm)
    pages = [{"page_number": 1, "text": "..."}]
    
    results = agent.extract(pages)
    
    assert len(results) == 1
    assert results[0].actor is None
    assert results[0].priority is None
    assert results[0].constraints == []

def test_malformed_llm_output():
    # Return something that doesn't match ExtractedRequirements
    mock_llm = get_mock_llm(return_value={"malformed": "data"})
    
    agent = RequirementExtractionAgent(llm=mock_llm)
    pages = [{"page_number": 1, "text": "..."}]
    
    results = agent.extract(pages)
    
    assert len(results) == 0  # Should handle gracefully and return empty list

def test_llm_api_failure():
    mock_llm = get_mock_llm(side_effect=Exception("API Timeout"))
    
    agent = RequirementExtractionAgent(llm=mock_llm)
    pages = [{"page_number": 1, "text": "..."}]
    
    results = agent.extract(pages)
    
    assert len(results) == 0  # Should catch Exception and continue

def test_integration_with_pdf_parser(tmp_path):
    # 1. Create a dummy PDF
    pdf_path = tmp_path / "test_srs.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 50), "The system shall log errors.")
    doc.save(str(pdf_path))
    doc.close()
    
    # 2. Parse PDF
    parsed_pages = parse_pdf(str(pdf_path))
    
    # 3. Mock LLM and extract
    mock_llm = get_mock_llm(return_value=ExtractedRequirements(
        requirements=[
            Requirement(
                original_text="The system shall log errors.",
                actor="The system",
                action="log",
                object="errors",
                source_page=1
            )
        ]
    ))
    
    agent = RequirementExtractionAgent(llm=mock_llm)
    results = agent.extract(parsed_pages)
    
    assert len(results) == 1
    assert results[0].actor == "The system"
    assert results[0].action == "log"
    assert results[0].source_page == 1
