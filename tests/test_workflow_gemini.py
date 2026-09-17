import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.graph.workflow import node_extract
from src.schemas.data_models import ExtractedRequirements, Requirement


def make_state(content):
    return {"srs_content": content}


def test_node_extract_uses_gemini_v2_extractor_and_preserves_pages():
    first_page = ExtractedRequirements(requirements=[
        Requirement(
            requirement_id="REQ-001",
            original_text="Students shall register on page one.",
            actor="Students",
            action="register",
            object="events",
            source_page=99,
        )
    ])
    second_page = ExtractedRequirements(requirements=[
        Requirement(
            requirement_id="REQ-002",
            original_text="Administrators shall manage events on page two.",
            actor="Administrators",
            action="manage",
            object="events",
            source_page=99,
        )
    ])
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = [first_page, second_page]

    fake_gemini = MagicMock(return_value=mock_llm)
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}), \
         patch.dict(sys.modules, {"langchain_google_genai": SimpleNamespace(ChatGoogleGenerativeAI=fake_gemini)}), \
         patch("src.graph.workflow.load_dotenv", create=True):
        result = node_extract(make_state("page one\n\npage two"))

    fake_gemini.assert_called_once_with(
        model="gemini-3.8-flash", temperature=0, google_api_key="test-key"
    )
    assert len(result["extracted_requirements"]) == 2
    assert result["extracted_requirements"][0].actor == "Students"
    assert result["extracted_requirements"][0].source_page == 1
    assert result["extracted_requirements"][1].source_page == 2


def test_node_extract_falls_back_without_google_api_key():
    with patch.dict("os.environ", {}, clear=True), \
         patch("src.graph.workflow.load_dotenv", create=True), \
         patch("src.graph.workflow.RequirementExtractionAgent") as extractor:
        result = node_extract(make_state("The system shall log in."))

    extractor.assert_not_called()
    assert result["extracted_requirements"][0].requirement_id == "REQ-001"
    assert result["extracted_requirements"][0].actor is None


def test_node_extract_falls_back_when_gemini_fails():
    fake_gemini = MagicMock(side_effect=RuntimeError("offline"))
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}), \
         patch("src.graph.workflow.load_dotenv", create=True), \
         patch.dict(sys.modules, {"langchain_google_genai": SimpleNamespace(ChatGoogleGenerativeAI=fake_gemini)}):
        result = node_extract(make_state("The system shall log in."))

    assert len(result["extracted_requirements"]) == 1
    assert result["extracted_requirements"][0].original_text == "The system shall log in."