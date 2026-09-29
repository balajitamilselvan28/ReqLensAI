import logging
from typing import List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from src.schemas.data_models import Requirement, ExtractedRequirements

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """
You are an expert Requirements Engineer analyzing a Software Requirements Specification (SRS) document.
Your task is to extract individual software requirements from the provided text and structure them according to the schema.

RULES:
- Do not invent requirements.
- Do not add information not present in the source.
- Preserve original requirement text exactly as found in the source. Do not paraphrase or rewrite.
- Extract only information supported by the SRS.
- Return structured data matching the Pydantic schema.
- Populate every applicable semantic field, not only requirement_id and original_text.
- Actor: identify who performs, requests, is permitted or required to perform, or is responsible for the action.
- Action: identify the main action or core verb, such as register, send, cancel, or respond.
- Object: identify what the action applies to.
- Constraints: extract explicit restrictions, permissions, prohibitions, limits, capacities, timing, and performance requirements.
- Conditions: extract explicit circumstances under which the requirement applies, such as after approval, before a deadline, if payment succeeds, or during normal operating conditions.
- If a requirement explicitly states a priority (e.g., High, Low), extract it. Do not invent a priority.
- Extract every explicit numerical value with its unit or meaning.
- Use null for actor, action, or object only when that information is genuinely absent from the source; do not leave them null merely because the wording is complex.
- Use empty lists only when there are no explicit constraints, conditions, or numerical values.
- Do not hallucinate unsupported information.
- A requirement might have an explicit ID (e.g., REQ-001). Extract it into requirement_id if present.
- Differentiate between constraints and conditions where possible.

Examples:

Requirement: "REQ-001: Students shall be allowed to register for public university events."
Expected semantic extraction:
- requirement_id = "REQ-001"
- actor = "Students"
- action = "register"
- object = "public university events"
- constraints = ["Students are allowed to register"]
- conditions = []
- priority = null
- numerical_values = []

Requirement: "REQ-002: Only administrators shall be allowed to register students for public university events."
Expected semantic extraction:
- requirement_id = "REQ-002"
- actor = "administrators"
- action = "register"
- object = "students for public university events"
- constraints = ["Only administrators are allowed to register"]
- conditions = []
- priority = null
- numerical_values = []

Requirement: "REQ-031: The system shall respond to user requests within 300 ms."
Expected semantic extraction:
- requirement_id = "REQ-031"
- actor = "The system"
- action = "respond"
- object = "user requests"
- constraints should capture the response-time requirement
- conditions = []
- priority = null
- numerical_values = [{{"value": 300, "unit": "ms"}}]

Source Page: {page_number}
Text to Analyze:
{text}
"""

class RequirementExtractionAgent:
    def __init__(self, llm: BaseChatModel):
        """
        Initializes the Requirement Extraction Agent.
        
        Args:
            llm: A LangChain chat model that supports structured output.
        """
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", EXTRACTION_PROMPT)
        ])
        self.structured_llm = self.llm.with_structured_output(ExtractedRequirements)

    def extract(self, pages: List[Dict[str, Any]]) -> List[Requirement]:
        """
        Extracts structured requirements from parsed PDF pages.
        
        Args:
            pages: A list of dictionaries, each containing 'page_number' and 'text'.
            
        Returns:
            A list of structured Requirement objects.
        """
        all_requirements = []
        generated_id_counter = 1
        
        for page in pages:
            page_num = page.get("page_number", 1)
            text = page.get("text", "")
            
            if not text.strip():
                continue
                
            try:
                # Format prompt and invoke LLM
                prompt_value = self.prompt.format_messages(page_number=page_num, text=text)
                result: ExtractedRequirements = self.structured_llm.invoke(prompt_value)
                
                if result and hasattr(result, 'requirements') and result.requirements:
                    for req in result.requirements:
                        # Ensure source_page matches the current page
                        req.source_page = page_num
                        
                        # Generate deterministic ID if missing
                        if not req.requirement_id:
                            req.requirement_id = f"REQ-{generated_id_counter:03d}"
                            generated_id_counter += 1
                        
                        all_requirements.append(req)
                        
            except Exception as e:
                logger.error(f"Failed to extract requirements from page {page_num}. Error: {e}")
                # Continue gracefully on LLM failure or parsing failure
                continue
                
        return all_requirements
