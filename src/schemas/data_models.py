from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Version 2 Schemas: Requirements
# ---------------------------------------------------------

class NumericalValue(BaseModel):
    value: float = Field(..., description="The numerical value extracted")
    unit: Optional[str] = Field(None, description="The unit of the numerical value (e.g., milliseconds, users)")

class Requirement(BaseModel):
    """Represents a structured software requirement."""
    requirement_id: Optional[str] = Field(None, description="Unique identifier explicitly found in the text (e.g., REQ-001). Leave null if none is explicitly stated.")
    original_text: str = Field(..., description="The exact original text of the requirement. Do not paraphrase.")
    actor: Optional[str] = Field(None, description="The entity performing the action (e.g., 'The system', 'The user').")
    action: Optional[str] = Field(None, description="The action being performed (e.g., 'shall respond', 'must encrypt').")
    object: Optional[str] = Field(None, description="The entity being acted upon.")
    constraints: List[str] = Field(default_factory=list, description="List of constraints on the requirement.")
    conditions: List[str] = Field(default_factory=list, description="Conditions under which the requirement applies.")
    priority: Optional[str] = Field(None, description="Priority explicitly stated in the document (e.g., High, Low). Do not invent a priority.")
    numerical_values: List[NumericalValue] = Field(default_factory=list, description="Structured numerical values with units.")
    source_page: int = Field(..., description="The 1-indexed page number where this requirement was found.")

class ExtractedRequirements(BaseModel):
    """A collection of extracted requirements from a page or chunk."""
    requirements: List[Requirement] = Field(..., description="List of requirements extracted from the text.")

# ---------------------------------------------------------
# Version 3 Schemas: Semantic Retrieval
# ---------------------------------------------------------

class CandidatePair(BaseModel):
    """Represents a pair of semantically related requirements."""
    requirement_id_1: str = Field(..., description="ID of the first requirement")
    requirement_id_2: str = Field(..., description="ID of the second requirement")
    similarity_score: float = Field(..., description="Semantic similarity score between the two requirements")
    requirement_text_1: str = Field(..., description="Original text of the first requirement")
    requirement_text_2: str = Field(..., description="Original text of the second requirement")

# ---------------------------------------------------------
# Version 4 Schemas: Knowledge Representation
# ---------------------------------------------------------

class Entity(BaseModel):
    """Represents a node in the knowledge graph."""
    id: str = Field(..., description="Unique ID for the entity")
    label: str = Field(..., description="The type/label (e.g., Actor, Action, Object, Condition)")
    name: str = Field(..., description="The textual representation of the entity")

class Relation(BaseModel):
    """Represents an edge in the knowledge graph."""
    source_id: str = Field(..., description="ID of the source entity")
    target_id: str = Field(..., description="ID of the target entity")
    relation_type: str = Field(..., description="Type of relationship (e.g., performs, targets, has_condition)")

class KnowledgeRepresentation(BaseModel):
    """Represents the structured knowledge extracted from a requirement."""
    requirement_id: str
    original_text: str
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)

# ---------------------------------------------------------
# Version 4 Schemas: Logical Reasoning
# ---------------------------------------------------------

class ReasoningResult(BaseModel):
    """Represents logical relationship facts between two requirements."""
    requirement_id_1: str
    requirement_id_2: str
    has_overlapping_actions: bool = False
    has_different_actors: bool = False
    has_numerical_overlap: bool = False
    numerical_comparison: Optional[str] = Field(None, description="E.g., '300 vs 500 ms'")
    has_conditional_implication: bool = False
    reasoning_details: str = Field(..., description="Explanation of the logical relationship found.")

# ---------------------------------------------------------
# Version 5 Schemas: Contradiction Detection
# ---------------------------------------------------------

class ContradictionType(str, Enum):
    PERMISSION_CONFLICT = "permission_conflict"
    NUMERICAL_CONFLICT = "numerical_conflict"
    CONDITIONAL_CONFLICT = "conditional_conflict"
    DIRECT_CONFLICT = "direct_conflict"
    REDUNDANCY = "redundancy"
    LOGICAL_CONFLICT = "logical_conflict"
    UNKNOWN = "unknown"

class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FindingStatus(str, Enum):
    POTENTIAL = "potential"
    VERIFIED = "verified"
    REJECTED = "rejected"

class Finding(BaseModel):
    """Represents a detected inconsistency or potential contradiction."""
    finding_id: str = Field(..., description="Unique ID for this finding")
    requirement_id_1: str
    requirement_id_2: str
    contradiction_type: ContradictionType
    severity: Severity
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., description="Human-readable explanation for a requirements analyst")
    evidence: str = Field(..., description="Supporting evidence, including original requirement texts")
    status: FindingStatus = FindingStatus.POTENTIAL

# ---------------------------------------------------------
# Version 5 Schemas: Dependency Verification
# ---------------------------------------------------------

class DependencyStatus(str, Enum):
    SUPPORTED = "supported"
    CONDITIONAL = "conditional"
    UNRELATED = "unrelated"
    INSUFFICIENT_INFORMATION = "insufficient_information"

class DependencyResult(BaseModel):
    """Represents the result of dependency verification for a finding."""
    requirement_id_1: str
    requirement_id_2: str
    dependency_status: DependencyStatus
    explanation: str
    supporting_evidence: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)

# ---------------------------------------------------------
# Version 6 Schemas: Recommendation Generation
# ---------------------------------------------------------

class Recommendation(BaseModel):
    """A context-aware recommendation generated for a verified finding."""
    finding_id: str
    requirement_id_1: str
    requirement_id_2: str
    contradiction_type: ContradictionType
    severity: Severity
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., description="Original finding explanation")
    recommendation_text: str = Field(..., description="Actionable recommendation for the requirements engineer")
    evidence: str = Field(..., description="Original requirement texts and evidence")
    used_llm: bool = Field(False, description="Whether an LLM was used to generate this recommendation")

# ---------------------------------------------------------
# Version 6 Schemas: Verification Agent
# ---------------------------------------------------------

class VerificationStatus(str, Enum):
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"

class VerificationResult(BaseModel):
    """Result of automated verification of a recommendation before human review."""
    finding_id: str
    verification_status: VerificationStatus
    verification_confidence: float = Field(..., ge=0.0, le=1.0)
    verified_claims: List[str] = Field(default_factory=list, description="Claims that passed verification")
    warnings: List[str] = Field(default_factory=list, description="Concerns or ambiguities")
    final_recommendation: str = Field(..., description="Recommendation text to present to the human analyst")
    requires_human_review: bool = True

# ---------------------------------------------------------
# Version 6 Schemas: Human-in-the-Loop Feedback
# ---------------------------------------------------------

class HumanDecision(str, Enum):
    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"

class HumanFeedback(BaseModel):
    """Represents structured human analyst feedback on a recommendation."""
    finding_id: str
    decision: HumanDecision
    analyst_comment: Optional[str] = Field(None, description="Optional reasoning from the analyst")
    modified_recommendation: Optional[str] = Field(
        None, description="Analyst-modified recommendation text. Stored separately; does not overwrite the original."
    )
    original_recommendation: str = Field(..., description="Original machine-generated recommendation (preserved immutably)")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp of the decision")

# ---------------------------------------------------------
# Legacy compatibility alias
# ---------------------------------------------------------

class Rule(BaseModel):
    """Represents a logical rule or constraint inferred from a requirement."""
    rule_id: str
    source_req_id: str
    condition: str
    action: str

