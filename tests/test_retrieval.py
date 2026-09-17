import pytest
from src.schemas.data_models import Requirement
from src.retrieval.vector_store import RequirementVectorStore
from src.agents.retrieval import SemanticRetrievalAgent

class MockEmbeddingModel:
    """Mock embedding model that generates deterministic 384-d vectors without network calls."""
    def __init__(self):
        self.dim = 384
        
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]
        
    def embed_query(self, text):
        vec = [0.0] * self.dim
        text_lower = text.lower()
        
        # Simple mapping to force high similarity among related topics
        if "login" in text_lower:
            vec[0] = 1.0
            if "fast" in text_lower:
                vec[1] = 0.5
        elif "database" in text_lower:
            vec[2] = 1.0
        else:
            vec[3] = 1.0
            
        # Normalize slightly to make it somewhat realistic for cosine similarity
        norm = sum(x**2 for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
            
        return vec

import uuid

@pytest.fixture
def mock_embedding_model():
    return MockEmbeddingModel()

@pytest.fixture
def vector_store(mock_embedding_model):
    # Use ephemeral in-memory chromadb with unique collection name per test
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    return RequirementVectorStore(embedding_model=mock_embedding_model, collection_name=collection_name)

@pytest.fixture
def requirements():
    return [
        Requirement(requirement_id="REQ-001", original_text="The system shall allow users to login.", source_page=1),
        Requirement(requirement_id="REQ-002", original_text="The login process shall be fast.", source_page=1),
        Requirement(requirement_id="REQ-003", original_text="The system shall use a SQL database.", source_page=1),
        Requirement(requirement_id="REQ-004", original_text="Database backups must occur daily.", source_page=1),
        Requirement(requirement_id="REQ-005", original_text="The UI shall be blue.", source_page=2)
    ]

def test_embedding_generation(mock_embedding_model):
    vec = mock_embedding_model.embed_query("login")
    assert len(vec) == 384
    assert vec[0] > 0

def test_vector_store_insertion_and_search(vector_store, requirements):
    vector_store.add_requirements(requirements)
    
    # Query for database
    results = vector_store.search_similar("database system", top_k=2, threshold=0.1)
    assert len(results) == 2
    ids = [r["id"] for r in results]
    assert "REQ-003" in ids
    assert "REQ-004" in ids
    
    # Query for login
    results = vector_store.search_similar("user login", top_k=2, threshold=0.1)
    ids = [r["id"] for r in results]
    assert "REQ-001" in ids
    assert "REQ-002" in ids

def test_semantic_retrieval_agent(vector_store, requirements):
    agent = SemanticRetrievalAgent(vector_store, top_k=5, threshold=0.8)
    pairs = agent.generate_candidate_pairs(requirements)
    
    # Should find pairs for (REQ-001, REQ-002) and (REQ-003, REQ-004)
    assert len(pairs) == 2
    
    pair_ids = [(p.requirement_id_1, p.requirement_id_2) for p in pairs]
    
    # Assert self-match exclusion
    for p in pairs:
        assert p.requirement_id_1 != p.requirement_id_2
        
    # Assert duplicate pair prevention (only one (001,002) exists)
    assert ("REQ-001", "REQ-002") in pair_ids or ("REQ-002", "REQ-001") in pair_ids
    assert ("REQ-003", "REQ-004") in pair_ids or ("REQ-004", "REQ-003") in pair_ids

def test_empty_and_single_input(vector_store):
    agent = SemanticRetrievalAgent(vector_store, top_k=5, threshold=0.8)
    
    assert agent.generate_candidate_pairs([]) == []
    assert agent.generate_candidate_pairs([Requirement(requirement_id="REQ-001", original_text="single", source_page=1)]) == []

def test_metadata_preservation_and_pydantic_validation(vector_store):
    agent = SemanticRetrievalAgent(vector_store, top_k=2, threshold=0.5)
    reqs = [
        Requirement(requirement_id="REQ-001", original_text="login 1", source_page=1),
        Requirement(requirement_id="REQ-002", original_text="login 2", source_page=2)
    ]
    pairs = agent.generate_candidate_pairs(reqs)
    
    assert len(pairs) == 1
    pair = pairs[0]
    
    # Validate Pydantic schema structure
    assert hasattr(pair, "requirement_id_1")
    assert hasattr(pair, "requirement_id_2")
    assert hasattr(pair, "similarity_score")
    assert hasattr(pair, "requirement_text_1")
    assert hasattr(pair, "requirement_text_2")
    
    # Validate metadata content
    assert pair.requirement_text_1 == "login 1" or pair.requirement_text_1 == "login 2"

def test_duplicate_requirement_ids_handled_gracefully(vector_store):
    agent = SemanticRetrievalAgent(vector_store, top_k=2, threshold=0.5)
    reqs = [
        Requirement(requirement_id="REQ-001", original_text="login 1", source_page=1),
        Requirement(requirement_id="REQ-001", original_text="login 2", source_page=2)
    ]
    # In our implementation, chroma will overwrite REQ-001, so only 1 item exists in DB.
    # Therefore, no candidate pairs can be formed.
    pairs = agent.generate_candidate_pairs(reqs)
    assert len(pairs) == 0

def test_empty_text_handled_gracefully(vector_store):
    agent = SemanticRetrievalAgent(vector_store, top_k=2, threshold=0.5)
    reqs = [
        Requirement(requirement_id="REQ-001", original_text="  ", source_page=1),
        Requirement(requirement_id="REQ-002", original_text="login", source_page=2)
    ]
    pairs = agent.generate_candidate_pairs(reqs)
    assert len(pairs) == 0

def test_integration_flow(vector_store):
    # Simulated V2 structured requirements -> Semantic Retrieval Agent -> Candidate Pairs
    reqs = [
        Requirement(requirement_id="REQ-101", original_text="The system shall securely hash passwords in the database.", source_page=1),
        Requirement(requirement_id="REQ-102", original_text="All database passwords must be stored using bcrypt.", source_page=2),
        Requirement(requirement_id="REQ-103", original_text="The login screen must have a blue button.", source_page=3)
    ]
    agent = SemanticRetrievalAgent(vector_store, top_k=5, threshold=0.7)
    pairs = agent.generate_candidate_pairs(reqs)
    
    # REQ-101 and REQ-102 both contain 'database', so they will pair up.
    # REQ-103 contains 'login', so it's isolated.
    assert len(pairs) == 1
    p = pairs[0]
    assert set([p.requirement_id_1, p.requirement_id_2]) == {"REQ-101", "REQ-102"}
