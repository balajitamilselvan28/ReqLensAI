"""
End-to-end pipeline test using the real SRS PDF and Gemini.
Run from project root: python data/e2e_srs_test.py
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)  # suppress info noise

from dotenv import load_dotenv
load_dotenv()

from src.utils.pdf_parser import parse_pdf

PDF_PATH = os.path.join(os.path.dirname(__file__),
                        "ReqLens_SRS_Inconsistency_Recommendation_Test.pdf")

print("=== ReqLens AI End-to-End Pipeline Test ===\n")

# Stage 1: PDF parsing
pages = parse_pdf(PDF_PATH)
srs_text = "\n\n".join(p["text"] for p in pages if p["text"])
print(f"[PDF]    Pages parsed: {len(pages)}")
print(f"         SRS text length: {len(srs_text)} chars")

# Stage 2: Gemini extraction
api_key = os.getenv("GOOGLE_API_KEY", "")
print(f"\n[Extractor] Gemini API key present: {'YES' if api_key else 'NO'}")

from src.agents.extractor import RequirementExtractionAgent
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, google_api_key=api_key)
extractor = RequirementExtractionAgent(llm=llm)
pages_for_extractor = [{"page_number": i+1, "text": p["text"]} for i, p in enumerate(pages)]
requirements = extractor.extract(pages_for_extractor)

print(f"[Extractor] Requirements extracted: {len(requirements)}")
# Show key requirements
key_ids = {"REQ-001", "REQ-002", "REQ-017", "REQ-018", "REQ-031", "REQ-032"}
found_key = {r.requirement_id: r for r in requirements if r.requirement_id in key_ids}
for rid in sorted(key_ids):
    req = found_key.get(rid)
    if req:
        print(f"  {rid}: actor='{req.actor}' action='{req.action}' object='{req.object}'")
    else:
        print(f"  {rid}: NOT FOUND in extracted requirements")

# Field population stats
actors_set = sum(1 for r in requirements if r.actor)
actions_set = sum(1 for r in requirements if r.action)
objects_set = sum(1 for r in requirements if r.object)
numericals_set = sum(1 for r in requirements if r.numerical_values)
print(f"\n  Semantic field coverage (out of {len(requirements)}):")
print(f"    actor populated     : {actors_set}")
print(f"    action populated    : {actions_set}")
print(f"    object populated    : {objects_set}")
print(f"    numerical_values    : {numericals_set}")

if len(requirements) < 2:
    print("\nInsufficient requirements for further pipeline stages. Exiting.")
    sys.exit(1)

# Stage 3: Semantic retrieval
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import RequirementVectorStore
from src.agents.retrieval import SemanticRetrievalAgent

vector_store = RequirementVectorStore(embedding_model=EmbeddingModel("all-MiniLM-L6-v2"))
retriever = SemanticRetrievalAgent(vector_store=vector_store, top_k=5, threshold=0.70)
candidate_pairs = retriever.generate_candidate_pairs(requirements)
print(f"\n[Retrieval] Candidate pairs: {len(candidate_pairs)}")
for cp in candidate_pairs[:5]:
    print(f"  {cp.requirement_id_1} <-> {cp.requirement_id_2}  sim={cp.similarity_score:.3f}")

# Stage 4: Knowledge representation
from src.agents.representation import KnowledgeRepresentationAgent
kg_agent = KnowledgeRepresentationAgent()
knowledge_reps = kg_agent.transform(requirements)
print(f"\n[Knowledge] Representations: {len(knowledge_reps)}")

# Stage 5: Logical reasoning
from src.agents.reasoning import LogicalReasoningAgent
reasoner = LogicalReasoningAgent()
reasoning_results = reasoner.evaluate_candidates(requirements, candidate_pairs)
print(f"\n[Reasoning] Results: {len(reasoning_results)}")
perm_conflicts = [r for r in reasoning_results if r.has_overlapping_actions and r.has_different_actors]
print(f"  Permission conflict candidates: {len(perm_conflicts)}")
for r in perm_conflicts[:3]:
    print(f"    {r.requirement_id_1}<->{r.requirement_id_2}: overlapping={r.has_overlapping_actions} diff_actors={r.has_different_actors}")

# Stage 6: Contradiction detection
from src.agents.contradiction import ContradictionDetectionAgent
detector = ContradictionDetectionAgent()
knowledge_map = {kr.requirement_id: kr for kr in knowledge_reps}
findings = detector.detect(
    requirements=requirements,
    candidate_pairs=candidate_pairs,
    reasoning_results=reasoning_results,
    knowledge_map=knowledge_map,
)
print(f"\n[Contradiction] Findings: {len(findings)}")
for f in findings:
    print(f"  {f.finding_id}: {f.contradiction_type.value}  severity={f.severity.value}  conf={f.confidence_score}  status={f.status.value}")
    print(f"    {f.requirement_id_1} <-> {f.requirement_id_2}")

# Stage 7: Dependency verification
from src.agents.dependency import DependencyVerificationAgent
dep_agent = DependencyVerificationAgent()
updated_findings, dep_results = dep_agent.verify(findings)
print(f"\n[Dependency] Results: {len(dep_results)}")
for d in dep_results:
    print(f"  {d.requirement_id_1}<->{d.requirement_id_2}: {d.dependency_status.value}  conf={d.confidence_score}")

# Stage 8: Recommendations
from src.agents.recommendation import RecommendationGenerationAgent
rec_agent = RecommendationGenerationAgent()
recommendations = rec_agent.generate(updated_findings, dep_results)
print(f"\n[Recommendations] Count: {len(recommendations)}")
for rec in recommendations:
    print(f"  {rec.finding_id}: {rec.recommendation_text[:80]}...")

# Stage 9: Verification
from src.agents.verification import VerificationAgent
ver_agent = VerificationAgent()
ver_results = ver_agent.verify(recommendations, updated_findings)
print(f"\n[Verification] Count: {len(ver_results)}")
for v in ver_results:
    print(f"  {v.finding_id}: {v.verification_status.value}  conf={v.verification_confidence}  human_review={v.requires_human_review}")

print("\n=== PIPELINE COMPLETE ===")
print(f"Requirements: {len(requirements)}")
print(f"Candidate Pairs: {len(candidate_pairs)}")
print(f"Knowledge Reps: {len(knowledge_reps)}")
print(f"Reasoning Results: {len(reasoning_results)}")
print(f"Findings: {len(findings)}")
print(f"Dependency Results: {len(dep_results)}")
print(f"Recommendations: {len(recommendations)}")
print(f"Verification Results: {len(ver_results)}")
