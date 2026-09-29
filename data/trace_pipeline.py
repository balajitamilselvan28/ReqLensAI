"""
Pipeline tracer: traces REQ-001 / REQ-002 through the full V5 pipeline
using the ACTUAL data formats Gemini returns (based on smoke test results).

Run from project root:
    python data/trace_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas.data_models import (
    Requirement, NumericalValue, CandidatePair, ReasoningResult
)
from src.agents.reasoning import LogicalReasoningAgent
from src.agents.contradiction import ContradictionDetectionAgent
from src.agents.dependency import DependencyVerificationAgent
from src.agents.recommendation import RecommendationGenerationAgent
from src.agents.verification import VerificationAgent

# ── Simulate EXACTLY what Gemini returns (based on previous smoke test) ──────
# The smoke test showed action = "shall register" — Gemini includes the modal.
# We test both variants to find the exact failure point.

def run_trace(label, r1_action, r2_action, r1_actor, r2_actor):
    print(f"\n{'='*60}")
    print(f"TRACE: {label}")
    print(f"  REQ-001 actor='{r1_actor}' action='{r1_action}'")
    print(f"  REQ-002 actor='{r2_actor}' action='{r2_action}'")
    print('='*60)

    req1 = Requirement(
        requirement_id="REQ-001",
        original_text="REQ-001: Students shall be allowed to register for public university events.",
        actor=r1_actor, action=r1_action, object="public university events",
        constraints=["Students are allowed to register"], conditions=[], source_page=1,
    )
    req2 = Requirement(
        requirement_id="REQ-002",
        original_text="REQ-002: Only administrators shall be allowed to register students for public university events.",
        actor=r2_actor, action=r2_action, object="students for public university events",
        constraints=["Only administrators are allowed to register"], conditions=[], source_page=1,
    )

    # Simulate retrieval: assume these two are a candidate pair (similarity ~0.90)
    pair = CandidatePair(
        requirement_id_1="REQ-001", requirement_id_2="REQ-002",
        similarity_score=0.90,
        requirement_text_1=req1.original_text,
        requirement_text_2=req2.original_text,
    )
    print(f"\n[Retrieval] CandidatePair: REQ-001 <-> REQ-002  sim={pair.similarity_score}")

    # Reasoning
    agent_reason = LogicalReasoningAgent()
    rr = agent_reason.evaluate_pair(req1, req2)
    print(f"\n[Reasoning]")
    print(f"  has_overlapping_actions : {rr.has_overlapping_actions}")
    print(f"  has_different_actors    : {rr.has_different_actors}")
    print(f"  has_numerical_overlap   : {rr.has_numerical_overlap}")
    print(f"  has_conditional_impl    : {rr.has_conditional_implication}")
    print(f"  details                 : {rr.reasoning_details}")

    # Contradiction detection
    agent_cont = ContradictionDetectionAgent()
    findings = agent_cont.detect(
        requirements=[req1, req2],
        candidate_pairs=[pair],
        reasoning_results=[rr],
    )
    print(f"\n[Contradiction Detection]  findings: {len(findings)}")
    for f in findings:
        print(f"  {f.finding_id}: {f.contradiction_type.value}  confidence={f.confidence_score}  status={f.status.value}")

    # Dependency verification
    if findings:
        agent_dep = DependencyVerificationAgent()
        updated_findings, dep_results = agent_dep.verify(findings)
        print(f"\n[Dependency Verification]  dep_results: {len(dep_results)}")
        for d in dep_results:
            print(f"  {d.requirement_id_1}<->{d.requirement_id_2}: {d.dependency_status.value}  conf={d.confidence_score}")
        print(f"  Updated finding status: {updated_findings[0].status.value}")

        # Recommendations
        agent_rec = RecommendationGenerationAgent()
        recs = agent_rec.generate(updated_findings, dep_results)
        print(f"\n[Recommendations]  count: {len(recs)}")
        for r in recs:
            print(f"  {r.finding_id}: {r.recommendation_text[:100]}...")

        # Verification
        agent_ver = VerificationAgent()
        ver_results = agent_ver.verify(recs, updated_findings)
        print(f"\n[Verification]  count: {len(ver_results)}")
        for v in ver_results:
            print(f"  {v.finding_id}: {v.verification_status.value}  conf={v.verification_confidence}")
    else:
        print("\n  *** PIPELINE STOPS HERE — no findings produced ***")

    return findings

# ── Test Case 1: Gemini returns "shall register" for both ────────────────────
findings_1 = run_trace(
    "Gemini returns same modal verb for both",
    r1_action="shall register", r2_action="shall register",
    r1_actor="Students", r2_actor="administrators",
)

# ── Test Case 2: Gemini returns different forms ──────────────────────────────
findings_2 = run_trace(
    "Gemini returns different modal: 'shall register' vs 'register'",
    r1_action="shall register", r2_action="register",
    r1_actor="Students", r2_actor="administrators",
)

# ── Test Case 3: Gemini returns clean actions without modal ──────────────────
findings_3 = run_trace(
    "Gemini strips modal: 'register' vs 'register'",
    r1_action="register", r2_action="register",
    r1_actor="Students", r2_actor="administrators",
)

print("\n" + "="*60)
print("SUMMARY")
print(f"  Case 1 (same modal)      → findings: {len(findings_1)}")
print(f"  Case 2 (different modal) → findings: {len(findings_2)}")
print(f"  Case 3 (no modal)        → findings: {len(findings_3)}")
print("="*60)
