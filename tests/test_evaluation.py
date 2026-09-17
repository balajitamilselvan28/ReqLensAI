import json
from pathlib import Path

import pytest

from evaluation.evaluate import (
    EvaluationResult,
    GroundTruthDataset,
    average_processing_time,
    contradiction_metrics,
    evaluate,
    load_ground_truth,
    recommendation_acceptance_rate,
    requirement_extraction_accuracy,
    semantic_retrieval_precision,
)
from src.schemas.data_models import (
    CandidatePair,
    ContradictionType,
    Finding,
    FindingStatus,
    HumanDecision,
    HumanFeedback,
    Requirement,
)


ROOT = Path(__file__).parents[1]


def make_requirement(requirement_id: str, text: str) -> Requirement:
    return Requirement(requirement_id=requirement_id, original_text=text, source_page=1)


def make_pair(id1: str, id2: str) -> CandidatePair:
    return CandidatePair(
        requirement_id_1=id1,
        requirement_id_2=id2,
        similarity_score=0.9,
        requirement_text_1=id1,
        requirement_text_2=id2,
    )


def make_finding(id1: str, id2: str, contradiction_type: ContradictionType) -> Finding:
    return Finding(
        finding_id=f"F-{id1}-{id2}",
        requirement_id_1=id1,
        requirement_id_2=id2,
        contradiction_type=contradiction_type,
        severity="high",
        confidence_score=0.8,
        explanation="Test finding",
        evidence="Test evidence",
        status=FindingStatus.VERIFIED,
    )


def make_dataset() -> GroundTruthDataset:
    return GroundTruthDataset(
        requirements=[
            {"requirement_id": "REQ-001", "original_text": "Students shall register."},
            {"requirement_id": "REQ-002", "original_text": "Admins shall register students."},
        ],
        relationships=[
            {
                "requirement_id_1": "REQ-001",
                "requirement_id_2": "REQ-002",
                "relationship_type": "contradiction",
                "contradiction_type": "permission_conflict",
            },
            {
                "requirement_id_1": "REQ-003",
                "requirement_id_2": "REQ-004",
                "relationship_type": "candidate_conflict",
            },
            {
                "requirement_id_1": "REQ-005",
                "requirement_id_2": "REQ-006",
                "relationship_type": "unrelated",
            },
        ],
    )


def test_ground_truth_file_loads():
    dataset = load_ground_truth(ROOT / "evaluation" / "ground_truth.json")
    assert len(dataset.requirements) == 8
    assert any(r.relationship_type == "unrelated" for r in dataset.relationships)
    assert any(r.relationship_type == "redundancy" for r in dataset.relationships)
    assert any(r.relationship_type == "dependency" for r in dataset.relationships)


def test_perfect_predictions():
    dataset = make_dataset()
    result = evaluate(
        dataset,
        [make_requirement("REQ-001", "Students shall register."), make_requirement("REQ-002", "Admins shall register students.")],
        [make_pair("REQ-001", "REQ-002"), make_pair("REQ-003", "REQ-004")],
        [make_finding("REQ-001", "REQ-002", ContradictionType.PERMISSION_CONFLICT)],
        [HumanFeedback(finding_id="F-1", decision=HumanDecision.APPROVE, original_recommendation="Review.")],
        [1.0, 3.0],
    )
    assert result.requirement_extraction_accuracy == 1.0
    assert result.semantic_retrieval_precision == 1.0
    assert result.contradiction_detection_accuracy == 1.0
    assert result.precision == result.recall == result.f1 == 1.0
    assert result.recommendation_acceptance_rate == 1.0
    assert result.average_processing_time == 2.0


def test_partial_predictions_and_false_positive():
    dataset = make_dataset()
    result = evaluate(
        dataset,
        [make_requirement("REQ-001", "Students shall register.")],
        [make_pair("REQ-001", "REQ-002"), make_pair("REQ-099", "REQ-100")],
        [
            make_finding("REQ-001", "REQ-002", ContradictionType.PERMISSION_CONFLICT),
            make_finding("REQ-099", "REQ-100", ContradictionType.REDUNDANCY),
        ],
    )
    assert result.requirement_extraction_accuracy == 0.5
    assert result.semantic_retrieval_precision == 0.5
    assert result.contradiction_detection_accuracy == 1.0
    assert result.precision == 0.5
    assert result.recall == 1.0
    assert result.f1 == pytest.approx(2 / 3)


def test_extraction_accuracy_penalizes_extra_predictions():
    dataset = make_dataset()
    predictions = [
        make_requirement("REQ-001", "Students shall register."),
        make_requirement("REQ-002", "Admins shall register students."),
        make_requirement("REQ-999", "An unreferenced requirement."),
    ]
    assert requirement_extraction_accuracy(predictions, dataset.requirements) == pytest.approx(2 / 3)


def test_no_predictions_are_zero():
    result = evaluate(make_dataset(), [], [], [])
    assert result.requirement_extraction_accuracy == 0.0
    assert result.semantic_retrieval_precision == 0.0
    assert result.contradiction_detection_accuracy == 0.0
    assert result.precision == result.recall == result.f1 == 0.0
    assert result.recommendation_acceptance_rate == 0.0
    assert result.average_processing_time == 0.0


def test_empty_ground_truth_is_safe():
    dataset = GroundTruthDataset()
    result = evaluate(dataset, [], [], [], [], [])
    assert isinstance(result, EvaluationResult)
    assert result.model_dump() == {
        "requirement_extraction_accuracy": 0.0,
        "semantic_retrieval_precision": 0.0,
        "contradiction_detection_accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "recommendation_acceptance_rate": 0.0,
        "average_processing_time": 0.0,
    }


def test_duplicate_predictions_do_not_inflate_metrics():
    dataset = make_dataset()
    pairs = [make_pair("REQ-001", "REQ-002"), make_pair("REQ-002", "REQ-001")]
    findings = [
        make_finding("REQ-001", "REQ-002", ContradictionType.PERMISSION_CONFLICT),
        make_finding("REQ-002", "REQ-001", ContradictionType.PERMISSION_CONFLICT),
    ]
    assert semantic_retrieval_precision(pairs, dataset.relationships) == 1.0
    assert contradiction_metrics(findings, dataset.relationships) == (1.0, 1.0, 1.0, 1.0)


def test_multiple_contradiction_types_are_distinct():
    dataset = GroundTruthDataset(
        relationships=[
            {
                "requirement_id_1": "REQ-001",
                "requirement_id_2": "REQ-002",
                "relationship_type": "contradiction",
                "contradiction_type": "permission_conflict",
            },
            {
                "requirement_id_1": "REQ-003",
                "requirement_id_2": "REQ-004",
                "relationship_type": "contradiction",
                "contradiction_type": "numerical_conflict",
            },
        ]
    )
    findings = [
        make_finding("REQ-001", "REQ-002", ContradictionType.PERMISSION_CONFLICT),
        make_finding("REQ-003", "REQ-004", ContradictionType.NUMERICAL_CONFLICT),
    ]
    assert contradiction_metrics(findings, dataset.relationships) == (1.0, 1.0, 1.0, 1.0)


def test_candidate_retrieval_evaluation_excludes_unrelated_pairs():
    dataset = make_dataset()
    pairs = [make_pair("REQ-003", "REQ-004"), make_pair("REQ-005", "REQ-006")]
    assert semantic_retrieval_precision(pairs, dataset.relationships) == 0.5


def test_recommendation_acceptance_rate_uses_human_approval():
    feedback = [
        HumanFeedback(finding_id="F-1", decision=HumanDecision.APPROVE, original_recommendation="A"),
        HumanFeedback(finding_id="F-2", decision=HumanDecision.MODIFY, original_recommendation="B"),
        HumanFeedback(finding_id="F-3", decision=HumanDecision.REJECT, original_recommendation="C"),
    ]
    assert recommendation_acceptance_rate(feedback) == pytest.approx(1 / 3)
    assert recommendation_acceptance_rate(feedback, total_recommendations=4) == 0.25


def test_duplicate_feedback_uses_latest_decision_once():
    feedback = [
        HumanFeedback(finding_id="F-1", decision=HumanDecision.APPROVE, original_recommendation="A"),
        HumanFeedback(finding_id="F-1", decision=HumanDecision.REJECT, original_recommendation="A"),
        HumanFeedback(finding_id="F-2", decision=HumanDecision.APPROVE, original_recommendation="B"),
    ]
    assert recommendation_acceptance_rate(feedback) == 0.5


def test_processing_time_average_and_validation():
    assert average_processing_time([1.0, 2.0, 4.0]) == pytest.approx(7 / 3)
    with pytest.raises(ValueError):
        average_processing_time([-1.0])
    with pytest.raises(ValueError):
        average_processing_time([float("nan")])


def test_precision_recall_and_f1_with_false_negative():
    dataset = make_dataset()
    findings = [make_finding("REQ-001", "REQ-002", ContradictionType.PERMISSION_CONFLICT)]
    cda, precision, recall, f1 = contradiction_metrics(findings, [
        *dataset.relationships,
        {
            "requirement_id_1": "REQ-007",
            "requirement_id_2": "REQ-008",
            "relationship_type": "contradiction",
            "contradiction_type": "numerical_conflict",
        },
    ])
    assert cda == 0.5
    assert precision == 1.0
    assert recall == 0.5
    assert f1 == pytest.approx(2 / 3)


def test_loader_rejects_invalid_json_shape(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"requirements": [{"requirement_id": "REQ-1"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(path)
