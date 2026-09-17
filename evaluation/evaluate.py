"""Deterministic research evaluation for ReqLens AI V7.3.

This module evaluates supplied predictions against a manually maintained ground-truth
reference. It does not execute the production workflow or call an LLM.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from pydantic import BaseModel, Field

from src.schemas.data_models import (
    CandidatePair,
    ContradictionType,
    Finding,
    HumanDecision,
    HumanFeedback,
    Requirement,
)


class GroundTruthRequirement(BaseModel):
    requirement_id: str
    original_text: str


class GroundTruthRelationship(BaseModel):
    requirement_id_1: str
    requirement_id_2: str
    relationship_type: str
    contradiction_type: Optional[ContradictionType] = None


class GroundTruthDataset(BaseModel):
    requirements: list[GroundTruthRequirement] = Field(default_factory=list)
    relationships: list[GroundTruthRelationship] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """The eight V7.3 research metrics for one evaluation run."""

    requirement_extraction_accuracy: float = Field(..., ge=0.0, le=1.0)
    semantic_retrieval_precision: float = Field(..., ge=0.0, le=1.0)
    contradiction_detection_accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    recommendation_acceptance_rate: float = Field(..., ge=0.0, le=1.0)
    average_processing_time: float = Field(..., ge=0.0)


def load_ground_truth(path: str | Path) -> GroundTruthDataset:
    """Load and validate a human-readable JSON ground-truth dataset."""
    with Path(path).open(encoding="utf-8") as handle:
        return GroundTruthDataset.model_validate(json.load(handle))


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _normalise_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _canonical_pair(id1: str, id2: str) -> tuple[str, str]:
    return tuple(sorted((id1, id2)))


def _candidate_pairs(items: Iterable[CandidatePair | dict[str, Any]]) -> set[tuple[str, str]]:
    pairs = set()
    for item in items:
        id1 = _value(item, "requirement_id_1")
        id2 = _value(item, "requirement_id_2")
        if id1 and id2:
            pairs.add(_canonical_pair(id1, id2))
    return pairs


def _finding_signatures(items: Iterable[Finding | dict[str, Any]]) -> set[tuple[str, str, str]]:
    signatures = set()
    for item in items:
        id1 = _value(item, "requirement_id_1")
        id2 = _value(item, "requirement_id_2")
        contradiction_type = _value(item, "contradiction_type")
        if id1 and id2 and contradiction_type:
            type_value = getattr(contradiction_type, "value", contradiction_type)
            signatures.add((*_canonical_pair(id1, id2), str(type_value)))
    return signatures


def requirement_extraction_accuracy(
    extracted_requirements: Sequence[Requirement | dict[str, Any]],
    ground_truth_requirements: Sequence[GroundTruthRequirement],
) -> float:
    """Return exact matches divided by the larger prediction/reference set."""
    expected = {
        (item.requirement_id, _normalise_text(item.original_text))
        for item in ground_truth_requirements
    }
    predicted = {
        (_value(item, "requirement_id"), _normalise_text(_value(item, "original_text") or ""))
        for item in extracted_requirements
        if _value(item, "requirement_id")
    }
    return _safe_ratio(len(expected & predicted), max(len(expected), len(predicted)))


def semantic_retrieval_precision(
    candidate_pairs: Sequence[CandidatePair | dict[str, Any]],
    ground_truth_relationships: Sequence[GroundTruthRelationship],
) -> float:
    """Return relevant retrieved pairs divided by all unique retrieved pairs.

    Every ground-truth relationship except ``unrelated`` is relevant for retrieval;
    relationship categories remain distinct for contradiction evaluation.
    """
    expected = {
        _canonical_pair(_value(item, "requirement_id_1"), _value(item, "requirement_id_2"))
        for item in ground_truth_relationships
        if _value(item, "relationship_type") != "unrelated"
    }
    predicted = _candidate_pairs(candidate_pairs)
    return _safe_ratio(len(predicted & expected), len(predicted))


def contradiction_metrics(
    findings: Sequence[Finding | dict[str, Any]],
    ground_truth_relationships: Sequence[GroundTruthRelationship],
) -> tuple[float, float, float, float]:
    """Return CDA, precision, recall, and F1 for exact pair/type findings.

    CDA is the proportion of known contradiction pair/type labels detected. Precision,
    recall, and F1 separately expose false positives and false negatives.
    Duplicate predictions are counted once.
    """
    expected = {
        (
            *_canonical_pair(_value(item, "requirement_id_1"), _value(item, "requirement_id_2")),
            getattr(_value(item, "contradiction_type"), "value", _value(item, "contradiction_type")),
        )
        for item in ground_truth_relationships
        if _value(item, "relationship_type") == "contradiction"
        and _value(item, "contradiction_type")
    }
    predicted = _finding_signatures(findings)
    true_positives = len(expected & predicted)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    cda = _safe_ratio(true_positives, len(expected))
    return cda, precision, recall, f1


def recommendation_acceptance_rate(
    feedback: Sequence[HumanFeedback | dict[str, Any]],
    total_recommendations: Optional[int] = None,
) -> float:
    """Return approved recommendations divided by evaluated recommendations."""
    latest_feedback = {}
    for item in feedback:
        finding_id = _value(item, "finding_id")
        if finding_id:
            latest_feedback[finding_id] = item

    denominator = len(latest_feedback) if total_recommendations is None else total_recommendations
    if denominator < 0:
        raise ValueError("total_recommendations cannot be negative")
    accepted = sum(
        1
        for item in latest_feedback.values()
        if (_value(item, "decision") == HumanDecision.APPROVE)
        or (_value(item, "decision") == HumanDecision.APPROVE.value)
    )
    if accepted > denominator:
        raise ValueError("accepted recommendations cannot exceed total recommendations")
    return _safe_ratio(accepted, denominator)


def average_processing_time(processing_times: Sequence[float]) -> float:
    """Return the arithmetic mean of measured non-negative processing times."""
    if any(not math.isfinite(value) or value < 0 for value in processing_times):
        raise ValueError("processing times must be finite and non-negative")
    return _safe_ratio(sum(processing_times), len(processing_times))


def evaluate(
    dataset: GroundTruthDataset,
    extracted_requirements: Sequence[Requirement | dict[str, Any]],
    candidate_pairs: Sequence[CandidatePair | dict[str, Any]],
    findings: Sequence[Finding | dict[str, Any]],
    human_feedback: Sequence[HumanFeedback | dict[str, Any]] = (),
    processing_times: Sequence[float] = (),
    total_recommendations: Optional[int] = None,
) -> EvaluationResult:
    """Calculate all V7.3 metrics from supplied predictions and measurements."""
    cda, precision, recall, f1 = contradiction_metrics(findings, dataset.relationships)
    return EvaluationResult(
        requirement_extraction_accuracy=requirement_extraction_accuracy(
            extracted_requirements, dataset.requirements
        ),
        semantic_retrieval_precision=semantic_retrieval_precision(
            candidate_pairs, dataset.relationships
        ),
        contradiction_detection_accuracy=cda,
        precision=precision,
        recall=recall,
        f1=f1,
        recommendation_acceptance_rate=recommendation_acceptance_rate(
            human_feedback, total_recommendations
        ),
        average_processing_time=average_processing_time(processing_times),
    )


__all__ = [
    "EvaluationResult",
    "GroundTruthDataset",
    "GroundTruthRelationship",
    "GroundTruthRequirement",
    "average_processing_time",
    "contradiction_metrics",
    "evaluate",
    "load_ground_truth",
    "recommendation_acceptance_rate",
    "requirement_extraction_accuracy",
    "semantic_retrieval_precision",
]
