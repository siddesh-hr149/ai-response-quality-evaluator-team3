"""
Shared data schemas for Team 3 evaluation agents.

These schemas define the common input and output formats
for the Accuracy Judge and Hallucination Detection Agent.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional


# Allowed claim classifications
ClaimClassification = Literal[
    "supported",
    "contradicted",
    "unsupported",
    "insufficient_evidence"
]


@dataclass
class EvidenceReference:
    """Reference evidence used to evaluate a claim."""

    source_id: str
    text: str


@dataclass
class ClaimAssessment:
    """Evaluation result for an individual factual claim."""

    claim_id: str
    claim: str

    classification: ClaimClassification

    evidence: List[EvidenceReference] = field(default_factory=list)

    explanation: str = ""


@dataclass
class AccuracyResult:
    """Structured output from the Accuracy Judge."""

    accuracy_score: Optional[float]

    total_claims: int

    supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    insufficient_evidence_claims: int

    claims: List[ClaimAssessment] = field(default_factory=list)

    status: str = "completed"


@dataclass
class HallucinationResult:
    """Structured output from the Hallucination Detection Agent."""

    hallucination_risk_score: Optional[float]

    total_claims: int

    unsupported_claims: int
    contradicted_claims: int

    detected_issues: List[ClaimAssessment] = field(default_factory=list)

    status: str = "completed"


def to_dict(result):
    """
    Convert an evaluation result into a dictionary
    that can be serialized as JSON.
    """

    return asdict(result)