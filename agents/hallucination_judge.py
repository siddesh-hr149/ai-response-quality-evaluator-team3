"""
Team 3 - Hallucination Detection Agent

Identifies unsupported and contradicted claims
using supplied reference evidence.
"""

from typing import Callable, List

from agents.schemas import (
    HallucinationResult,
    ClaimAssessment,
    EvidenceReference,
)


VALID_CLASSIFICATIONS = {
    "supported",
    "contradicted",
    "unsupported",
    "insufficient_evidence",
}


class HallucinationJudge:
    """
    Evaluates claims to identify potential hallucinations.

    The claim evaluator can later be connected to
    GPT-OSS through Groq.
    """

    def __init__(self, claim_evaluator: Callable):

        if not callable(claim_evaluator):
            raise TypeError("claim_evaluator must be callable.")

        self.claim_evaluator = claim_evaluator

    def evaluate(
        self,
        question: str,
        answer: str,
        evidence: List[EvidenceReference],
    ) -> HallucinationResult:

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not evidence:
            raise ValueError("Reference evidence is required.")

        if not all(
            isinstance(item, EvidenceReference) for item in evidence
        ):
            raise TypeError(
                "Evidence must contain EvidenceReference objects."
            )

        assessments = self.claim_evaluator(
            question=question,
            answer=answer,
            evidence=evidence,
        )

        if not isinstance(assessments, list):
            raise TypeError("Evaluator must return a list of claims.")

        claim_ids = set()

        for claim in assessments:

            if not isinstance(claim, ClaimAssessment):
                raise TypeError(
                    "Each assessment must be a ClaimAssessment."
                )

            if not claim.claim_id or not claim.claim.strip():
                raise ValueError("Every claim needs an ID and text.")

            if claim.claim_id in claim_ids:
                raise ValueError(
                    f"Duplicate claim ID: {claim.claim_id}"
                )

            claim_ids.add(claim.claim_id)

            if claim.classification not in VALID_CLASSIFICATIONS:
                raise ValueError(
                    f"Invalid classification: {claim.classification}"
                )

        contradicted = sum(
            c.classification == "contradicted"
            for c in assessments
        )

        unsupported = sum(
            c.classification == "unsupported"
            for c in assessments
        )

        total = len(assessments)

        # Provisional risk heuristic:
        # Contradicted claims receive full weight.
        # Unsupported claims receive half weight.
        # This is not a calibrated probability.
        if total:
            risk_score = round(
                ((contradicted + 0.5 * unsupported) / total) * 100,
                2,
            )
        else:
            risk_score = None

        detected_issues = [
            claim
            for claim in assessments
            if claim.classification in {
                "contradicted",
                "unsupported",
            }
        ]

        return HallucinationResult(
            hallucination_risk_score=risk_score,
            total_claims=total,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            detected_issues=detected_issues,
            status="completed",
        )