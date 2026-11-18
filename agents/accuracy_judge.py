"""
Team 3 - Accuracy Judge

Evaluates factual claims in an AI-generated response
against trusted reference evidence.

The claim evaluator will later connect to GPT-OSS
through Groq.
"""

from typing import Callable, List

from agents.schemas import (
    AccuracyResult,
    ClaimAssessment,
    EvidenceReference,
)


VALID_CLASSIFICATIONS = {
    "supported",
    "contradicted",
    "unsupported",
    "insufficient_evidence",
}


class AccuracyJudge:
    """
    Evaluates factual accuracy using claim-level assessments.

    The evaluator is injected so the agent can work with
    different LLM providers without changing its core logic.
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
    ) -> AccuracyResult:

        # Validate inputs
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

        # Ask the configured evaluator to assess individual claims.
        assessments = self.claim_evaluator(
            question=question,
            answer=answer,
            evidence=evidence,
        )

        if not isinstance(assessments, list):
            raise TypeError("Evaluator must return a list of claims.")

        # Validate and count claim classifications.
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

        supported = sum(
            c.classification == "supported" for c in assessments
        )

        contradicted = sum(
            c.classification == "contradicted" for c in assessments
        )

        unsupported = sum(
            c.classification == "unsupported" for c in assessments
        )

        insufficient = sum(
            c.classification == "insufficient_evidence"
            for c in assessments
        )

        # Only supported and contradicted claims determine
        # factual accuracy. Unsupported claims are not
        # automatically considered false.
        decidable_claims = supported + contradicted

        if decidable_claims:
            accuracy_score = round(
                (supported / decidable_claims) * 100,
                2,
            )
        else:
            accuracy_score = None

        return AccuracyResult(
            accuracy_score=accuracy_score,
            total_claims=len(assessments),
            supported_claims=supported,
            contradicted_claims=contradicted,
            unsupported_claims=unsupported,
            insufficient_evidence_claims=insufficient,
            claims=assessments,
            status="completed",
        )