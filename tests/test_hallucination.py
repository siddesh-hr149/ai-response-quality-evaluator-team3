"""
Unit tests for Team 3 Hallucination Detection Agent.
"""

import unittest

from agents.schemas import EvidenceReference, ClaimAssessment
from agents.hallucination_judge import HallucinationJudge


class TestHallucinationJudge(unittest.TestCase):

    def setUp(self):
        self.evidence = [
            EvidenceReference(
                source_id="DOC1",
                text="Water boils at approximately 100°C "
                     "at standard atmospheric pressure."
            )
        ]

    def test_supported_claim(self):

        def mock_evaluator(**kwargs):
            return [
                ClaimAssessment(
                    claim_id="C1",
                    claim="Water boils at 100°C.",
                    classification="supported",
                    evidence=self.evidence,
                    explanation="Supported by reference."
                )
            ]

        judge = HallucinationJudge(mock_evaluator)

        result = judge.evaluate(
            question="What is the boiling point of water?",
            answer="Water boils at 100°C.",
            evidence=self.evidence
        )

        self.assertEqual(result.hallucination_risk_score, 0.0)
        self.assertEqual(len(result.detected_issues), 0)

    def test_contradicted_claim(self):

        def mock_evaluator(**kwargs):
            return [
                ClaimAssessment(
                    claim_id="C1",
                    claim="Water boils at 150°C.",
                    classification="contradicted",
                    evidence=self.evidence,
                    explanation="Contradicts reference."
                )
            ]

        judge = HallucinationJudge(mock_evaluator)

        result = judge.evaluate(
            question="What is the boiling point of water?",
            answer="Water boils at 150°C.",
            evidence=self.evidence
        )

        self.assertEqual(result.hallucination_risk_score, 100.0)
        self.assertEqual(result.contradicted_claims, 1)
        self.assertEqual(len(result.detected_issues), 1)

    def test_unsupported_claim(self):

        def mock_evaluator(**kwargs):
            return [
                ClaimAssessment(
                    claim_id="C1",
                    claim="Water is used in every machine.",
                    classification="unsupported",
                    evidence=[],
                    explanation="Not established by reference."
                )
            ]

        judge = HallucinationJudge(mock_evaluator)

        result = judge.evaluate(
            question="Tell me about water.",
            answer="Water is used in every machine.",
            evidence=self.evidence
        )

        self.assertEqual(result.hallucination_risk_score, 50.0)
        self.assertEqual(result.unsupported_claims, 1)

    def test_mixed_claims(self):

        def mock_evaluator(**kwargs):
            return [
                ClaimAssessment(
                    claim_id="C1",
                    claim="Water boils at 100°C.",
                    classification="supported"
                ),
                ClaimAssessment(
                    claim_id="C2",
                    claim="Water boils at 150°C.",
                    classification="contradicted"
                ),
                ClaimAssessment(
                    claim_id="C3",
                    claim="Water is used in every machine.",
                    classification="unsupported"
                )
            ]

        judge = HallucinationJudge(mock_evaluator)

        result = judge.evaluate(
            question="Tell me about water.",
            answer="Water boils at 100°C and 150°C.",
            evidence=self.evidence
        )

        self.assertEqual(result.total_claims, 3)
        self.assertEqual(result.contradicted_claims, 1)
        self.assertEqual(result.unsupported_claims, 1)
        self.assertEqual(result.hallucination_risk_score, 50.0)
        self.assertEqual(len(result.detected_issues), 2)

    def test_empty_question(self):

        judge = HallucinationJudge(lambda **kwargs: [])

        with self.assertRaises(ValueError):
            judge.evaluate(
                question="",
                answer="Water boils at 100°C.",
                evidence=self.evidence
            )

    def test_missing_evidence(self):

        judge = HallucinationJudge(lambda **kwargs: [])

        with self.assertRaises(ValueError):
            judge.evaluate(
                question="What is water?",
                answer="Water is a liquid.",
                evidence=[]
            )


if __name__ == "__main__":
    unittest.main()