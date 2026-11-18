"""
Unit tests for Team 3 Accuracy Judge.
"""

import unittest

from agents.schemas import EvidenceReference, ClaimAssessment
from agents.accuracy_judge import AccuracyJudge


class TestAccuracyJudge(unittest.TestCase):

    def setUp(self):

        self.evidence = [
            EvidenceReference(
                source_id="DOC1",
                text="Water boils at approximately 100°C "
                     "at standard atmospheric pressure."
            )
        ]

    def test_correct_claim(self):

        def mock_evaluator(**kwargs):
            return [
                ClaimAssessment(
                    claim_id="C1",
                    claim="Water boils at 100°C at sea level.",
                    classification="supported",
                    evidence=self.evidence,
                    explanation="Supported by the reference."
                )
            ]

        judge = AccuracyJudge(mock_evaluator)

        result = judge.evaluate(
            question="What is the boiling point of water?",
            answer="Water boils at 100°C at sea level.",
            evidence=self.evidence
        )

        self.assertEqual(result.accuracy_score, 100.0)
        self.assertEqual(result.supported_claims, 1)
        self.assertEqual(result.contradicted_claims, 0)

    def test_contradicted_claim(self):

        def mock_evaluator(**kwargs):
            return [
                ClaimAssessment(
                    claim_id="C1",
                    claim="Water boils at 150°C at sea level.",
                    classification="contradicted",
                    evidence=self.evidence,
                    explanation="Contradicts the reference."
                )
            ]

        judge = AccuracyJudge(mock_evaluator)

        result = judge.evaluate(
            question="What is the boiling point of water?",
            answer="Water boils at 150°C at sea level.",
            evidence=self.evidence
        )

        self.assertEqual(result.accuracy_score, 0.0)
        self.assertEqual(result.contradicted_claims, 1)

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
                    claim="Water is used in industries.",
                    classification="unsupported"
                )
            ]

        judge = AccuracyJudge(mock_evaluator)

        result = judge.evaluate(
            question="Tell me about water.",
            answer="Water boils at 100°C, 150°C and is used in industries.",
            evidence=self.evidence
        )

        self.assertEqual(result.total_claims, 3)
        self.assertEqual(result.supported_claims, 1)
        self.assertEqual(result.contradicted_claims, 1)
        self.assertEqual(result.unsupported_claims, 1)
        self.assertEqual(result.accuracy_score, 50.0)

    def test_empty_question(self):

        judge = AccuracyJudge(lambda **kwargs: [])

        with self.assertRaises(ValueError):
            judge.evaluate(
                question="",
                answer="Water boils at 100°C.",
                evidence=self.evidence
            )

    def test_missing_evidence(self):

        judge = AccuracyJudge(lambda **kwargs: [])

        with self.assertRaises(ValueError):
            judge.evaluate(
                question="What is water?",
                answer="Water is a liquid.",
                evidence=[]
            )


if __name__ == "__main__":
    unittest.main()