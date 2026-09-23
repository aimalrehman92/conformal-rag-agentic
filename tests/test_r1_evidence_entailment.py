import unittest

from src.subclaim_processor.scorer.r1_evidence_entailment import (
    DEFAULT_R1_NLI_BATCH_SIZE,
    DEFAULT_R1_NLI_MAX_LENGTH,
    DEFAULT_R1_NLI_MODEL,
    R1EvidenceEntailmentScorer,
    extract_document_text,
    r1_score,
    resolve_entailment_index,
)


class R1EvidenceEntailmentTests(unittest.TestCase):
    def test_r1_is_maximum_document_entailment_probability(self):
        self.assertAlmostEqual(r1_score([0.15, 0.82, 0.41]), 0.82)
        self.assertEqual(r1_score([]), 0.0)

    def test_document_extraction_matches_repository_serialization(self):
        document = (
            "page_content='Paris is the capital of France.' "
            "metadata={'title': 'Paris'} indice=7 fileposition=12 score=0.91"
        )
        self.assertEqual(
            extract_document_text(document),
            "Paris is the capital of France.",
        )

    def test_entailment_label_is_resolved_by_label_name(self):
        self.assertEqual(
            resolve_entailment_index(
                {0: "contradiction", 1: "neutral", 2: "entailment"},
                num_labels=3,
            ),
            2,
        )

    def test_r1_scorer_aligns_each_document_with_same_claim(self):
        captured = {}

        def fake_probabilities(premises, hypotheses):
            captured["premises"] = list(premises)
            captured["hypotheses"] = list(hypotheses)
            return [0.2, 0.9]

        scorer = R1EvidenceEntailmentScorer(
            entailment_probability_fn=fake_probabilities
        )

        docs = [
            "page_content='Evidence A' metadata={} indice=1 fileposition=1 score=0.8",
            "page_content='Evidence B' metadata={} indice=2 fileposition=2 score=0.7",
        ]

        score = scorer.score("The claim", docs)

        self.assertEqual(score, 0.9)
        self.assertEqual(captured["premises"], ["Evidence A", "Evidence B"])
        self.assertEqual(captured["hypotheses"], ["The claim", "The claim"])

    def test_notebook_defaults_are_preserved(self):
        scorer = R1EvidenceEntailmentScorer()
        self.assertEqual(scorer.model_id, DEFAULT_R1_NLI_MODEL)
        self.assertEqual(scorer.max_length, DEFAULT_R1_NLI_MAX_LENGTH)
        self.assertEqual(scorer.batch_size, DEFAULT_R1_NLI_BATCH_SIZE)
        self.assertEqual(DEFAULT_R1_NLI_MODEL, "tasksource/deberta-small-long-nli")
        self.assertEqual(DEFAULT_R1_NLI_MAX_LENGTH, 1664)
        self.assertEqual(DEFAULT_R1_NLI_BATCH_SIZE, 8)


if __name__ == "__main__":
    unittest.main()
