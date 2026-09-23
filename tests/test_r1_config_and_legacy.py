import unittest

from src.calibration.author_conformal import AuthorSplitConformalCalibration
from src.calibration.calibration_factory import create_split_conformal_calibration
from src.common.config_manager import ConfigManager


class R1ConfigAndLegacyTests(unittest.TestCase):
    def test_r1_relevance_model_defaults_match_scorer_study(self):
        manager = ConfigManager()
        normalized = manager._normalize_config(
            {
                "models": {
                    "generator": {"provider": "openai", "name": "gpt-4o-mini"},
                    "embedding": {
                        "provider": "openai",
                        "name": "text-embedding-3-large",
                    },
                },
                "conformal": {
                    "implementation": "aimal",
                    "aggregation_strategy": "max",
                    "scoring_strategy": "r1",
                    "alphas": {"start": 0.05, "end": 0.15, "step": 0.05},
                },
            }
        )

        relevance = normalized["models"]["relevance_scorer"]
        self.assertEqual(relevance["provider"], "huggingface")
        self.assertEqual(relevance["name"], "tasksource/deberta-small-long-nli")
        self.assertEqual(relevance["max_length"], 1664)
        self.assertEqual(relevance["batch_size"], 8)
        self.assertEqual(normalized["conformal"]["scoring_strategy"], "r1")

    def test_explicit_r1_model_settings_are_preserved(self):
        manager = ConfigManager()
        normalized = manager._normalize_config(
            {
                "models": {
                    "relevance_scorer": {
                        "provider": "huggingface",
                        "name": "example/nli",
                        "max_length": 1024,
                        "batch_size": 4,
                    }
                },
                "conformal": {"implementation": "aimal"},
            }
        )

        self.assertEqual(
            normalized["models"]["relevance_scorer"],
            {
                "provider": "huggingface",
                "name": "example/nli",
                "max_length": 1024,
                "batch_size": 4,
            },
        )

    def test_legacy_alias_selects_original_author_implementation(self):
        manager = ConfigManager()
        normalized = manager._normalize_config(
            {"conformal": {"implementation": "legacy"}}
        )
        self.assertEqual(normalized["conformal"]["implementation"], "legacy")

        calibrator = create_split_conformal_calibration(
            implementation="legacy",
            dataset_name="hotpot_qa",
            runs=2,
            seed=42,
        )
        self.assertIsInstance(calibrator, AuthorSplitConformalCalibration)


if __name__ == "__main__":
    unittest.main()
