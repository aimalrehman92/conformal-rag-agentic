import unittest

from src.calibration.base_calibration import (
    DEFAULT_CONFIDENCE_METHODS,
    normalize_confidence_methods,
)
from src.calibration.calibration_factory import create_split_conformal_calibration
from src.common.config_manager import ConfigManager


class ConfidenceMethodSelectionTests(unittest.TestCase):
    def test_default_confidence_methods_preserve_existing_behavior(self):
        manager = ConfigManager()
        normalized = manager._normalize_config(
            {"conformal": {"implementation": "aimal"}}
        )
        self.assertEqual(
            normalized["conformal"]["confidence_methods"],
            list(DEFAULT_CONFIDENCE_METHODS),
        )

    def test_explicit_r1_only_method_is_preserved(self):
        manager = ConfigManager()
        normalized = manager._normalize_config(
            {
                "conformal": {
                    "implementation": "aimal",
                    "scoring_strategy": "r1",
                    "confidence_methods": ["relavance"],
                }
            }
        )
        self.assertEqual(
            normalized["conformal"]["confidence_methods"],
            ["relavance"],
        )

    def test_invalid_confidence_method_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_confidence_methods(["relavance", "not-a-score"])

    def test_factory_limits_calibrator_to_selected_method(self):
        calibrator = create_split_conformal_calibration(
            implementation="legacy",
            dataset_name="hotpot_qa",
            runs=2,
            seed=42,
            confidence_methods=["relavance"],
        )
        self.assertEqual(calibrator.confidence_method, ["relavance"])


if __name__ == "__main__":
    unittest.main()
