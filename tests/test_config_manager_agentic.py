import unittest

from src.common.config_manager import ConfigManager


class ConfigManagerAgenticTests(unittest.TestCase):
    def test_agentic_policy_defaults_to_generator_provider_and_model(self):
        config = {
            "models": {
                "generator": {
                    "provider": "huggingface",
                    "name": "meta-llama/Llama-3.1-8B-Instruct",
                }
            },
            "retrieval": {
                "strategy": "agentic",
            },
        }

        normalized = ConfigManager()._normalize_config(config)
        agentic = normalized["retrieval"]["agentic"]

        self.assertEqual(agentic["max_steps"], 3)
        self.assertEqual(agentic["max_retrievals"], 3)
        self.assertTrue(agentic["initial_retrieval"])
        self.assertFalse(agentic["stop_if_no_new_documents"])
        self.assertEqual(agentic["policy"]["provider"], "huggingface")
        self.assertEqual(
            agentic["policy"]["model"],
            "meta-llama/Llama-3.1-8B-Instruct",
        )

    def test_explicit_agentic_policy_overrides_generator(self):
        config = {
            "models": {
                "generator": {
                    "provider": "huggingface",
                    "name": "meta-llama/Llama-3.1-8B-Instruct",
                }
            },
            "retrieval": {
                "strategy": "agentic",
                "agentic": {
                    "max_steps": 5,
                    "max_retrievals": 2,
                    "policy": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "temperature": 0.0,
                    },
                },
            },
        }

        normalized = ConfigManager()._normalize_config(config)
        agentic = normalized["retrieval"]["agentic"]

        self.assertEqual(agentic["max_steps"], 5)
        self.assertEqual(agentic["max_retrievals"], 2)
        self.assertEqual(agentic["policy"]["provider"], "openai")
        self.assertEqual(agentic["policy"]["model"], "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
