import unittest
from types import SimpleNamespace

from src.common.agentic_policy import AgenticAction
from src.common.huggingface_agentic_policy import HuggingFaceAgenticPolicy
from src.common.llm.huggingface_runtime import HuggingFaceRuntime
from src.common.llm_agentic_policy import LLMAgenticPolicy
from src.common.openai_agentic_policy import OpenAIAgenticPolicy


class StubLLMPolicy(LLMAgenticPolicy):
    def __init__(self, output, **kwargs):
        super().__init__(**kwargs)
        self.output = output
        self.messages = None

    def _generate_action_text(self, messages):
        self.messages = messages
        return self.output


class FakeOpenAICompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ]
        )


class FakeOpenAIClient:
    def __init__(self, content):
        self.completions = FakeOpenAICompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


class FakeHuggingFaceRuntime:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(text=self.text)]


class LLMAgenticPolicyTests(unittest.TestCase):
    def test_parses_retrieve_json(self):
        policy = StubLLMPolicy(
            '{"action":"retrieve","query":"capital of France"}'
        )

        action = policy.decide(
            original_query="What is the capital of France?",
            retrieved_docs=["doc"],
            previous_queries=["France capital"],
            step=1,
            remaining_retrievals=2,
        )

        self.assertEqual(action, AgenticAction.retrieve("capital of France"))

    def test_parses_finish_json_inside_code_fence(self):
        policy = StubLLMPolicy('```json\n{"action":"finish"}\n```')

        action = policy.decide(
            original_query="q",
            retrieved_docs=["doc"],
            previous_queries=["q"],
            step=1,
            remaining_retrievals=1,
        )

        self.assertEqual(action, AgenticAction.finish())

    def test_rejects_invalid_action(self):
        policy = StubLLMPolicy('{"action":"browse","query":"q2"}')

        with self.assertRaises(ValueError):
            policy.decide(
                original_query="q",
                retrieved_docs=[],
                previous_queries=[],
                step=1,
                remaining_retrievals=1,
            )

    def test_prompt_bounds_evidence_and_includes_history(self):
        policy = StubLLMPolicy(
            '{"action":"finish"}',
            max_documents=2,
            max_chars_per_document=4,
        )

        policy.decide(
            original_query="original",
            retrieved_docs=["AAAAA", "BBBBB", "CCCCC"],
            previous_queries=["first", "second"],
            step=2,
            remaining_retrievals=1,
        )

        user_prompt = policy.messages[1]["content"]
        self.assertNotIn("AAAA", user_prompt)
        self.assertIn("BBBB", user_prompt)
        self.assertIn("CCCC", user_prompt)
        self.assertIn("1. first", user_prompt)
        self.assertIn("2. second", user_prompt)
        self.assertIn("Decision step: 2", user_prompt)

    def test_openai_policy_uses_injected_client(self):
        client = FakeOpenAIClient('{"action":"finish"}')
        policy = OpenAIAgenticPolicy(
            model="gpt-4o-mini",
            client=client,
            temperature=0.0,
        )

        action = policy.decide(
            original_query="q",
            retrieved_docs=["doc"],
            previous_queries=["q"],
            step=1,
            remaining_retrievals=1,
        )

        self.assertEqual(action, AgenticAction.finish())
        self.assertEqual(len(client.completions.calls), 1)
        self.assertEqual(
            client.completions.calls[0]["model"],
            "gpt-4o-mini",
        )

    def test_huggingface_policy_uses_injected_runtime(self):
        runtime = FakeHuggingFaceRuntime(
            '{"action":"retrieve","query":"follow up"}'
        )
        policy = HuggingFaceAgenticPolicy(
            model="test-model",
            runtime=runtime,
            temperature=0.0,
        )

        action = policy.decide(
            original_query="q",
            retrieved_docs=["doc"],
            previous_queries=["q"],
            step=1,
            remaining_retrievals=2,
        )

        self.assertEqual(action, AgenticAction.retrieve("follow up"))
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(runtime.calls[0]["n_samples"], 1)

    def test_huggingface_policy_reuses_cached_runtime(self):
        runtime = HuggingFaceRuntime.for_model("unit-test-model")
        policy = HuggingFaceAgenticPolicy(model="unit-test-model")

        self.assertIs(policy.runtime, runtime)


if __name__ == "__main__":
    unittest.main()
