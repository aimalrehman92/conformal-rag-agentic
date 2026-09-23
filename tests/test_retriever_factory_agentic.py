import sys
import types
import unittest


# Keep this unit test independent of optional runtime dependencies used by the
# repository's production FAISS/OpenAI adapters. The factory behavior tested
# below does not call those dependencies.
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

if "src.common.faiss_manager" not in sys.modules:
    faiss_stub = types.ModuleType("src.common.faiss_manager")

    class FAISSIndexManager:
        pass

    faiss_stub.FAISSIndexManager = FAISSIndexManager
    sys.modules["src.common.faiss_manager"] = faiss_stub

from src.common.agentic_policy import AgenticAction, AgenticPolicy
from src.common.agentic_retriever import AgenticRetriever
from src.common.huggingface_agentic_policy import HuggingFaceAgenticPolicy
from src.common.llm.huggingface_runtime import HuggingFaceRuntime
from src.common.openai_agentic_policy import OpenAIAgenticPolicy
from src.common.retriever_factory import create_retriever
from src.common.single_hop_retriever import SingleHopRetriever


class FinishPolicy(AgenticPolicy):
    def decide(
        self,
        original_query,
        retrieved_docs,
        previous_queries,
        step,
        remaining_retrievals,
    ):
        return AgenticAction.finish()


class RetrieverFactoryAgenticTests(unittest.TestCase):
    def test_agentic_strategy_wraps_existing_single_hop_retriever(self):
        policy = FinishPolicy()
        faiss_manager = object()

        retriever = create_retriever(
            strategy="agentic",
            faiss_manager=faiss_manager,
            truncation_strategy="fixed_length",
            truncate_by="\n",
            agentic_config={
                "max_steps": 4,
                "max_retrievals": 3,
                "initial_retrieval": True,
                "stop_if_no_new_documents": False,
            },
            agentic_policy=policy,
        )

        self.assertIsInstance(retriever, AgenticRetriever)
        self.assertIsInstance(retriever.base_retriever, SingleHopRetriever)
        self.assertIs(retriever.policy, policy)
        self.assertEqual(retriever.max_steps, 4)
        self.assertEqual(retriever.max_retrievals, 3)
        self.assertTrue(retriever.initial_retrieval)
        self.assertFalse(retriever.stop_if_no_new_documents)

    def test_factory_builds_openai_policy_lazily(self):
        retriever = create_retriever(
            strategy="agentic",
            faiss_manager=object(),
            agentic_config={
                "policy": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                }
            },
        )

        self.assertIsInstance(retriever.policy, OpenAIAgenticPolicy)
        self.assertEqual(retriever.policy.model, "gpt-4o-mini")
        self.assertIsNone(retriever.policy._client)

    def test_factory_builds_huggingface_policy_with_shared_runtime(self):
        model = "factory-unit-test-model"
        shared_runtime = HuggingFaceRuntime.for_model(model)

        retriever = create_retriever(
            strategy="agentic",
            faiss_manager=object(),
            agentic_config={
                "policy": {
                    "provider": "huggingface",
                    "model": model,
                }
            },
        )

        self.assertIsInstance(retriever.policy, HuggingFaceAgenticPolicy)
        self.assertIs(retriever.policy.runtime, shared_runtime)
        self.assertFalse(shared_runtime.is_loaded)

    def test_existing_single_hop_strategy_is_unchanged(self):
        retriever = create_retriever(
            strategy="single_hop",
            faiss_manager=object(),
        )

        self.assertIsInstance(retriever, SingleHopRetriever)

    def test_unknown_strategy_error_lists_agentic(self):
        with self.assertRaisesRegex(ValueError, "agentic"):
            create_retriever(
                strategy="unknown",
                faiss_manager=object(),
            )


if __name__ == "__main__":
    unittest.main()
