import unittest

from src.common.agentic_policy import AgenticAction, AgenticPolicy
from src.common.agentic_retriever import AgenticRetriever
from src.common.retriever import Retriever


class FakeRetriever(Retriever):
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def retrieve(self, query, top_k=10, threshold=0.5):
        self.calls.append((query, top_k, threshold))
        return list(self.responses.get(query, []))


class ScriptedPolicy(AgenticPolicy):
    def __init__(self, actions):
        self.actions = list(actions)
        self.observations = []

    def decide(
        self,
        original_query,
        retrieved_docs,
        previous_queries,
        step,
        remaining_retrievals,
    ):
        self.observations.append(
            {
                "original_query": original_query,
                "retrieved_docs": list(retrieved_docs),
                "previous_queries": list(previous_queries),
                "step": step,
                "remaining_retrievals": remaining_retrievals,
            }
        )
        return self.actions.pop(0)


class AgenticRetrieverTests(unittest.TestCase):
    def test_initial_retrieval_then_finish(self):
        base = FakeRetriever({"q": ["doc-a score=0.90"]})
        policy = ScriptedPolicy([AgenticAction.finish()])
        retriever = AgenticRetriever(base, policy)

        docs = retriever.retrieve("q", top_k=5, threshold=0.2)

        self.assertEqual(docs, ["doc-a score=0.90"])
        self.assertEqual(base.calls, [("q", 5, 0.2)])
        self.assertEqual(policy.observations[0]["previous_queries"], ["q"])

    def test_policy_can_issue_targeted_followup_and_documents_are_deduplicated(self):
        base = FakeRetriever(
            {
                "q": ["doc-a score=0.90", "doc-b score=0.80"],
                "followup": ["doc-a score=0.70", "doc-c score=0.60"],
            }
        )
        policy = ScriptedPolicy(
            [AgenticAction.retrieve("followup"), AgenticAction.finish()]
        )
        retriever = AgenticRetriever(base, policy)

        docs = retriever.retrieve("q")

        self.assertEqual(
            docs,
            ["doc-a score=0.90", "doc-b score=0.80", "doc-c score=0.60"],
        )
        self.assertEqual([call[0] for call in base.calls], ["q", "followup"])

    def test_repeated_policy_query_stops_without_duplicate_retrieval(self):
        base = FakeRetriever({"q": ["doc-a score=0.90"]})
        policy = ScriptedPolicy([AgenticAction.retrieve(" q ")])
        retriever = AgenticRetriever(base, policy)

        docs = retriever.retrieve("q")

        self.assertEqual(docs, ["doc-a score=0.90"])
        self.assertEqual(len(base.calls), 1)

    def test_retrieval_budget_is_enforced(self):
        base = FakeRetriever(
            {
                "q": ["doc-a score=0.90"],
                "q2": ["doc-b score=0.80"],
                "q3": ["doc-c score=0.70"],
            }
        )
        policy = ScriptedPolicy(
            [AgenticAction.retrieve("q2"), AgenticAction.retrieve("q3")]
        )
        retriever = AgenticRetriever(
            base,
            policy,
            max_steps=5,
            max_retrievals=2,
        )

        docs = retriever.retrieve("q")

        self.assertEqual(docs, ["doc-a score=0.90", "doc-b score=0.80"])
        self.assertEqual([call[0] for call in base.calls], ["q", "q2"])
        self.assertEqual(len(policy.observations), 1)

    def test_action_validation_rejects_empty_retrieval_query(self):
        with self.assertRaises(ValueError):
            AgenticAction(action="retrieve", query="  ")


if __name__ == "__main__":
    unittest.main()
