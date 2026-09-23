import logging
import re

from src.common.agentic_policy import AgenticAction, AgenticPolicy
from src.common.retriever import Retriever


logger = logging.getLogger(__name__)


class AgenticRetriever(Retriever):
    """
    Bounded Agentic RAG retrieval controller.

    Unlike SingleHopRetriever and MultiHopRetriever, the next retrieval step is
    selected by an AgenticPolicy after observing the evidence accumulated so
    far. The downstream generation and conformal-prediction pipeline remains
    unchanged: this class only returns retrieved documents.

    By default, the system performs one retrieval with the original question
    before consulting the policy. This keeps the system in the RAG regime while
    still allowing the policy to decide whether additional retrieval is useful.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        policy: AgenticPolicy,
        max_steps: int = 4,
        max_retrievals: int = 3,
        initial_retrieval: bool = True,
        stop_if_no_new_documents: bool = False,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if max_retrievals < 1:
            raise ValueError("max_retrievals must be at least 1.")
        if initial_retrieval and max_retrievals < 1:
            raise ValueError(
                "max_retrievals must allow the configured initial retrieval."
            )

        self.base_retriever = base_retriever
        self.policy = policy
        self.max_steps = max_steps
        self.max_retrievals = max_retrievals
        self.initial_retrieval = initial_retrieval
        self.stop_if_no_new_documents = stop_if_no_new_documents

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.strip().lower().split())

    @staticmethod
    def _document_identity(document: str) -> str:
        """
        Remove the trailing FAISS similarity score from document identity.

        This mirrors the MultiHopRetriever deduplication rule so the same
        underlying chunk is not counted twice merely because its retrieval
        score differs between agent actions.
        """
        return re.sub(
            r"\s+score=[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*$",
            "",
            document,
        ).strip()

    def _retrieve_new_documents(
        self,
        query: str,
        top_k: int,
        threshold: float,
        seen_documents: set[str],
    ) -> list[str]:
        retrieved_docs = self.base_retriever.retrieve(
            query=query,
            top_k=top_k,
            threshold=threshold,
        )

        new_documents = []
        for document in retrieved_docs:
            identity = self._document_identity(document)
            if identity in seen_documents:
                continue
            seen_documents.add(identity)
            new_documents.append(document)

        return new_documents

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[str]:
        if not query or not query.strip():
            raise ValueError("A non-empty query must be provided.")

        original_query = query.strip()
        accumulated_documents: list[str] = []
        seen_documents: set[str] = set()
        seen_queries: set[str] = set()
        previous_queries: list[str] = []
        retrieval_count = 0

        def execute_retrieval(retrieval_query: str) -> int:
            nonlocal retrieval_count

            normalized_query = self._normalize_query(retrieval_query)
            if normalized_query in seen_queries:
                logger.info(
                    "Agentic retrieval stopped: policy repeated query=%r.",
                    retrieval_query,
                )
                return -1

            seen_queries.add(normalized_query)
            previous_queries.append(retrieval_query)
            retrieval_count += 1

            new_documents = self._retrieve_new_documents(
                query=retrieval_query,
                top_k=top_k,
                threshold=threshold,
                seen_documents=seen_documents,
            )
            accumulated_documents.extend(new_documents)

            logger.info(
                "Agentic retrieval action %d/%d: query=%r, new_documents=%d.",
                retrieval_count,
                self.max_retrievals,
                retrieval_query,
                len(new_documents),
            )
            return len(new_documents)

        if self.initial_retrieval:
            initial_new_count = execute_retrieval(original_query)
            if initial_new_count < 0:
                return accumulated_documents
            if (
                self.stop_if_no_new_documents
                and initial_new_count == 0
            ):
                logger.info(
                    "Agentic retrieval stopped after initial retrieval: "
                    "no new documents."
                )
                return accumulated_documents

        for step in range(1, self.max_steps + 1):
            remaining_retrievals = self.max_retrievals - retrieval_count
            if remaining_retrievals <= 0:
                logger.info(
                    "Agentic retrieval stopped: reached max_retrievals=%d.",
                    self.max_retrievals,
                )
                break

            action = self.policy.decide(
                original_query=original_query,
                retrieved_docs=list(accumulated_documents),
                previous_queries=list(previous_queries),
                step=step,
                remaining_retrievals=remaining_retrievals,
            )

            if not isinstance(action, AgenticAction):
                raise TypeError(
                    "AgenticPolicy.decide() must return AgenticAction. "
                    f"Got {type(action).__name__}."
                )

            if action.action == "finish":
                logger.info(
                    "Agentic retrieval stopped at policy step %d: finish action.",
                    step,
                )
                break

            new_count = execute_retrieval(action.query)
            if new_count < 0:
                break
            if self.stop_if_no_new_documents and new_count == 0:
                logger.info(
                    "Agentic retrieval stopped after policy step %d: "
                    "no new documents.",
                    step,
                )
                break
        else:
            logger.info(
                "Agentic retrieval stopped: reached max_steps=%d.",
                self.max_steps,
            )

        return accumulated_documents
