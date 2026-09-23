from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional


AgenticActionName = Literal["retrieve", "finish"]


@dataclass(frozen=True)
class AgenticAction:
    """One bounded action selected by an Agentic RAG policy."""

    action: AgenticActionName
    query: Optional[str] = None

    def __post_init__(self) -> None:
        normalized_action = self.action.strip().lower()
        if normalized_action not in {"retrieve", "finish"}:
            raise ValueError(
                "Agentic action must be either 'retrieve' or 'finish'. "
                f"Got {self.action!r}."
            )
        object.__setattr__(self, "action", normalized_action)

        if normalized_action == "retrieve":
            if self.query is None or not self.query.strip():
                raise ValueError("A retrieve action requires a non-empty query.")
            object.__setattr__(self, "query", self.query.strip())
        else:
            object.__setattr__(self, "query", None)

    @classmethod
    def retrieve(cls, query: str) -> "AgenticAction":
        return cls(action="retrieve", query=query)

    @classmethod
    def finish(cls) -> "AgenticAction":
        return cls(action="finish")


class AgenticPolicy(ABC):
    """
    Provider-neutral decision interface for Agentic RAG.

    The policy does not retrieve documents itself. It observes the original
    question, accumulated evidence, and retrieval history, then chooses one of
    two bounded actions: issue another retrieval query or finish retrieval.
    """

    @abstractmethod
    def decide(
        self,
        original_query: str,
        retrieved_docs: list[str],
        previous_queries: list[str],
        step: int,
        remaining_retrievals: int,
    ) -> AgenticAction:
        raise NotImplementedError
