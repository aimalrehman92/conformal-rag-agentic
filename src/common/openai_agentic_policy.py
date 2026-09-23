from typing import Any, Optional

from src.common.llm_agentic_policy import LLMAgenticPolicy


class OpenAIAgenticPolicy(LLMAgenticPolicy):
    """OpenAI-backed decision policy for Agentic RAG retrieval."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: Optional[Any] = None,
        temperature: float = 0.0,
        max_documents: int = 5,
        max_chars_per_document: int = 1200,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        if temperature < 0:
            raise ValueError("temperature must be non-negative.")

        super().__init__(
            max_documents=max_documents,
            max_chars_per_document=max_chars_per_document,
        )
        self.model = model.strip()
        self._client = client
        self.temperature = temperature

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _generate_action_text(self, messages: list[dict[str, str]]) -> str:
        response = self._get_client().chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=128,
            messages=messages,
        )

        if not response.choices:
            raise ValueError("OpenAI agentic policy returned no choices.")

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI agentic policy returned empty content.")

        return content
