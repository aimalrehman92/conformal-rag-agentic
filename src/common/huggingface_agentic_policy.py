from src.common.llm.huggingface_runtime import HuggingFaceRuntime
from src.common.llm_agentic_policy import LLMAgenticPolicy


class HuggingFaceAgenticPolicy(LLMAgenticPolicy):
    """Hugging Face-backed decision policy for Agentic RAG retrieval."""

    def __init__(
        self,
        model: str,
        runtime: HuggingFaceRuntime = None,
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
        self.runtime = runtime or HuggingFaceRuntime.for_model(self.model)
        self.temperature = temperature

    def _generate_action_text(self, messages: list[dict[str, str]]) -> str:
        generations = self.runtime.generate(
            messages=messages,
            temperature=self.temperature,
            n_samples=1,
            max_new_tokens=128,
            return_token_probabilities=False,
        )

        if len(generations) != 1:
            raise ValueError(
                "Agentic policy generation must return exactly one sequence. "
                f"Received {len(generations)}."
            )

        return generations[0].text
