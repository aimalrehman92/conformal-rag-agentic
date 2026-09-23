import re
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


DEFAULT_R1_NLI_MODEL = "tasksource/deberta-small-long-nli"
DEFAULT_R1_NLI_MAX_LENGTH = 1664
DEFAULT_R1_NLI_BATCH_SIZE = 8


_DOCUMENT_PATTERN = re.compile(
    r"page_content='(.*?)'\smetadata=(\{.*?\})\s"
    r"indice=(\d+)\sfileposition=(\d+)\s"
    r"score=([+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?)",
    re.DOTALL,
)


def r1_score(document_entailment_probabilities: Iterable[float]) -> float:
    """
    Pure Evidence Entailment (R1):

        R1(c) = max_j E(c | d_j)

    This is the exact scalar definition used in the R1-R4 scorer-study
    notebook. Empty evidence receives score 0.0.
    """
    values = np.asarray(list(document_entailment_probabilities), dtype=float)

    if values.size == 0:
        return 0.0

    return float(np.max(values))


def extract_document_text(document) -> str:
    """
    Extract page_content from the repository's serialized retrieval result.

    The primary parser and conservative fallback mirror the scorer-study
    notebook so R1 is computed from the same premise text used there.
    """
    if not isinstance(document, str):
        return str(document)

    match = _DOCUMENT_PATTERN.search(document)

    if match is not None:
        return match.group(1).strip()

    doc_parts = document.rsplit("metadata=", 1)
    page_content = doc_parts[0].strip()

    if page_content.startswith("page_content="):
        page_content = page_content.removeprefix("page_content=").strip()

    if (
        len(page_content) >= 2
        and page_content[0] == page_content[-1]
        and page_content[0] in {"'", '"'}
    ):
        page_content = page_content[1:-1]

    return page_content.strip()


def resolve_entailment_index(id2label: dict, num_labels: int) -> int:
    """Return the unique entailment class index required by the R1 notebook."""
    if num_labels != 3:
        raise ValueError(
            "The configured R1 NLI model must expose exactly three labels."
        )

    normalized = {
        int(index): str(label).lower()
        for index, label in id2label.items()
    }

    entailment_indices = [
        index
        for index, label in normalized.items()
        if "entail" in label
    ]

    if len(entailment_indices) != 1:
        raise ValueError(
            "Could not identify exactly one entailment label from "
            f"NLI id2label={normalized}."
        )

    return entailment_indices[0]


class R1EvidenceEntailmentScorer:
    """
    Repository implementation of the scorer-study R1 relevance function.

    The Hugging Face NLI model is loaded lazily and only when R1 is selected.
    Tests can inject ``entailment_probability_fn`` to validate R1 behavior
    without downloading the model.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_R1_NLI_MODEL,
        max_length: int = DEFAULT_R1_NLI_MAX_LENGTH,
        batch_size: int = DEFAULT_R1_NLI_BATCH_SIZE,
        entailment_probability_fn: Optional[
            Callable[[Sequence[str], Sequence[str]], Sequence[float]]
        ] = None,
    ):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("R1 NLI model_id must be a non-empty string.")
        if max_length < 1:
            raise ValueError("R1 NLI max_length must be at least 1.")
        if batch_size < 1:
            raise ValueError("R1 NLI batch_size must be at least 1.")

        self.model_id = model_id.strip()
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self._injected_probability_fn = entailment_probability_fn

        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._entailment_index = None

    def _ensure_runtime(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "R1 scoring requires torch and transformers. "
                "Install the repository's Hugging Face dependencies before "
                "running scoring_strategy='r1'."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id
        ).to(device)
        model.eval()

        entailment_index = resolve_entailment_index(
            id2label=model.config.id2label,
            num_labels=model.config.num_labels,
        )

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._entailment_index = entailment_index

    def entailment_probabilities(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> np.ndarray:
        """Return P(entailment) for aligned premise-hypothesis pairs."""
        if len(premises) != len(hypotheses):
            raise ValueError("Premise and hypothesis counts must match.")

        if len(premises) == 0:
            return np.empty(0, dtype=float)

        if self._injected_probability_fn is not None:
            probabilities = np.asarray(
                self._injected_probability_fn(premises, hypotheses),
                dtype=float,
            )
            if probabilities.shape != (len(premises),):
                raise ValueError(
                    "Injected R1 entailment_probability_fn must return one "
                    "probability for every premise-hypothesis pair."
                )
            return probabilities

        self._ensure_runtime()

        probabilities = []

        for start in range(0, len(premises), self.batch_size):
            end = start + self.batch_size
            batch_premises = premises[start:end]
            batch_hypotheses = hypotheses[start:end]

            encoded = self._tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self._device)
                for key, value in encoded.items()
            }

            with self._torch.no_grad():
                logits = self._model(**encoded).logits
                batch_probabilities = (
                    self._torch.softmax(logits.float(), dim=-1)[
                        :, self._entailment_index
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

            probabilities.extend(batch_probabilities.tolist())

        return np.asarray(probabilities, dtype=float)

    def score(self, claim: str, retrieved_docs: Sequence[str]) -> float:
        document_texts = [
            extract_document_text(document)
            for document in retrieved_docs
        ]
        document_texts = [
            text for text in document_texts if text.strip()
        ]

        if not document_texts:
            return 0.0

        hypotheses = [claim] * len(document_texts)
        probabilities = self.entailment_probabilities(
            document_texts,
            hypotheses,
        )

        return r1_score(probabilities)
