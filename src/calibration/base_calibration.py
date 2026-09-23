from abc import ABC, abstractmethod


DEFAULT_CONFIDENCE_METHODS = (
    "relavance",
    "frequency",
    "query_claim_cosine_similarity",
    "doc_claim_cosine_similarity",
    "min_log_prob",
    "random",
    "ordinal",
)


def normalize_confidence_methods(confidence_methods=None):
    """Return a validated ordered list of confidence-score keys."""
    if confidence_methods is None:
        return list(DEFAULT_CONFIDENCE_METHODS)

    if not isinstance(confidence_methods, (list, tuple)) or not confidence_methods:
        raise ValueError("confidence_methods must be a non-empty list or tuple.")

    normalized = []
    for method in confidence_methods:
        if not isinstance(method, str) or not method.strip():
            raise ValueError("Each confidence method must be a non-empty string.")
        method = method.strip()
        if method not in DEFAULT_CONFIDENCE_METHODS:
            raise ValueError(
                f"Unknown confidence method {method!r}. Supported methods are: "
                f"{list(DEFAULT_CONFIDENCE_METHODS)}"
            )
        if method in normalized:
            raise ValueError(f"Duplicate confidence method: {method!r}.")
        normalized.append(method)

    return normalized


class ICalibration(ABC):
    """
    Interface for calibration methods.
    """

    @abstractmethod
    def plot_conformal_removal(
        self, data, alphas, a, fig_filename, csv_filename, plot_group_results=False
    ):
        pass

    @abstractmethod
    def plot_factual_removal(
        self, data, alphas, a, fig_filename, csv_filename, plot_group_results=False
    ):
        pass
