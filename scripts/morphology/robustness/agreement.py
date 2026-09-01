"""Inter-rater agreement helpers shared by the robustness and judge-validation scripts.

Every agreement number in this release goes through `kappa` and `agreement_rate` so the
same convention applies whether the raters are two induction replicas, two reruns of the
judge, or the judge and a human annotator.
"""
from typing import Literal

from sklearn.metrics import accuracy_score, cohen_kappa_score

# The judge's three-point ordinal, before rescaling to integer class labels.
ORDINAL_SCALE = (0.0, 0.5, 1.0)


def ordinal_labels(scores: list[float]) -> list[int]:
    """Map the judge's 0.0/0.5/1.0 scale to integer class labels.

    sklearn's cohen_kappa_score infers a "continuous" target type from any float sequence
    containing a non-integer value like 0.5 and refuses to score it, even though it is
    really a 3-point ordinal scale. Rescaling to {0, 1, 2} keeps the ordering (so
    weights="linear" still penalizes by label distance) while making the input discrete.
    """
    return [round(score * 2) for score in scores]


def kappa(
    left: list[int],
    right: list[int],
    weights: Literal["linear", "quadratic"] | None = None,
) -> float:
    """Cohen's kappa with a degenerate-input fallback.

    sklearn returns NaN when both label sequences are constant (no variance to explain) --
    a common case here given how often exact/bag match is 1.0 across the whole item set.
    Score that as perfect agreement (both sides said the same thing on every item) rather
    than propagating NaN.
    """
    if len(set(left)) == 1 and len(set(right)) == 1:
        return 1.0 if left[0] == right[0] else 0.0
    return float(cohen_kappa_score(left, right, weights=weights))


def agreement_rate(left: list[int], right: list[int]) -> float:
    """Fraction of paired observations where both raters gave the same label."""
    return float(accuracy_score(left, right))


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
