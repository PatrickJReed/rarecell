"""Unit tests for scripts.benchmark_isolation.score_isolation."""

from __future__ import annotations

import pandas as pd

from scripts.benchmark_isolation import score_isolation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _labels(mapping: dict[str, str]) -> pd.Series:
    """Build a pd.Series from a barcode -> label dict."""
    return pd.Series(mapping)


# ---------------------------------------------------------------------------
# Basic confusion-matrix test
# ---------------------------------------------------------------------------


def test_score_isolation_basic_confusion_matrix() -> None:
    """Known confusion matrix: TP=3, FP=2, FN=1.

    Universe of 8 cells:
        a0, a1, a2  -> Astrocyte  (positive class)
        n0, n1      -> Neuron
        o0, o1, o2  -> Oligo

    Isolated = {a0, a1, a2, n0, n1}  (missed a2 -> wait, see below)

    Let's be precise:
        target = "Astrocyte"
        isolated = {a0, a1, n0, n1, o0}   <- 5 cells
        TP = a0, a1           (isolated AND Astrocyte) = 2
        FP = n0, n1, o0       (isolated AND NOT Astrocyte) = 3
        FN = a2               (NOT isolated AND Astrocyte) = 1
        precision = 2 / (2+3) = 0.4
        recall    = 2 / (2+1) = 2/3 ≈ 0.6667
        F1        = 2 * 0.4 * (2/3) / (0.4 + 2/3) = 8/15 ≈ 0.5333
    """
    labels = _labels(
        {
            "a0": "Astrocyte",
            "a1": "Astrocyte",
            "a2": "Astrocyte",
            "n0": "Neuron",
            "n1": "Neuron",
            "o0": "Oligo",
            "o1": "Oligo",
            "o2": "Oligo",
        }
    )
    isolated = ["a0", "a1", "n0", "n1", "o0"]
    result = score_isolation(isolated, labels, "Astrocyte")

    assert result["tp"] == 2
    assert result["fp"] == 3
    assert result["fn"] == 1
    assert result["n_isolated"] == 5
    assert result["n_target"] == 3
    assert result["n_total"] == 8

    assert abs(result["precision"] - 2 / 5) < 1e-9  # type: ignore[operator]
    assert abs(result["recall"] - 2 / 3) < 1e-9  # type: ignore[operator]
    expected_f1 = 2 * (2 / 5) * (2 / 3) / ((2 / 5) + (2 / 3))
    assert abs(result["f1"] - expected_f1) < 1e-9  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Perfect isolation
# ---------------------------------------------------------------------------


def test_score_isolation_perfect() -> None:
    """Isolating exactly the target class gives P=R=F1=1.0."""
    labels = _labels({"a0": "Astrocyte", "a1": "Astrocyte", "n0": "Neuron", "n1": "Neuron"})
    result = score_isolation(["a0", "a1"], labels, "Astrocyte")

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


# ---------------------------------------------------------------------------
# Zero isolated — recall 0, precision defined as 0
# ---------------------------------------------------------------------------


def test_score_isolation_zero_isolated() -> None:
    """When nothing is isolated precision and recall are both 0."""
    labels = _labels({"a0": "Astrocyte", "n0": "Neuron", "n1": "Neuron"})
    result = score_isolation([], labels, "Astrocyte")

    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 1  # one Astrocyte exists, none isolated
    assert result["n_isolated"] == 0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


# ---------------------------------------------------------------------------
# Target label absent from dataset
# ---------------------------------------------------------------------------


def test_score_isolation_target_absent() -> None:
    """When the target label doesn't exist recall=0, precision=0."""
    labels = _labels({"n0": "Neuron", "n1": "Neuron"})
    result = score_isolation(["n0"], labels, "Astrocyte")

    assert result["tp"] == 0
    assert result["fn"] == 0  # no ground-truth positives to miss
    assert result["n_target"] == 0
    assert result["recall"] == 0.0
    # precision: isolated_Astrocyte / n_isolated = 0/1 = 0
    assert result["precision"] == 0.0


# ---------------------------------------------------------------------------
# Barcodes in isolated but absent from labels are ignored
# ---------------------------------------------------------------------------


def test_score_isolation_unknown_barcodes_ignored() -> None:
    """Barcodes not in labels.index are silently ignored."""
    labels = _labels({"a0": "Astrocyte", "n0": "Neuron"})
    # "ghost" is not in labels — should not affect counts.
    result = score_isolation(["a0", "ghost"], labels, "Astrocyte")

    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    # n_isolated counts the raw input set length (including ghost)
    assert result["n_isolated"] == 2


# ---------------------------------------------------------------------------
# All isolated are wrong class (pure FP)
# ---------------------------------------------------------------------------


def test_score_isolation_all_wrong_class() -> None:
    """Isolated only non-target cells: precision=0, recall=0."""
    labels = _labels({"a0": "Astrocyte", "n0": "Neuron", "n1": "Neuron"})
    result = score_isolation(["n0", "n1"], labels, "Astrocyte")

    assert result["tp"] == 0
    assert result["fp"] == 2
    assert result["fn"] == 1
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
