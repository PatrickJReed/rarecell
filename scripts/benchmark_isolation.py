"""Precision / recall / F1 benchmark for a rarecell isolation result.

Usage
-----
    uv run python scripts/benchmark_isolation.py \\
        --full    full_dataset.h5ad \\
        --isolated isolated.h5ad \\
        --label-col cell_type_original \\
        --target  Astro

Inputs
------
--full         Full dataset .h5ad file that contains every cell and the
               ground-truth obs column (--label-col).
--isolated     .h5ad whose obs_names are the isolated barcodes.  All
               barcodes must be a subset of the full dataset's obs_names.
--label-col    Name of the obs column in the full dataset that holds
               ground-truth cell-type labels (e.g. "cell_type_original").
--target       The ground-truth label considered the positive class
               (e.g. "Astro").

The script also exposes the pure function ``score_isolation`` which can
be unit-tested without any .h5ad files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable


def score_isolation(
    isolated_barcodes: Iterable[str],
    labels: pd.Series,
    target: str,
) -> dict[str, object]:
    """Compute precision / recall / F1 of an isolation against ground truth.

    Parameters
    ----------
    isolated_barcodes:
        Iterable of cell barcodes that were isolated.  Need not be a
        pandas Index — any iterable of strings works.
    labels:
        A ``pd.Series`` mapping barcode (index) -> ground-truth label for
        **every** cell in the full dataset.  Barcodes in
        ``isolated_barcodes`` that are absent from ``labels.index`` are
        silently ignored (they contribute to FP only if the label is
        unknown, but by convention we skip them so the caller controls
        universe).
    target:
        The ground-truth label to treat as the positive class.

    Returns
    -------
    dict with keys:
        tp, fp, fn, precision, recall, f1,
        n_isolated, n_target, n_total
    """
    isolated_set = set(isolated_barcodes)

    # Restrict to barcodes present in labels.
    known_isolated = isolated_set & set(labels.index)

    tp = int(sum(1 for b in known_isolated if labels[b] == target))
    fp = len(known_isolated) - tp
    n_target = int((labels == target).sum())
    fn = n_target - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_isolated": len(isolated_set),
        "n_target": n_target,
        "n_total": len(labels),
    }


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Score a rarecell isolation result against ground-truth labels."
    )
    ap.add_argument("--full", required=True, type=Path, help="Full dataset .h5ad")
    ap.add_argument(
        "--isolated",
        required=True,
        type=Path,
        help=".h5ad whose obs_names are the isolated barcodes",
    )
    ap.add_argument(
        "--label-col",
        required=True,
        help="obs column in --full that holds ground-truth labels",
    )
    ap.add_argument(
        "--target",
        required=True,
        help="Ground-truth label to treat as the positive class",
    )
    args = ap.parse_args(argv)

    # Lazy import so unit tests of score_isolation don't need anndata.
    import anndata as ad

    full = ad.read_h5ad(args.full, backed="r")
    if args.label_col not in full.obs.columns:
        sys.exit(
            f"ERROR: label column {args.label_col!r} not found in {args.full}.\n"
            f"Available columns: {list(full.obs.columns)}"
        )
    labels: pd.Series = full.obs[args.label_col].astype(str)

    iso = ad.read_h5ad(args.isolated, backed="r")
    isolated_barcodes = list(iso.obs_names)

    result = score_isolation(isolated_barcodes, labels, args.target)

    # Human-readable one-liner.
    print(
        f"target={args.target!r}  "
        f"precision={_fmt_pct(result['precision'])}  "  # type: ignore[arg-type]
        f"recall={_fmt_pct(result['recall'])}  "  # type: ignore[arg-type]
        f"F1={result['f1']:.3f}  "  # type: ignore[str-bytes-safe]
        f"TP={result['tp']}  FP={result['fp']}  FN={result['fn']}  "
        f"n_isolated={result['n_isolated']} / n_target={result['n_target']}"
    )

    # Machine-readable JSON.
    def _round(v: object) -> object:
        if isinstance(v, float) and not math.isnan(v):
            return round(v, 6)
        return v

    print(json.dumps({k: _round(v) for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
