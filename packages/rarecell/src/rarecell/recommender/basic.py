"""Heuristic-only recommender. No LLM; used when [agent] extra not installed."""

from __future__ import annotations

import pandas as pd

from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.base import Recommendation, Recommender


class BasicRecommender(Recommender):
    """Threshold-based keep/drop/purify per cluster.

    keep:    any positive panel pass_frac >= 0.5 AND is_contaminant_frac < 0.1
    drop:    no positive panel pass_frac >= 0.1 OR is_contaminant_frac > 0.4
    purify:  otherwise (mixed signal)
    """

    def __init__(self, profile: TargetCellProfile):
        self.profile = profile

    def recommend(self, table: pd.DataFrame) -> list[Recommendation]:
        positive_names = list(self.profile.positive_markers.keys())
        out: list[Recommendation] = []
        for _, row in table.iterrows():
            pass_fracs = [row.get(f"pass_{n}_frac", 0.0) for n in positive_names]
            best_pass = max(pass_fracs) if pass_fracs else 0.0
            contam = row.get("is_contaminant_frac", 0.0)

            if best_pass >= 0.5 and contam < 0.1:
                rec, conf = "keep", 0.9
                reasoning = f"Strong positive signal ({best_pass:.2f}) and low contamination."
            elif best_pass < 0.1 or contam > 0.4:
                rec, conf = "drop", 0.85
                reasoning = (
                    f"Weak positive ({best_pass:.2f}) " f"or heavy contamination ({contam:.2f})."
                )
            else:
                rec, conf = "purify", 0.55
                reasoning = "Mixed signal — recommend surgical subclustering."

            ev = {n: float(row.get(f"score_{n}_mean", 0.0)) for n in positive_names}
            ev["is_contaminant_frac"] = float(contam)
            out.append(
                Recommendation(
                    cluster_id=str(row["cluster"]),
                    recommendation=rec,
                    confidence=conf,
                    evidence_summary=ev,
                    reasoning=reasoning,
                )
            )
        return out
