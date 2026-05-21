"""IsolateRunner — executes the S0..S7 state machine with a pluggable Recommender."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import anndata as ad

from rarecell.core import annotate, clustering, evidence, ingest, io, markers, purify, qc
from rarecell.errors import IsolationAbortedError, UnreviewedProfileError
from rarecell.logging import get_logger
from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.base import Recommendation, Recommender
from rarecell.report import Decision, DecisionLog, write_isolation_report
from rarecell.state_machine.states import IsolateState

AutoPolicyName = Literal[
    "recommendation",
    "abort_on_ambiguity",
    "conservative_drop",
    "from_decisions",
]


@dataclass
class IsolateResult:
    """Outcome of a successful IsolateRunner.run()."""

    isolated: ad.AnnData
    final_state: IsolateState
    decisions_path: Path


class IsolateRunner:
    """Drive the IsolateState machine end-to-end on an AnnData input.

    The runner threads a frozen TargetCellProfile through ingest → QC →
    clustering → evidence-based recommendation → optional purify → final
    selection, then writes ``isolated.h5ad`` and ``decisions.jsonl`` to
    ``out_dir``. Gate decisions are produced by ``recommender`` and converted
    into final user decisions according to ``auto_policy``.
    """

    def __init__(
        self,
        *,
        adata: ad.AnnData,
        profile: TargetCellProfile,
        recommender: Recommender,
        out_dir: Path,
        auto_policy: AutoPolicyName = "recommendation",
        replay_decisions_path: Path | None = None,
    ):
        if not profile.frozen:
            raise UnreviewedProfileError(
                "IsolateRunner requires a frozen profile. "
                "Call profile.freeze() (which requires human_reviewed=True)."
            )
        self.adata = adata
        self.profile = profile
        self.recommender = recommender
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.auto_policy: AutoPolicyName = auto_policy
        self.replay_decisions_path = replay_decisions_path
        self.log = DecisionLog(self.out_dir / "decisions.jsonl")
        self.state = IsolateState.S0_LOAD
        self.logger = get_logger("rarecell.runner")
        # Capture input summary before any QC mutation.
        self._input_n_obs = adata.n_obs
        self._input_n_vars = adata.n_vars
        self._input_sample_ids = sorted(set(map(str, adata.obs.get("sample_id", ["_"]))))

    # ── pipeline stages ────────────────────────────────────────────────

    def _s1_ingest(self) -> None:
        ingest.validate_counts(self.adata)

    def _s2_qc(self) -> None:
        import scanpy as sc

        self.adata = qc.run_qc(self.adata, self.profile.qc)
        self.adata = qc.run_scrublet(self.adata, batch_key=self.profile.batch_correction.batch_key)
        # Bundle normalization into QC so downstream stages (clustering,
        # marker scoring) always see log1p(CP10K) in .X. qc.run_qc already
        # stashes raw counts to layers["counts"]; defensively check.
        if "counts" not in self.adata.layers:
            self.adata.layers["counts"] = self.adata.X.copy()
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)

    def _s3_cluster(self) -> None:
        markers.score_profile_markers(self.adata, self.profile, use_raw=False)
        markers.score_negative_panels(self.adata, self.profile, use_raw=False)
        if self.profile.reference_labels.celltypist_models:
            annotate.annotate_celltypist(self.adata, self.profile)
        clustering.taxonomy_cluster(self.adata, self.profile, stage="class")
        if self.profile.biccn_rules.enabled:
            evidence.score_biccn_evidence(self.adata, self.profile, cluster_key="leiden")

    # ── gate decision helpers ──────────────────────────────────────────

    def _decide_for_gate(self, gate: int, recs: list[Recommendation]) -> dict[str, str]:
        decisions: dict[str, str] = {}
        if self.auto_policy == "from_decisions":
            assert (
                self.replay_decisions_path is not None
            ), "auto_policy='from_decisions' requires replay_decisions_path"
            for d in DecisionLog.iter_decisions(self.replay_decisions_path):
                if d.gate == gate:
                    decisions[d.cluster_id] = d.user_decision
            return decisions
        for r in recs:
            if self.auto_policy == "recommendation":
                decisions[r.cluster_id] = r.recommendation
            elif self.auto_policy == "conservative_drop":
                decisions[r.cluster_id] = (
                    "drop" if r.recommendation == "purify" else r.recommendation
                )
            elif self.auto_policy == "abort_on_ambiguity":
                threshold = self.profile.auto_policy.gates.min_recommendation_confidence
                if r.confidence < threshold:
                    decisions[r.cluster_id] = "abort"
                else:
                    decisions[r.cluster_id] = r.recommendation
        return decisions

    def _log_decisions(
        self,
        gate: Literal[1, 2, 3],
        recs: list[Recommendation],
        user_decisions: dict[str, str],
    ) -> None:
        for r in recs:
            ud = user_decisions.get(r.cluster_id, r.recommendation)
            self.log.append(
                Decision(
                    gate=gate,
                    cluster_id=r.cluster_id,
                    recommendation=r.recommendation,
                    user_decision=ud,  # type: ignore[arg-type]
                    confidence=r.confidence,
                    evidence=r.evidence_summary,
                    reasoning=r.reasoning,
                    citations=r.citations,
                )
            )

    def _s4_gate1(self) -> tuple[list[str], list[str]]:
        table = evidence.score_evidence(self.adata, self.profile, cluster_key="leiden")
        recs = self.recommender.recommend(table)
        user_decisions = self._decide_for_gate(1, recs)
        self._log_decisions(1, recs, user_decisions)
        kept = [cid for cid, d in user_decisions.items() if d == "keep"]
        purify_ids = [cid for cid, d in user_decisions.items() if d == "purify"]
        if any(d == "abort" for d in user_decisions.values()):
            raise RuntimeError(
                f"Gate 1 produced an 'abort' decision under auto_policy={self.auto_policy!r}."
            )
        return kept, purify_ids

    def _s5_purify(self, suspect: list[str]) -> ad.AnnData | None:
        if not suspect or not self.profile.purify.enabled:
            return None
        return purify.subcluster_and_purify(
            self.adata,
            self.profile,
            suspect_clusters=suspect,
            cluster_key="leiden",
        )

    def _s5_gate2(self, purified_adata: ad.AnnData) -> list[str]:
        """Gate 2: per-sub-cluster decisions after surgical purify.

        Returns the list of sub-cluster IDs (in ``purified_adata``'s leiden
        labels) that the user/policy decided to keep.
        """
        table = evidence.score_evidence(purified_adata, self.profile, cluster_key="leiden")
        recs = self.recommender.recommend(table)
        user_decisions = self._decide_for_gate(2, recs)
        self._log_decisions(2, recs, user_decisions)
        return [cid for cid, d in user_decisions.items() if d == "keep"]

    def _select_isolated(self, kept_clusters: list[str]) -> ad.AnnData:
        mask = self.adata.obs["leiden"].astype(str).isin(set(kept_clusters))
        return self.adata[mask].copy()

    def _s6_gate3(self, isolated: ad.AnnData, input_n_obs: int) -> None:
        """Gate 3: final abundance abort policy.

        If ``profile.auto_policy.gates.gate3_final == "abort_on_anomaly"`` and
        the isolated fraction is outside ``expected_abundance`` widened by
        ``max_abundance_deviation``, raise :class:`IsolationAbortedError`.
        """
        policy = self.profile.auto_policy.gates
        if policy.gate3_final != "abort_on_anomaly":
            return
        frac = isolated.n_obs / max(input_n_obs, 1)
        lo = self.profile.expected_abundance.min_fraction / policy.max_abundance_deviation
        hi = self.profile.expected_abundance.max_fraction * policy.max_abundance_deviation
        if not (lo <= frac <= hi):
            raise IsolationAbortedError(
                f"Gate 3 abort: isolated abundance {frac:.4f} is outside "
                f"expected bounds [{lo:.4f}, {hi:.4f}] "
                f"(max_abundance_deviation={policy.max_abundance_deviation})."
            )

    # ── public entry point ─────────────────────────────────────────────

    def run(self) -> IsolateResult:
        started_at = datetime.now(UTC)
        try:
            self.state = IsolateState.S1_INGEST
            self.logger.info("runner.state", state=self.state.name)
            self._s1_ingest()

            self.state = IsolateState.S2_QC
            self.logger.info("runner.state", state=self.state.name)
            self._s2_qc()

            self.state = IsolateState.S3_CLUSTER
            self.logger.info("runner.state", state=self.state.name)
            self._s3_cluster()

            self.state = IsolateState.S4_GATE1
            self.logger.info("runner.state", state=self.state.name)
            kept, suspect = self._s4_gate1()

            if suspect:
                self.state = IsolateState.S5_PURIFY
                self.logger.info("runner.state", state=self.state.name)
                purified = self._s5_purify(suspect)
                if purified is not None:
                    # purify returns the full AnnData containing non-suspect
                    # cells plus the cells that survived sub-cluster filtering
                    # from suspect clusters. Gate 2 then runs the recommender
                    # on the purified subset to decide which sub-clusters to
                    # keep.
                    self.adata = purified
                    self.state = IsolateState.S5_GATE2
                    self.logger.info("runner.state", state=self.state.name)
                    sub_kept = self._s5_gate2(purified)
                    kept = sorted(set(kept) | set(sub_kept))

            self.state = IsolateState.S6_FINAL
            self.logger.info("runner.state", state=self.state.name)
            isolated = self._select_isolated(kept)

            # Gate 3: final abundance abort policy.
            self.state = IsolateState.S6_GATE3
            self.logger.info("runner.state", state=self.state.name)
            self._s6_gate3(isolated, self._input_n_obs)

            self.state = IsolateState.S7_REPORT
            self.logger.info(
                "runner.state",
                state=self.state.name,
                n_isolated=int(isolated.n_obs),
                kept_clusters=kept,
            )
            io.save_h5ad(isolated, self.out_dir / "isolated.h5ad")

            write_isolation_report(
                out_dir=self.out_dir,
                profile=self.profile,
                input_n_obs=self._input_n_obs,
                input_n_vars=self._input_n_vars,
                input_sample_ids=self._input_sample_ids,
                isolated=isolated,
                started_at=started_at,
                decisions_path=self.log.path,
            )

            return IsolateResult(
                isolated=isolated,
                final_state=self.state,
                decisions_path=self.log.path,
            )
        except Exception:
            self.state = IsolateState.S_ABORTED
            self.logger.exception("runner.aborted")
            raise
