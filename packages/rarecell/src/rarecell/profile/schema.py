"""TargetCellProfile pydantic schema. v1.0."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from rarecell.errors import UnreviewedProfileError


class Citation(BaseModel):
    source_id: str
    source: Literal[
        "europepmc", "pubmed", "cellmarker", "panglaodb", "msigdb", "enrichr", "manual", "preset"
    ]
    title: str | None = None
    url: str | None = None


class MarkerPanel(BaseModel):
    genes: list[str]
    threshold_z: float = Field(ge=0)
    citations: list[Citation] = Field(default_factory=list)


class NegativePanel(BaseModel):
    genes: list[str]
    exclusion_threshold_z: float = Field(ge=0, default=1.5)
    citations: list[Citation] = Field(default_factory=list)


class CellTypistRef(BaseModel):
    model: str
    match_patterns: list[str]
    enabled: bool = True


class ReferenceLabels(BaseModel):
    celltypist_models: list[CellTypistRef] = Field(default_factory=list)


class BICCNRules(BaseModel):
    enabled: bool = False
    class_filter: list[str] = Field(default_factory=list)
    subclass_filter: list[str] = Field(default_factory=list)


class CNSTaxonomyConfig(BaseModel):
    enabled: bool = False
    target_node: str | None = None  # e.g. "Astrocyte"
    target_level: Literal["supercluster", "cluster"] = "supercluster"
    reference_release: str | None = None  # bundle tag or "local:<path>"
    min_confidence: float = Field(default=0.5, ge=0, le=1)
    on_missing: Literal["marker_fallback", "skip"] = "marker_fallback"
    mode: Literal["node", "program"] = "node"
    characterize_level: Literal["cluster", "subcluster"] = "cluster"
    rationale: str | None = None
    citations: list[str] = Field(default_factory=list)


class ExpectedAbundance(BaseModel):
    min_fraction: float = Field(ge=0, le=1)
    max_fraction: float = Field(ge=0, le=1)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_order(self) -> ExpectedAbundance:
        if self.min_fraction > self.max_fraction:
            raise ValueError("min_fraction must be <= max_fraction")
        return self


class QCParams(BaseModel):
    min_genes_per_cell: int = Field(ge=1)
    max_pct_mt: float = Field(ge=0, le=100)
    max_genes_per_cell: int = Field(default=10000, ge=1)
    min_cells_per_gene: int = Field(default=3, ge=1)
    rationale: str | None = None


class PurifyParams(BaseModel):
    enabled: bool = True
    high_resolution: float = Field(default=2.0, ge=0)
    min_cluster_purity: float = Field(default=0.7, ge=0, le=1)


class BatchCorrection(BaseModel):
    in_dataset: Literal["harmony", "none"] = "harmony"
    batch_key: str = "sample_id"


class GateAutoPolicy(BaseModel):
    gate1_cluster_decisions: Literal[
        "recommendation", "abort_on_ambiguity", "conservative_drop"
    ] = "recommendation"
    gate2_purify_decisions: Literal["recommendation", "abort_on_ambiguity", "conservative_drop"] = (
        "recommendation"
    )
    gate3_final: Literal["accept", "abort_on_anomaly"] = "accept"
    min_recommendation_confidence: float = Field(default=0.6, ge=0, le=1)
    max_abundance_deviation: float = Field(default=5.0, ge=1.0)


class AutoPolicy(BaseModel):
    gates: GateAutoPolicy = Field(default_factory=GateAutoPolicy)


class DraftedFrom(BaseModel):
    user_prompt: str | None = None
    drafted_by: str | None = None
    drafted_at: datetime | None = None
    rag_sources_consulted: list[str] = Field(default_factory=list)


class TargetCellProfile(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    profile_id: str
    name: str
    description: str
    target_lineage: str
    tissue: list[str]
    expected_abundance: ExpectedAbundance

    positive_markers: dict[str, MarkerPanel]
    negative_markers: dict[str, NegativePanel]
    reference_labels: ReferenceLabels = Field(default_factory=ReferenceLabels)
    biccn_rules: BICCNRules = Field(default_factory=BICCNRules)
    cns_taxonomy: CNSTaxonomyConfig = Field(default_factory=CNSTaxonomyConfig)
    qc: QCParams
    purify: PurifyParams = Field(default_factory=PurifyParams)
    batch_correction: BatchCorrection = Field(default_factory=BatchCorrection)
    auto_policy: AutoPolicy = Field(default_factory=AutoPolicy)

    drafted_from: DraftedFrom = Field(default_factory=DraftedFrom)
    human_reviewed: bool = False
    reviewer: str | None = None
    frozen: bool = False
    content_hash: str | None = None

    @field_validator("positive_markers")
    @classmethod
    def _at_least_one_positive(cls, v: dict[str, MarkerPanel]) -> dict[str, MarkerPanel]:
        if not v:
            raise ValueError("at least one positive_markers panel is required")
        return v

    @model_validator(mode="after")
    def _frozen_requires_review(self) -> TargetCellProfile:
        if self.frozen and not self.human_reviewed:
            raise UnreviewedProfileError(
                "Cannot set frozen=True without human_reviewed=True. "
                "A human must review and sign off on the profile before it is frozen."
            )
        return self

    def freeze(self) -> TargetCellProfile:
        """Return a frozen copy with content_hash set. Requires human_reviewed=True."""
        if not self.human_reviewed:
            raise UnreviewedProfileError(
                "freeze() requires human_reviewed=True. "
                "Set human_reviewed=True and provide reviewer email before freezing."
            )
        h = self.compute_content_hash()
        return self.model_copy(update={"frozen": True, "content_hash": h})

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> TargetCellProfile:
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def to_yaml_path(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))

    def compute_content_hash(self) -> str:
        canonical = self.model_dump(mode="json", exclude={"content_hash", "frozen"})
        payload = yaml.safe_dump(canonical, sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()
