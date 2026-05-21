# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # rarecell — paper-to-cells demo (Colab-ready)
#
# You read a recent paper describing a novel disease-associated cell
# population. You want to isolate that population from your own dataset to
# study it further. Can rarecell do that?
#
# This notebook walks the full workflow on a real public example:
#
# - **Paper:** Ling et al., _Nature_ 2024 — _A concerted neuron-astrocyte
#   program declines in ageing and schizophrenia_
#   ([PMID:38448582](https://pubmed.ncbi.nlm.nih.gov/38448582/)).
#   The paper identifies *SNAP* (Synaptic Neuron and Astrocyte Program), a
#   gene program whose expression in astrocytes (and matching neurons)
#   declines in schizophrenia and aging.
# - **Dataset:** one SZ DLPFC sample from the SZBDMulti-Seq cohort hosted
#   on [brainSCOPE](https://brainscope.gersteinlab.org/) — 7,912 cells,
#   pre-annotated, ~6% astrocytes. Streamed in via remote-zip Range
#   requests, no full archive download.
#
# Pipeline:
#
# 1. Install `rarecell` from GitHub.
# 2. Stream one SZ sample's annotated matrix from brainSCOPE and convert
#    to AnnData.
# 3. Compose a Ling-anchored astrocyte profile (canonical astrocyte
#    markers + Ling's SNAP gene set as supplementary signal). Two paths:
#    - Default: hand-written profile (no API key required).
#    - Optional: `draft_profile_from_prompt(anchor_paper=PMID)` — lets
#      Claude compose the profile grounded in the paper's abstract.
# 4. `rarecell validate-profile` — pre-flight check: do the markers exist
#    in this dataset?
# 5. `IsolateRunner` — run the pipeline.
# 6. Inspect results, including how rarecell's isolated subset compares
#    against the original brainSCOPE annotations.
# 7. Replay determinism.

# %% [markdown]
# ## 1. Install rarecell

# %%
# !pip install -q "git+https://github.com/PatrickJReed/rarecell.git#subdirectory=packages/rarecell"
# !pip install -q remotezip

# %% [markdown]
# ## 2. Stream one SZ sample from brainSCOPE
#
# `brainscope.gersteinlab.org` hosts per-cohort `.zip` archives (each
# ~1–2.5 GB) containing per-sample annotated TSV matrices. We use
# `remotezip` to fetch only the bytes for one sample — ~24 MB compressed —
# without downloading the full archive.

# %%
from pathlib import Path

import remotezip

ARCHIVE_URL = "https://brainscope.gersteinlab.org/data/snrna_expr_matrices_zip/SZBDMulti-Seq.zip"
SAMPLE_PATH_IN_ZIP = "SZBDMulti-Seq/SZ11-annotated_matrix.txt.gz"

local_root = Path("./rarecell_demo")
local_root.mkdir(exist_ok=True)
local_sample = local_root / "SZ11-annotated_matrix.txt.gz"

if not local_sample.exists():
    with remotezip.RemoteZip(ARCHIVE_URL) as rz:
        rz.extract(SAMPLE_PATH_IN_ZIP, str(local_root))
    # remotezip extracts preserving the in-zip path; flatten it
    (local_root / SAMPLE_PATH_IN_ZIP).rename(local_sample)
    (local_root / "SZBDMulti-Seq").rmdir()

size_mb = local_sample.stat().st_size / 1e6
print(f"Downloaded {local_sample.name} ({size_mb:.1f} MB compressed)")

# %% [markdown]
# ## 3. Convert TSV → AnnData
#
# The brainSCOPE format: rows are genes, columns are cells, with cell-type
# annotations as the first row of the header (instead of a separate
# metadata file). We transpose to the standard AnnData orientation
# (cells × genes) and preserve the original annotations in
# `obs["cell_type_original"]`.

# %%
import gzip

import anndata as ad
import numpy as np
import pandas as pd

with gzip.open(local_sample, "rt") as f:
    header = next(f).rstrip("\n").split("\t")
cell_types_original = header[1:]

df = pd.read_csv(local_sample, sep="\t", index_col=0)
df.index.name = "gene"
print(f"raw shape (genes × cells): {df.shape}")
df.columns = [f"cell_{i:04d}" for i in range(df.shape[1])]

adata = ad.AnnData(X=df.T.values.astype(np.float32))
adata.obs_names = list(df.columns)
adata.var_names = list(df.index)
adata.var_names_make_unique()
adata.obs["cell_type_original"] = pd.Categorical(cell_types_original)
adata.obs["sample_id"] = "SZ11"
adata.obs["disease"] = "schizophrenia"
print(f"AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
print("\nOriginal cell-type composition (top 10):")
print(adata.obs["cell_type_original"].value_counts().head(10))

# %% [markdown]
# ## 4. Compose a Ling-anchored astrocyte profile
#
# Ling et al. characterise SNAP within astrocytes by its enrichment for
# (i) synaptic-support genes and (ii) cholesterol-synthesis genes. We
# build a profile that targets astrocytes broadly — canonical markers as
# the positive panel, brain-cell exclusions as negative panels — and add a
# `snap_astrocyte` positive panel with Ling's signature genes so the
# isolated subset is biased toward SNAP-expressing astrocytes (the
# population that *declines* in SZ).
#
# This is the no-API-key path. The cell after the next shows the
# alternative: using `draft_profile_from_prompt(anchor_paper=...)` to let
# Claude compose this from the paper's abstract.

# %%
from rarecell.profile.schema import (
    BatchCorrection,
    BICCNRules,
    Citation,
    ExpectedAbundance,
    MarkerPanel,
    NegativePanel,
    PurifyParams,
    QCParams,
    ReferenceLabels,
    TargetCellProfile,
)

LING_CITE = Citation(
    source_id="pmid:38448582",
    source="europepmc",
    title="A concerted neuron-astrocyte program declines in ageing and schizophrenia",
    url="https://pubmed.ncbi.nlm.nih.gov/38448582/",
)

profile = TargetCellProfile(
    profile_id="ling-2024-astrocyte-snap",
    name="Astrocytes (SNAP-aware)",
    description=(
        "Astrocyte isolation profile anchored on Ling et al. Nature 2024 "
        "(SNAP — synaptic neuron and astrocyte program). Pan-astrocyte "
        "markers + SNAP-associated synaptic + cholesterol synthesis genes."
    ),
    target_lineage="neural",
    tissue=["brain", "dlpfc"],
    expected_abundance=ExpectedAbundance(
        min_fraction=0.02,
        max_fraction=0.20,
        notes="Astrocytes are typically 5-15% of DLPFC nuclei.",
    ),
    positive_markers={
        "pan_astrocyte": MarkerPanel(
            genes=["GFAP", "AQP4", "ALDH1L1", "SLC1A2", "SLC1A3", "S100B", "GJA1"],
            threshold_z=1.0,
            citations=[
                Citation(source_id="panglaodb:Astrocyte", source="panglaodb"),
            ],
        ),
        "snap_astrocyte_synaptic": MarkerPanel(
            genes=["NRXN1", "NLGN1", "GLUL", "SPARCL1", "NCAN"],
            threshold_z=1.0,
            citations=[LING_CITE],
        ),
        "snap_astrocyte_cholesterol": MarkerPanel(
            genes=["HMGCR", "SQLE", "DHCR7", "FDPS", "INSIG1", "LDLR"],
            threshold_z=1.0,
            citations=[LING_CITE],
        ),
    },
    negative_markers={
        "neuron": NegativePanel(
            genes=["RBFOX3", "SYT1", "SNAP25", "NRGN", "SLC17A7"],
            exclusion_threshold_z=1.5,
        ),
        "oligodendrocyte": NegativePanel(
            genes=["MBP", "MOG", "PLP1", "MAG", "MOBP"],
            exclusion_threshold_z=1.5,
        ),
        "microglia": NegativePanel(
            genes=["CX3CR1", "P2RY12", "TMEM119", "AIF1"],
            exclusion_threshold_z=1.5,
        ),
    },
    reference_labels=ReferenceLabels(celltypist_models=[]),
    biccn_rules=BICCNRules(enabled=False),
    qc=QCParams(
        min_genes_per_cell=200,
        max_pct_mt=10,
        max_genes_per_cell=10000,
        min_cells_per_gene=3,
        rationale="Standard snRNA-seq QC for postmortem DLPFC.",
    ),
    purify=PurifyParams(enabled=True, min_cluster_purity=0.2),
    batch_correction=BatchCorrection(in_dataset="harmony", batch_key="sample_id"),
)

profile = profile.model_copy(
    update={
        "human_reviewed": True,
        "reviewer": "colab-demo@example.com",
    }
).freeze()

print(f"Profile: {profile.profile_id}")
print(f"Frozen: {profile.frozen}")
print(f"Positive panels: {list(profile.positive_markers.keys())}")

# %% [markdown]
# ### Optional: let Claude draft the profile from the paper
#
# Skip this cell to use the hand-written profile above. To enable, install
# the `[agent]` extra and provide your Anthropic API key:
#
# ```python
# # !pip install -q "git+https://github.com/PatrickJReed/rarecell.git#subdirectory=packages/rarecell[agent]" "git+https://github.com/PatrickJReed/rarecell.git#subdirectory=packages/rarecell-mcp-knowledge"
# # import os
# # os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # or use Colab secrets
# #
# # from rarecell.agent.client import AnthropicClient
# # from rarecell.agent.draft import draft_profile_from_prompt
# # from rarecell.rag.knowledge import build_knowledge_session
# #
# # session = build_knowledge_session(
# #     catalog_path=Path.home() / ".cache/rarecell/markers.sqlite",
# #     cache_path=Path.home() / ".cache/rarecell/mcp_knowledge.sqlite",
# # )
# # client = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])
# #
# # drafted = draft_profile_from_prompt(
# #     prompt=(
# #         "Astrocytes in postmortem DLPFC snRNA-seq, with a focus on the "
# #         "SNAP-expressing subset described by Ling 2024 (synaptic + "
# #         "cholesterol synthesis genes; declines in schizophrenia and aging)."
# #     ),
# #     client=client,
# #     session=session,
# #     anchor_paper="38448582",  # Ling et al. 2024 PMID
# # )
# # # Review drafted, set human_reviewed=True, freeze, then use it below.
# ```

# %% [markdown]
# ## 5. Pre-flight: does the profile fit this dataset?
#
# `validate-profile` reports per-panel gene overlap, expression
# prevalence, and panel-score statistics — fast and informative before
# committing to a full isolation run.

# %%
from rarecell.validate import validate_profile_against_adata

report = validate_profile_against_adata(adata, profile)

print(
    f"Dataset: {report['dataset']['n_obs']} cells, "
    f"{report['dataset']['n_vars']} genes\n"
    f"Expected abundance: {report['expected_abundance']}\n"
)
print("Positive panels:")
for name, p in report["positive_markers"].items():
    status = "OK " if p["gene_overlap_fraction"] >= 0.5 else "LOW"
    print(
        f"  [{status}] {name}: {p['gene_overlap_count']}/{p['gene_overlap_total']} "
        f"genes ({p['gene_overlap_fraction']:.0%}), "
        f"prevalence={p['mean_prevalence']:.1%}, "
        f"score={p['score_mean']:+.3f}±{p['score_std']:.3f}"
    )
print("\nNegative panels:")
for name, p in report["negative_markers"].items():
    print(
        f"  {name}: {p['gene_overlap_count']}/{p['gene_overlap_total']} "
        f"genes ({p['gene_overlap_fraction']:.0%})"
    )
print(f"\nOverall: {report['overall_status'].upper()}")

# %% [markdown]
# ## 6. Run isolation

# %%
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner

out_dir = local_root / "run1"
out_dir.mkdir(exist_ok=True)

runner = IsolateRunner(
    adata=adata.copy(),
    profile=profile,
    recommender=BasicRecommender(profile),
    out_dir=out_dir,
    auto_policy="recommendation",
)
result = runner.run()
frac = result.isolated.n_obs / adata.n_obs
print(f"\nIsolated {result.isolated.n_obs} / {adata.n_obs} cells ({frac:.2%})")

# %% [markdown]
# ## 7. Compare against the original brainSCOPE annotations
#
# The dataset arrived with per-cell labels (`cell_type_original`). We can
# now ask: of the cells rarecell isolated, how many were originally
# labeled as astrocytes?

# %%
from collections import Counter

isolated_labels = Counter(result.isolated.obs["cell_type_original"].astype(str))
total_labels = Counter(adata.obs["cell_type_original"].astype(str))
n_astro_total = total_labels.get("Astro", 0)
n_astro_isolated = isolated_labels.get("Astro", 0)
recall = n_astro_isolated / max(n_astro_total, 1)
precision = n_astro_isolated / max(result.isolated.n_obs, 1)

print(f"Original 'Astro' count in dataset: {n_astro_total}")
print(f"Isolated cells originally labeled 'Astro': {n_astro_isolated}")
print(f"\nRecall  (isolated_Astro / total_Astro) = {recall:.2%}")
print(f"Precision (isolated_Astro / isolated_total) = {precision:.2%}")

print("\nFull composition of the isolated subset:")
for label, count in isolated_labels.most_common():
    print(f"  {count:>4} {label}")

# %% [markdown]
# ## 8. Inspect the IsolationReport
#
# rarecell ships a complete reproducibility artifact:

# %%
import json

for p in sorted(out_dir.iterdir()):
    if p.is_file():
        print(f"  {p.name:<20} ({p.stat().st_size:>6} bytes)")

manifest = json.loads((out_dir / "manifest.json").read_text())
print("\nKey manifest fields:")
print(f"  profile_id: {manifest['profile_id']}")
print(f"  profile_content_hash: {manifest['profile_content_hash']}")
print(f"  isolated_summary: {manifest['isolated_summary']}")
print(f"  decision_count: {manifest['decision_count']}")

# %% [markdown]
# ## 9. UMAP of the isolated astrocytes

# %%
import matplotlib.pyplot as plt
import scanpy as sc

sc.settings.set_figure_params(dpi=80, facecolor="white")

isolated = result.isolated.copy()
sc.pp.neighbors(isolated, random_state=0)
sc.tl.umap(isolated, random_state=0)
sc.pl.umap(
    isolated,
    color=["leiden", "cell_type_original"],
    title=[
        f"Isolated cells (n={isolated.n_obs}) — leiden",
        "Original brainSCOPE labels",
    ],
)
plt.show()

# %% [markdown]
# ## 10. Replay determinism

# %%
out_dir2 = local_root / "run2"
out_dir2.mkdir(exist_ok=True)

runner2 = IsolateRunner(
    adata=adata.copy(),
    profile=profile,
    recommender=BasicRecommender(profile),
    out_dir=out_dir2,
    auto_policy="from_decisions",
    replay_decisions_path=result.decisions_path,
)
result2 = runner2.run()
assert (
    result2.isolated.n_obs == result.isolated.n_obs
), f"Replay mismatch: {result2.isolated.n_obs} vs {result.isolated.n_obs}"
print(f"\n✓ Replay deterministic: both isolated {result.isolated.n_obs} cells")
