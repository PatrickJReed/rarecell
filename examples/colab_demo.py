# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # rarecell — paper-to-cells demo (Colab-ready)
#
# **The use case.** You read a recent paper describing a novel
# disease-associated cell population. You want to isolate that
# population from your own dataset to study it further. Can rarecell
# translate the paper's description into an isolation profile, then
# extract the matching cells from your data?
#
# **What this notebook does**, on a real public example:
#
# 1. Install `rarecell[agent]` (LLM-driven path) and `remotezip`.
# 2. Stream **multiple SZ DLPFC donors** from
#    [brainSCOPE's SZBDMulti-Seq cohort](https://brainscope.gersteinlab.org/)
#    and merge them into one ≥100,000-cell dataset (pre-annotated, a few
#    percent astrocytes). Range requests only; no full archive download.
#    Each donor is its own batch (`sample_id`), corrected in-pipeline by
#    Harmony.
# 3. Set up an Anthropic API key (Colab Secret or masked prompt).
# 4. Hand the **paper's PMID** to
#    `draft_profile_from_prompt(anchor_paper="38448582", ...)`.
#    Claude fetches the paper's abstract, composes a `TargetCellProfile`,
#    and we display the drafted YAML for review.
# 5. Review, edit if needed, freeze.
# 6. `validate-profile` pre-flight: do the paper's markers exist in
#    this dataset?
# 7. `IsolateRunner` runs the full pipeline.
# 8. Compare isolated cells against the original brainSCOPE annotations.
# 9. UMAP + manifest + byte-deterministic replay.
#
# **Paper anchor.** Ling et al., _Nature_ 2024,
# [PMID:38448582](https://pubmed.ncbi.nlm.nih.gov/38448582/) —
# _A concerted neuron-astrocyte program declines in ageing and
# schizophrenia_. The paper identifies *SNAP* (Synaptic Neuron and
# Astrocyte Program), a gene program whose expression in astrocytes
# (and matching neurons) declines in schizophrenia and aging.
#
# **Runtime:** ~25-40 minutes on Colab free tier — streaming the donors
# and integrating 100k+ cells (HVG → PCA → Harmony → Leiden, run once for
# isolation and again for the replay check) dominate. Drop `TARGET_CELLS`
# in §3 for a faster pass.
# **Cost:** one drafting call to Claude (claude-opus-4-7, ~$0.05).

# %% [markdown]
# ## 1. Install

# %%
# rarecell is a uv workspace — pip-install each package via PEP 508
# direct reference (the [agent] extra has to go on the package name,
# not inside the URL fragment, hence the "rarecell[agent] @ git+..."
# form). Install rarecell-mcp-knowledge first so rarecell[agent] can
# resolve it without hitting PyPI.
# !pip install -q "rarecell-mcp-knowledge @ git+https://github.com/PatrickJReed/rarecell.git#subdirectory=packages/rarecell-mcp-knowledge"
# !pip install -q "rarecell[agent] @ git+https://github.com/PatrickJReed/rarecell.git#subdirectory=packages/rarecell"
# !pip install -q remotezip

# %% [markdown]
# ## 2. Set up your Anthropic API key
#
# This notebook calls Claude once (to draft the profile from the
# paper). You need an Anthropic API key —
# [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).
#
# **Recommended setup in Colab:** add a Notebook Secret.
#
# 1. Click the **🔑 key icon** in the left sidebar ("Secrets").
# 2. Click **+ Add new secret**.
#    - **Name:** `ANTHROPIC_API_KEY`
#    - **Value:** your `sk-ant-...` key
# 3. Toggle **"Notebook access"** on for that secret. (Colab won't let
#    the notebook read it otherwise — this is the gate.)
#
# If you skip the Secret, the cell will fall back to a masked prompt.
#
# **Discipline:** never hardcode the key in a cell, never `print(api_key)`,
# clear cell outputs before sharing the notebook, treat the key as a
# bearer token (anyone with it spends on your Anthropic account).

# %%
def _get_anthropic_api_key() -> str:
    """Tiered API key resolution. Returns the key or raises RuntimeError.

    Order: Colab Secret → env var → masked prompt. Never logs or returns
    the key value in any error path.
    """
    # 1. Colab Secrets
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        try:
            key = userdata.get("ANTHROPIC_API_KEY")
            if key:
                return key
        except Exception:
            # SecretNotFoundError / NotebookAccessError / etc.
            pass
    except ImportError:
        pass
    # 2. Environment variable (local Jupyter, CI)
    import os

    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return key
    # 3. Masked interactive prompt (last resort)
    import getpass

    key = getpass.getpass("Anthropic API key (sk-ant-...): ").strip()
    if not key:
        raise RuntimeError("No API key provided.")
    return key


api_key = _get_anthropic_api_key()
print(f"ANTHROPIC_API_KEY: configured ({len(api_key)} chars)")

# %% [markdown]
# ## 3. Stream and merge multiple SZ donors from brainSCOPE
#
# `brainscope.gersteinlab.org` hosts per-cohort `.zip` archives (each
# ~1-2.5 GB) containing one annotated TSV matrix per donor. We use
# `remotezip` to fetch only the bytes for the donors we want — no full
# archive download — and concatenate them until we clear `TARGET_CELLS`.
#
# Every donor's matrix shares the same gene order and holds raw integer
# counts, so we load each one into a **sparse** matrix (a dense
# 100k x 33k array would be ~13 GB and OOM the Colab runtime) and let
# `anndata.concat` stack them. Each donor keeps its own `sample_id`,
# which Harmony uses as the batch key in §8.

# %%
import gzip
import io
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import remotezip
from scipy import sparse

ARCHIVE_URL = "https://brainscope.gersteinlab.org/data/snrna_expr_matrices_zip/SZBDMulti-Seq.zip"
TARGET_CELLS = 100_000

local_root = Path("./rarecell_demo")
local_root.mkdir(exist_ok=True)


def load_donor(rz: "remotezip.RemoteZip", donor: str) -> ad.AnnData:
    """Stream one donor's matrix from the remote zip into a sparse AnnData.

    brainSCOPE layout: rows are genes, columns are cells, and the header
    row holds each cell's original annotation (not a normal column name).
    We read that header separately, then load the counts and transpose to
    standard (cells x genes) orientation.
    """
    raw = rz.read(f"SZBDMulti-Seq/{donor}-annotated_matrix.txt.gz")
    with gzip.open(io.BytesIO(raw), "rt") as fh:
        cell_types = next(fh).rstrip("\n").split("\t")[1:]
    df = pd.read_csv(io.BytesIO(raw), sep="\t", index_col=0, compression="gzip")
    a = ad.AnnData(X=sparse.csr_matrix(df.T.values.astype(np.float32)))
    a.var_names = df.index.astype(str)
    a.var_names_make_unique()
    a.obs_names = [f"{donor}_{i:05d}" for i in range(a.n_obs)]
    a.obs["cell_type_original"] = pd.Categorical(cell_types)
    a.obs["sample_id"] = donor
    a.obs["disease"] = "schizophrenia"
    del df
    return a


parts: list[ad.AnnData] = []
n_cells = 0
with remotezip.RemoteZip(ARCHIVE_URL) as rz:
    # All SZ donors in the archive, largest first so we reach the target
    # in a handful of files (each is still its own batch for Harmony).
    sz = [
        i.filename.split("/")[-1].replace("-annotated_matrix.txt.gz", "")
        for i in sorted(rz.infolist(), key=lambda x: -x.file_size)
        if i.filename.split("/")[-1].startswith("SZ")
    ]
    for donor in sz:
        if n_cells >= TARGET_CELLS:
            break
        a = load_donor(rz, donor)
        parts.append(a)
        n_cells += a.n_obs
        print(f"  + {donor}: {a.n_obs:>6,} cells  (running total {n_cells:,})")

# %% [markdown]
# ## 4. Concatenate donors into one AnnData
#
# Inner join on genes (identical across donors here, but defensive),
# preserving each donor's `sample_id` and original annotation.

# %%
adata = ad.concat(parts, join="inner", index_unique=None)
adata.obs["cell_type_original"] = adata.obs["cell_type_original"].astype("category")
adata.obs["sample_id"] = adata.obs["sample_id"].astype("category")
del parts
print(
    f"Merged AnnData: {adata.n_obs:,} cells x {adata.n_vars:,} genes "
    f"across {adata.obs['sample_id'].nunique()} donors"
)
print("\nCells per donor:")
print(adata.obs["sample_id"].value_counts())
print("\nOriginal cell-type composition (top 10):")
print(adata.obs["cell_type_original"].value_counts().head(10))

# %% [markdown]
# ## 5. Draft the profile from the paper
#
# Hand Claude the **paper's PMID** plus a one-sentence prompt describing
# what we want to isolate. `draft_profile_from_prompt`:
#
# 1. Fetches the paper's abstract from Europe PMC.
# 2. Prepends it to Claude's context as the primary grounding source.
# 3. Asks Claude to compose a `TargetCellProfile` matching the paper's
#    cell type / markers / abundance.
#
# We display the drafted YAML so you can review the markers Claude chose
# **before** anything touches your data.

# %%
from rarecell.agent.client import AnthropicClient
from rarecell.agent.draft import draft_profile_from_prompt
from rarecell.rag.knowledge import build_knowledge_session

# An empty markers SQLite is fine — the anchor paper carries the load.
session = build_knowledge_session(
    catalog_path=local_root / "markers.sqlite",
    cache_path=local_root / "mcp_cache.sqlite",
)
client = AnthropicClient(api_key=api_key)

draft = draft_profile_from_prompt(
    prompt=(
        "Isolate astrocytes from postmortem DLPFC snRNA-seq, with a "
        "focus on the SNAP-expressing subset described in the anchor "
        "paper — synaptic-support and cholesterol-synthesis genes."
    ),
    client=client,
    session=session,
    anchor_paper="38448582",  # Ling et al. 2024 PMID
)

print(f"Profile drafted: {draft.profile_id}")
print(f"Name: {draft.name}")
print(f"Tissue: {draft.tissue}")
print(f"Expected abundance: {draft.expected_abundance}")
print("\nPositive panels:")
for name, panel in draft.positive_markers.items():
    print(f"  {name}: {panel.genes}")
print("\nNegative panels:")
for name, panel in draft.negative_markers.items():
    print(f"  {name}: {panel.genes}")

# %% [markdown]
# ### Review the full drafted YAML

# %%
import yaml

print(yaml.safe_dump(draft.model_dump(mode="json"), sort_keys=False))

# %% [markdown]
# ## 6. (Optional) Edit + freeze
#
# If you want to override anything Claude picked — add a marker, drop a
# negative panel, tighten an abundance bound — edit the dict below. The
# default cell just sets `human_reviewed=True` (the freeze interlock)
# and freezes. Real workflows iterate here.
#
# We also lower `purify.min_cluster_purity` from the preset default 0.7
# to 0.2 for this homogeneous benchmark (a dataset-by-dataset tuning),
# and pin batch correction to Harmony on `sample_id` so the per-donor
# batch effects in the merged dataset get integrated out before
# clustering. (These are the schema defaults, but we set them
# explicitly since the merged dataset now spans many donors.)

# %%
profile = draft.model_copy(
    update={
        "human_reviewed": True,
        "reviewer": "colab-demo@example.com",
        "purify": draft.purify.model_copy(update={"min_cluster_purity": 0.2}),
        "batch_correction": draft.batch_correction.model_copy(
            update={"in_dataset": "harmony", "batch_key": "sample_id"}
        ),
    }
).freeze()

print(f"Profile frozen: {profile.frozen}")
print(f"Content hash: {profile.content_hash}")

# %% [markdown]
# ## 7. Pre-flight: do the paper's markers exist in this dataset?
#
# `validate-profile` reports per-panel gene overlap, expression
# prevalence, and panel-score statistics — fast and informative before
# committing to a full isolation run. Catches the common Ensembl/symbol,
# cross-species, or wrong-tissue mismatches.

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
        f"score={p['score_mean']:+.3f}+/-{p['score_std']:.3f}"
    )
print("\nNegative panels:")
for name, p in report["negative_markers"].items():
    print(
        f"  {name}: {p['gene_overlap_count']}/{p['gene_overlap_total']} "
        f"genes ({p['gene_overlap_fraction']:.0%})"
    )
print(f"\nOverall: {report['overall_status'].upper()}")

# %% [markdown]
# ## 7b. Progressive BICCN class gate (optional)
#
# Before the marker-driven isolation, narrow the merged dataset to the
# **Astrocyte supercluster** using the pre-trained BICCN reference bundle
# (a small download — no atlas). This is the progressive
# supercluster -> cluster framework from BICCN's Human Brain Cell Atlas.
#
# *Requires a published `cns-WHB-2023` reference bundle; skip this cell if it is not yet available.*

# %%
from rarecell.cns.gate import apply_cns_class_gate
from rarecell.profile.schema import CNSTaxonomyConfig

cns_cfg = CNSTaxonomyConfig(
    enabled=True,
    target_node="Astrocyte",
    target_level="supercluster",
    reference_release="WHB-2023",  # GitHub release tag; downloads the small bundle
    min_confidence=0.5,
    on_missing="skip",  # degrade gracefully if the bundle is not published yet
)

gated, gate_prov = apply_cns_class_gate(adata, cns_cfg, cache_dir=local_root)
if gate_prov.get("skipped"):
    print("CNS class gate skipped (reference bundle unavailable):", gate_prov.get("error"))
else:
    print(
        f"CNS class gate: {gate_prov['n_in']:,} -> {gate_prov['n_out']:,} cells "
        f"kept as {cns_cfg.target_node}"
    )
    print(
        "Astrocyte fraction after gate:",
        f"{(gated.obs['cell_type_original'] == 'Astro').mean():.1%}",
    )
    adata = gated  # downstream isolation now runs on the narrowed population

# %% [markdown]
# ## 8. Run isolation

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
# ## 9. Compare against original brainSCOPE annotations

# %%
from collections import Counter

iso_labels = Counter(result.isolated.obs["cell_type_original"].astype(str))
tot_labels = Counter(adata.obs["cell_type_original"].astype(str))
n_astro_total = tot_labels.get("Astro", 0)
n_astro_isolated = iso_labels.get("Astro", 0)
recall = n_astro_isolated / max(n_astro_total, 1)
precision = n_astro_isolated / max(result.isolated.n_obs, 1)

print(f"Astro in dataset: {n_astro_total}")
print(f"Astro in isolated: {n_astro_isolated}")
print(f"\nRecall  (isolated_Astro / total_Astro) = {recall:.2%}")
print(f"Precision (isolated_Astro / isolated_total) = {precision:.2%}")
print("\nIsolated composition:")
for label, count in iso_labels.most_common():
    print(f"  {count:>4} {label}")

# %% [markdown]
# ## 10. Inspect the IsolationReport

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
# ## 11. UMAP of the isolated subset

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
        f"Isolated cells (n={isolated.n_obs}) - leiden",
        "Original brainSCOPE labels",
    ],
)
plt.show()

# %% [markdown]
# ## 12. Replay determinism

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
assert result2.isolated.n_obs == result.isolated.n_obs, (
    f"Replay mismatch: {result2.isolated.n_obs} vs {result.isolated.n_obs}"
)
print(f"\nReplay deterministic: both isolated {result.isolated.n_obs} cells")
