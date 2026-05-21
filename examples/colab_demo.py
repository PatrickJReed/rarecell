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
# **The use case.** You read a recent paper describing a novel
# disease-associated cell population. You want to isolate that
# population from your own dataset to study it further. Can rarecell
# translate the paper's description into an isolation profile, then
# extract the matching cells from your data?
#
# **What this notebook does**, on a real public example:
#
# 1. Install `rarecell[agent]` (LLM-driven path) and `remotezip`.
# 2. Stream one SZ DLPFC sample from
#    [brainSCOPE's SZBDMulti-Seq cohort](https://brainscope.gersteinlab.org/) —
#    7,912 pre-annotated cells, ~6% astrocytes. Range requests only;
#    no full archive download.
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
# **Runtime:** ~10 minutes on Colab free tier.
# **Cost:** one drafting call to Claude (claude-opus-4-7, ~$0.05).

# %% [markdown]
# ## 1. Install

# %%
# !pip install -q "git+https://github.com/PatrickJReed/rarecell.git#subdirectory=packages/rarecell-mcp-knowledge"
# !pip install -q "git+https://github.com/PatrickJReed/rarecell.git#subdirectory=packages/rarecell[agent]"
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
# ## 3. Stream one SZ sample from brainSCOPE
#
# `brainscope.gersteinlab.org` hosts per-cohort `.zip` archives
# (each ~1-2.5 GB) containing per-sample annotated TSV matrices. We
# use `remotezip` to fetch only the bytes for one sample
# (~24 MB compressed) without downloading the full archive.

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
    (local_root / SAMPLE_PATH_IN_ZIP).rename(local_sample)
    (local_root / "SZBDMulti-Seq").rmdir()

print(f"Downloaded {local_sample.name} ({local_sample.stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ## 4. Convert TSV to AnnData
#
# brainSCOPE format: rows are genes, columns are cells, with cell-type
# annotations as the first row of the header. We transpose to standard
# AnnData orientation (cells x genes) and preserve the original
# annotations in `obs["cell_type_original"]`.

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
df.columns = [f"cell_{i:04d}" for i in range(df.shape[1])]

adata = ad.AnnData(X=df.T.values.astype(np.float32))
adata.obs_names = list(df.columns)
adata.var_names = list(df.index)
adata.var_names_make_unique()
adata.obs["cell_type_original"] = pd.Categorical(cell_types_original)
adata.obs["sample_id"] = "SZ11"
adata.obs["disease"] = "schizophrenia"
print(f"AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
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
# to 0.2 for this homogeneous benchmark (a dataset-by-dataset tuning).

# %%
profile = draft.model_copy(
    update={
        "human_reviewed": True,
        "reviewer": "colab-demo@example.com",
        "purify": draft.purify.model_copy(update={"min_cluster_purity": 0.2}),
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
assert (
    result2.isolated.n_obs == result.isolated.n_obs
), f"Replay mismatch: {result2.isolated.n_obs} vs {result.isolated.n_obs}"
print(f"\nReplay deterministic: both isolated {result.isolated.n_obs} cells")
