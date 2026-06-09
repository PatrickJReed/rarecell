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
# # Build the CNS reference bundle (BICCN WHB) on Colab
#
# **Maintainer / one-time task.** This notebook distills the BICCN **Human
# Brain Cell Atlas v1.0** (Siletti et al., *Science* 2023) into a small
# `rarecell` **CNS reference bundle** — a 31-way supercluster CellTypist model
# plus per-supercluster cluster models and marker panels — and packages it as a
# tarball you upload as a GitHub release asset. The `rarecell` runtime then
# fetches that bundle (`reference_release="WHB-2023"`) to run the progressive
# class gate. End users never run this.
#
# **What it does:** streams the ~105 per-dissection H5ADs from the
# [CELLxGENE Discover collection](https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443),
# sampling a bounded number of cells per supercluster from each file (so the
# full ~3.4M-cell atlas is never held in memory), remaps Ensembl IDs to gene
# symbols, trains the models, and writes the bundle.
#
# **Runtime:** ~30–60 min (mostly downloading ~5–10 GB of H5ADs). A standard
# Colab runtime is sufficient (memory stays bounded); high-RAM helps the final
# concat. **Cost:** none (no API keys needed).

# %% [markdown]
# ## 1. Clone the repo and install
#
# The build pipeline lives in `scripts/build_cns_reference/` (repo-only dev
# tooling, not shipped in the wheel), so we clone the repo and install the
# `rarecell` package plus `httpx` (the build-only HTTP client).

# %%
# !git clone --depth 1 https://github.com/PatrickJReed/rarecell.git
# %cd rarecell
# !pip install -q ./packages/rarecell httpx

# %% [markdown]
# ## 2. Build the bundle
#
# Sampling is **per cluster** (the finest level we train), so rare clusters get
# represented for the per-supercluster cluster models; superclusters aggregate
# up. The defaults below are tuned for the Siletti atlas:
#
# - `--min-donors 2` — the atlas has only ~3 donors, so the generic default of
#   10 would drop every class (now the default; shown for clarity).
# - `--per-file-cap 100` — max cells per *cluster* taken from each dissection file.
# - `--max-per-cluster 1000` — max cells per cluster across all files (bounds
#   memory to ~n_clusters x this; never loads the full 3.4M-cell atlas).
# - `--cells-per-class 5000` — final balanced training cap per class.

# %%
# !python -m scripts.build_cns_reference \
#     --out ./cns-reference-WHB-2023 \
#     --cache-dir ./biccn_cache \
#     --biccn-release WHB-2023 \
#     --min-donors 2 \
#     --per-file-cap 100 \
#     --max-per-cluster 1000 \
#     --cells-per-class 5000

# %% [markdown]
# ## 3. Inspect the bundle
#
# Confirm the supercluster decision covers all classes and the per-supercluster
# cluster models were written, and check held-out accuracy per decision.

# %%
from pathlib import Path

from rarecell.cns import format as fmt

bundle = Path("./cns-reference-WHB-2023")
manifest = fmt.load_manifest(bundle)
print(f"BICCN release: {manifest.biccn_release}")
print(f"Decisions: {len(manifest.decisions)}")
for d in manifest.decisions:
    acc = d.metrics.get("heldout_accuracy")
    label = d.level if d.parent is None else f"{d.level}/{d.parent}"
    print(f"  {label:<42} {len(d.classes):>3} classes  heldout_acc={acc:.3f}")

tree = fmt.load_taxonomy(bundle)
print(
    f"\nTaxonomy: {len(tree)} superclusters; Astrocyte clusters: {tree.get('Astrocyte', [])[:6]}..."
)

# Per-cluster biological annotations (Allen ABC neurotransmitter + Siletti
# Table S3 class / subtype / curated marker panel).
ann = fmt.load_annotations(bundle)
print(f"\nAnnotated clusters: {len(ann)}")
for name, a in list(ann.items())[:4]:
    print(
        f"  {name}: class={a.get('class', '?')} nt={a.get('neurotransmitter', '?')} "
        f"markers={a.get('markers', [])[:5]}"
    )

# %% [markdown]
# ## 4. Package the bundle
#
# Tar it up. The runtime expects an asset named
# `cns-reference-WHB-2023.tar.gz` that extracts to `cns-reference-WHB-2023/`.

# %%
# !tar -czf cns-reference-WHB-2023.tar.gz cns-reference-WHB-2023
# !ls -lh cns-reference-WHB-2023.tar.gz

# %% [markdown]
# ## 5. Publish as a GitHub release asset
#
# The runtime resolves `reference_release="WHB-2023"` to
# `https://github.com/PatrickJReed/rarecell/releases/download/cns-WHB-2023/cns-reference-WHB-2023.tar.gz`.
# So the release **tag must be `cns-WHB-2023`** and the **asset name must be
# `cns-reference-WHB-2023.tar.gz`**.
#
# **Option A — download and upload via the web UI (simplest):** download the
# tarball from Colab, then create the release on GitHub and drag the file in.

# %%
from google.colab import files  # type: ignore[import-not-found]

files.download("cns-reference-WHB-2023.tar.gz")

# %% [markdown]
# **Option B — upload from Colab with the `gh` CLI** (needs a token with
# `repo` scope). Uncomment and set your token:
#
# ```bash
# !echo "$GH_TOKEN" | gh auth login --with-token
# !gh release create cns-WHB-2023 cns-reference-WHB-2023.tar.gz \
#     --repo PatrickJReed/rarecell \
#     --title "CNS reference bundle WHB-2023" \
#     --notes "BICCN Human Brain Cell Atlas v1.0 supercluster+cluster CellTypist models."
# ```
#
# Once published, the Colab **demo** notebook's progressive class-gate cell
# (`reference_release="WHB-2023"`) and the `rarecell` runtime will fetch this
# bundle automatically.
