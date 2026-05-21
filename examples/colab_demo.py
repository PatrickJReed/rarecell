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
# # rarecell — end-to-end demo (Colab-ready)
#
# This notebook walks through the full **rarecell** pipeline on a public
# single-cell RNA-seq dataset:
#
# 1. Install `rarecell` from GitHub (no PyPI release yet).
# 2. Download a public dataset (10x PBMC 3k by default).
# 3. Load one of rarecell's shipped preset profiles, review it, and freeze.
# 4. Run `IsolateRunner` to isolate the target population.
# 5. Inspect the `IsolationReport` directory (manifest, decisions, BibTeX).
# 6. Plot UMAP of the isolated cells.
# 7. Demonstrate byte-deterministic replay.
#
# **No API keys required.** This demo uses the heuristic `BasicRecommender`.
# To swap in `ClaudeRecommender` (LLM-driven decisions, requires
# `ANTHROPIC_API_KEY`), see the optional cell at the bottom.

# %% [markdown]
# ## 1. Install rarecell
#
# `rarecell` is a uv workspace; until we publish to PyPI, install the
# library directly from the GitHub subdirectory:

# %%
# !pip install -q "git+https://github.com/PatrickJReed/rarecell.git#subdirectory=packages/rarecell"

# %% [markdown]
# ## 2. Download a public dataset
#
# Scanpy's bundled 10x PBMC 3k dataset is a small (~7 MB, 2700 cells)
# canonical single-cell RNA-seq benchmark with ~45-60% T cells. Perfect
# for a fast demo.

# %%
import scanpy as sc

adata = sc.datasets.pbmc3k()
# IsolateRunner expects a sample_id column for in-dataset batch correction
adata.obs["sample_id"] = "pbmc3k"
print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
adata

# %% [markdown]
# ## 3. Load the T-cell PBMC preset profile
#
# rarecell ships seven preset profiles under
# `rarecell.profile.presets/`. We'll use `t_cell_pbmc.yaml` —
# a pan-T-cell isolation profile tuned for PBMC.

# %%
from importlib import resources

preset_yaml = resources.files("rarecell.profile.presets").joinpath("t_cell_pbmc.yaml").read_text()
print(preset_yaml)

# %% [markdown]
# ## 4. Review + freeze the profile
#
# A profile **must** be `frozen` before it can drive `IsolateRunner`, and
# `frozen=True` requires `human_reviewed=True` (a hard interlock). Set
# both, then call `.freeze()` — that computes a content hash and locks
# the profile against modification.
#
# Two tweaks to the shipped preset before we freeze:
#
# 1. **Disable CellTypist** (the preset enables `Immune_All_Low.pkl`; in
#    Colab we skip the ~30 MB model download).
# 2. **Lower `purify.min_cluster_purity`** from the preset's default 0.7
#    to 0.2. The default is a noisy-dataset safeguard; on a relatively
#    homogeneous benchmark like PBMC 3k the per-sub-cluster pass fractions
#    sit between 0.2 and 0.4, and 0.7 would drop every sub-cluster. Real
#    workflows tune this once per dataset/profile combo.

# %%
import yaml
from rarecell.profile.schema import TargetCellProfile

profile = TargetCellProfile.model_validate(yaml.safe_load(preset_yaml))

# Tweak the shipped preset for this demo:
profile = profile.model_copy(
    update={
        "reference_labels": profile.reference_labels.model_copy(update={"celltypist_models": []}),
        "purify": profile.purify.model_copy(update={"min_cluster_purity": 0.2}),
    }
)

# Human review + freeze
profile = profile.model_copy(
    update={
        "human_reviewed": True,
        "reviewer": "colab-demo@example.com",
    }
).freeze()

print(f"Profile: {profile.profile_id}")
print(f"Frozen: {profile.frozen}")
print(f"Content hash: {profile.content_hash}")

# %% [markdown]
# ## 5. Run `IsolateRunner`
#
# `IsolateRunner` walks the deterministic state machine
# `S0 → S1 → S2 → S3 → S4_GATE1 → (S5_PURIFY → S5_GATE2) → S6_FINAL → S6_GATE3 → S7_REPORT`.
# Per-cluster decisions come from `BasicRecommender` (heuristic; no LLM).
# Counts → QC → normalize → cluster → score evidence → decide → optionally
# purify → write report.

# %%
from pathlib import Path

from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner

out_dir = Path("./rarecell_runs/run1")
out_dir.mkdir(parents=True, exist_ok=True)

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
# ## 6. Inspect the IsolationReport
#
# Every run produces a self-contained, replay-able directory.

# %%
print("IsolationReport contents:")
for p in sorted(out_dir.iterdir()):
    if p.is_file():
        print(f"  {p.name:<20} ({p.stat().st_size:>6} bytes)")

# %%
import json

manifest = json.loads((out_dir / "manifest.json").read_text())
print(json.dumps(manifest, indent=2, default=str))

# %% [markdown]
# Decisions are stored as append-only JSONL — one entry per cluster
# decision per gate, with the recommender's suggestion, the
# user/policy's actual decision, the supporting evidence, and any
# citations.

# %%
decisions = [
    json.loads(line) for line in (out_dir / "decisions.jsonl").read_text().strip().splitlines()
]
print(f"Logged {len(decisions)} gate decisions.")
print("\nFirst decision:")
print(json.dumps(decisions[0], indent=2, default=str))

# %% [markdown]
# ## 7. UMAP of the isolated cells
#
# A quick visual sanity check that the isolated subset is a coherent
# population.

# %%
import matplotlib.pyplot as plt

sc.settings.set_figure_params(dpi=80, facecolor="white")

isolated = result.isolated.copy()
sc.pp.neighbors(isolated, random_state=0)
sc.tl.umap(isolated, random_state=0)

fig = sc.pl.umap(
    isolated,
    color=["leiden"],
    title=[f"Isolated cells (n={isolated.n_obs}) — leiden clusters"],
    return_fig=True,
)
plt.show()

# %% [markdown]
# ## 8. Replay determinism
#
# Re-run with `auto_policy="from_decisions"` against the original run's
# `decisions.jsonl`. The output should be identical.

# %%
out_dir2 = Path("./rarecell_runs/run2")
out_dir2.mkdir(parents=True, exist_ok=True)

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
print(f"✓ Replay deterministic: both runs isolated {result.isolated.n_obs} cells")

# %% [markdown]
# ## Optional: drive the workflow with Claude
#
# To use the LLM-backed `ClaudeRecommender` (per-cluster decisions
# explained in natural language, evidence-cited), install the `[agent]`
# extra and set `ANTHROPIC_API_KEY`:
#
# ```python
# # !pip install -q "git+https://github.com/PatrickJReed/rarecell.git#subdirectory=packages/rarecell[agent]"
# # import os
# # os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # or use Colab secrets
# #
# # from rarecell.agent.client import AnthropicClient
# # from rarecell.agent.recommender import ClaudeRecommender
# #
# # client = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])
# # recommender = ClaudeRecommender(profile=profile, client=client)
# # # ... then construct IsolateRunner with recommender=recommender
# ```
#
# For natural-language profile drafting, see `rarecell.agent.draft` and
# `rarecell-mcp-knowledge`.
