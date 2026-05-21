# rarecell examples

## `colab_demo.ipynb` — end-to-end demo (Colab-ready)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PatrickJReed/rarecell/blob/main/examples/colab_demo.ipynb)

A self-contained walkthrough of the full **rarecell** pipeline on a public
single-cell RNA-seq dataset (10x PBMC 3k). No API keys required.

Covers:

1. Install `rarecell` from GitHub (subdirectory syntax until PyPI release).
2. Download the dataset via `scanpy.datasets.pbmc3k`.
3. Load + freeze a shipped preset profile (`t_cell_pbmc.yaml`).
4. Run `IsolateRunner` end-to-end.
5. Inspect the `IsolationReport` (manifest, decisions, BibTeX).
6. UMAP-plot the isolated subset.
7. Demonstrate byte-deterministic replay.

Final cell shows how to swap in `ClaudeRecommender` for LLM-driven
decisions (requires the `[agent]` extra and `ANTHROPIC_API_KEY`).

### Local execution

The notebook is paired with `colab_demo.py` via
[jupytext](https://jupytext.readthedocs.io/). Edit either file; the other
is regenerated with:

```bash
uv run jupytext --sync examples/colab_demo.ipynb
```

To execute locally end-to-end:

```bash
uv sync --all-packages --all-extras --dev
uv run python -m ipykernel install --user --name rarecell --display-name "rarecell"
uv run jupyter nbconvert --to notebook --execute examples/colab_demo.ipynb \
  --output /tmp/executed.ipynb --ExecutePreprocessor.kernel_name=rarecell
```
