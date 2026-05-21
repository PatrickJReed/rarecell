# rarecell examples

## `colab_demo.ipynb` — paper-to-cells demo (Colab-ready)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PatrickJReed/rarecell/blob/main/examples/colab_demo.ipynb)

A self-contained walkthrough answering the question *"can I isolate a
disease-associated cell population from my dataset using markers from a
specific paper?"*

- **Paper:** Ling et al., _Nature_ 2024 — _A concerted neuron-astrocyte
  program declines in ageing and schizophrenia_
  ([PMID:38448582](https://pubmed.ncbi.nlm.nih.gov/38448582/)).
- **Dataset:** one SZ DLPFC sample from
  [brainSCOPE's SZBDMulti-Seq cohort](https://brainscope.gersteinlab.org/) —
  7,912 pre-annotated cells, ~6% astrocytes. Streamed via remote-zip
  Range requests (no full archive download).

End-to-end runtime on Colab free tier: ~10 minutes. No API keys required.

The notebook covers:

1. Install `rarecell` from GitHub.
2. Stream one SZ sample's annotated matrix from brainSCOPE and convert
   to AnnData.
3. Compose a Ling-anchored astrocyte profile — canonical astrocyte
   markers plus Ling's SNAP signature genes (synaptic + cholesterol
   synthesis). Optional cell shows how to let Claude draft this profile
   from the paper's abstract using `draft_profile_from_prompt(anchor_paper=PMID)`.
4. `rarecell validate-profile` — pre-flight: do the paper's markers exist in
   your dataset?
5. `IsolateRunner` runs the full pipeline.
6. Compare isolated cells against the original brainSCOPE annotations —
   ~99% recall on the `Astro` label.
7. Inspect the `IsolationReport` (manifest, decisions, BibTeX).
8. UMAP-plot the isolated subset.
9. Demonstrate byte-deterministic replay.

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
