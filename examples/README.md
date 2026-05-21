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

End-to-end runtime on Colab free tier: ~10 minutes. Requires an Anthropic
API key (≈$0.05 for the single drafting call).

The notebook covers:

1. Install `rarecell[agent]`, `rarecell-mcp-knowledge`, `remotezip`.
2. Set up your Anthropic API key (Colab Secret recommended; see below).
3. Stream one SZ sample's annotated matrix from brainSCOPE and convert
   to AnnData.
4. Hand the **paper's PMID** to
   `draft_profile_from_prompt(anchor_paper="38448582", ...)`. Claude
   fetches the paper's abstract, composes a `TargetCellProfile`, and we
   display the drafted YAML for review.
5. Review, edit if needed, freeze.
6. `rarecell validate-profile` pre-flight: do the paper's markers exist
   in your dataset?
7. `IsolateRunner` runs the full pipeline.
8. Compare isolated cells against the original brainSCOPE annotations.
9. UMAP + manifest + byte-deterministic replay.

### Anthropic API key — recommended setup in Colab

1. Click the 🔑 key icon in the Colab sidebar ("Secrets").
2. Add a secret named `ANTHROPIC_API_KEY` with your `sk-ant-...` value.
3. Toggle **Notebook access** on for that secret. Colab won't expose it
   otherwise — this is the gate.

Without a Secret the notebook falls back to a masked `getpass` prompt.

### Best practices when sharing notebooks publicly

- Never hardcode the key in cell source. Never set
  `os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."` either — that string
  lands in cell outputs.
- Never `print(api_key)` for debugging. Print `bool(api_key)` or the
  first 4 chars at most.
- Clear cell outputs before sharing. Better: structure the code so the
  key never appears in any cell's output (this demo prints
  `"ANTHROPIC_API_KEY: configured (N chars)"` instead of the value).
- Use `getpass.getpass()` not `input()` for fallback prompts. Masked,
  doesn't show in screen recordings.
- Spin up a single-purpose, revocable key for the demo. Anthropic keys
  are bearer tokens — anyone with the key spends on your account.
- HuggingFace Spaces / GitHub Actions: use platform-native secret stores
  (Spaces Settings → Variables and Secrets; GitHub Actions secrets).
  Same fallback code reads `os.environ`.

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
