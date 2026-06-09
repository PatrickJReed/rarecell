# CNS Reference Bundle — Build & Publish Runbook

This runbook is for **maintainers only**. End users never run these steps;
the `rarecell` runtime fetches the finished bundle from the GitHub release
automatically when `reference_release="WHB-2023"` is passed.

---

## Release artifact naming convention

The `ReferenceBundle.resolve()` runtime (`packages/rarecell/src/rarecell/cns/bundle.py`)
derives the download URL from a single `reference_release` string:

```
release tag  : cns-{reference_release}
asset name   : cns-reference-{reference_release}.tar.gz
extracts to  : cns-reference-{reference_release}/   (directory inside the tarball)
full URL     : https://github.com/PatrickJReed/rarecell/releases/download/
               cns-{reference_release}/cns-reference-{reference_release}.tar.gz
```

For the current BICCN release:

| Field        | Value                               |
|---|---|
| `reference_release` | `WHB-2023`                 |
| Release tag  | `cns-WHB-2023`                      |
| Asset name   | `cns-reference-WHB-2023.tar.gz`     |
| Extracts to  | `cns-reference-WHB-2023/`           |

---

## Step 1 — Build the bundle

> **Requires real compute.** Streams ~105 H5AD files (~5–10 GB) from
> CELLxGENE Discover. Runtime: ~30–60 min on Colab free tier, ~15–30 min
> on a machine with fast I/O. Memory stays bounded (never loads the full
> 3.4M-cell atlas).

### Option A — Command line (repo root, build-reference deps synced)

```bash
# Sync the build-reference dependency group (httpx).
uv sync --group build-reference

python -m scripts.build_cns_reference \
    --out ./cns-reference-WHB-2023 \
    --cache-dir ./biccn_cache \
    --biccn-release WHB-2023 \
    --min-donors 2 \
    --per-file-cap 100 \
    --max-per-cluster 1000 \
    --cells-per-class 5000
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--out` | *(required)* | Output directory; will be created. |
| `--cache-dir` | `./biccn_cache` | Where streaming H5ADs are cached. |
| `--biccn-release` | `WHB-2023` | Written into `manifest.json`. |
| `--min-donors` | `2` | The Siletti atlas has ~3 donors; use 2. |
| `--per-file-cap` | `100` | Max cells per cluster per dissection file. |
| `--max-per-cluster` | `1000` | Max cells per cluster across all files. |
| `--cells-per-class` | `5000` | Final balanced training cap per class. |
| `--top-genes` | `300` | HVGs fed to each CellTypist model. |
| `--seed` | `0` | RNG seed for reproducibility. |

Expected outputs in `./cns-reference-WHB-2023/`:

```
manifest.json          # BundleManifest (biccn_release, decisions list)
taxonomy.json          # supercluster -> [clusters] tree
annotations.json       # per-cluster neurotransmitter / class / markers
decisions/
  supercluster.pkl     # 31-way CellTypist model (root gate)
  cluster/<sc>/*.pkl   # per-supercluster cluster models
```

Typical held-out accuracy: supercluster model >0.95; cluster models vary
(common types 0.85–0.99).

### Option B — Colab notebook

Open `examples/build_cns_reference_colab.py` (or the paired `.ipynb`) in
Google Colab and run all cells in order. The notebook mirrors the CLI
exactly. After the build completes, download the tarball with
`files.download("cns-reference-WHB-2023.tar.gz")` (cell §4).

---

## Step 2 — Package as a tarball

Run from the directory **containing** the bundle folder:

```bash
tar -czf cns-reference-WHB-2023.tar.gz cns-reference-WHB-2023
ls -lh cns-reference-WHB-2023.tar.gz   # expect ~50–200 MB
```

The tarball must extract to a top-level `cns-reference-WHB-2023/` directory.
Verify:

```bash
tar -tzf cns-reference-WHB-2023.tar.gz | head -5
# should show: cns-reference-WHB-2023/manifest.json  etc.
```

---

## Step 3 — Publish as a GitHub release

> **Requires** a GitHub token with `repo` scope (or maintainer web access).
> The repo is currently private — the token must have access to
> `PatrickJReed/rarecell`.

### Option A — GitHub web UI

1. Go to `https://github.com/PatrickJReed/rarecell/releases/new`.
2. **Tag:** `cns-WHB-2023` (create new tag on publish).
3. **Title:** `CNS reference bundle WHB-2023`.
4. **Description:** e.g. `BICCN Human Brain Cell Atlas v1.0 supercluster + cluster CellTypist models.`
5. Drag `cns-reference-WHB-2023.tar.gz` into the assets section.
6. Click **Publish release**.

### Option B — gh CLI (from Colab or local)

```bash
# Authenticate once (needs GH_TOKEN with repo scope).
echo "$GH_TOKEN" | gh auth login --with-token

gh release create cns-WHB-2023 cns-reference-WHB-2023.tar.gz \
    --repo PatrickJReed/rarecell \
    --title "CNS reference bundle WHB-2023" \
    --notes "BICCN Human Brain Cell Atlas v1.0 supercluster + cluster CellTypist models."
```

After publishing, confirm the asset URL is reachable:

```bash
curl -fI "https://github.com/PatrickJReed/rarecell/releases/download/cns-WHB-2023/cns-reference-WHB-2023.tar.gz"
# HTTP/2 200  (or 302 redirect — both are fine)
```

---

## Step 4 — Verify the published bundle resolves

Run this snippet (requires no API key; only `rarecell` installed and internet access):

```python
from pathlib import Path
from rarecell.cns.bundle import ReferenceBundle

bundle = ReferenceBundle.resolve(
    "WHB-2023",
    cache_dir=Path("./verify_cache"),
)
manifest = bundle.manifest
print(f"BICCN release : {manifest.biccn_release}")
print(f"Decisions     : {len(manifest.decisions)}")
for d in manifest.decisions[:4]:
    label = d.level if d.parent is None else f"{d.level}/{d.parent}"
    acc = d.metrics.get("heldout_accuracy")
    print(f"  {label:<44} {len(d.classes):>3} classes  heldout_acc={acc:.3f}")
```

Expected output (values approximate):

```
BICCN release : WHB-2023
Decisions     : 32          # 1 supercluster + ~31 per-supercluster cluster models
  supercluster                              31 classes  heldout_acc=0.96x
  cluster/Astrocyte                          N classes  heldout_acc=0.8xx
  ...
```

---

## Step 5 — Fill the README benchmark metric slot

Once the bundle is published and the demo dataset is available, run the
precision/recall benchmark:

```bash
# Score astrocyte isolation from the Colab demo run.
# Assumes:
#   full.h5ad        — full dataset with obs["cell_type_original"]
#   isolated.h5ad    — isolation output (obs_names are the isolated barcodes)
uv run python scripts/benchmark_isolation.py \
    --full    ./full.h5ad \
    --isolated ./rarecell_demo/run1/isolated.h5ad \
    --label-col cell_type_original \
    --target  Astro
```

The script prints both a human-readable summary and a JSON block.
Copy the `precision`, `recall`, and `f1` values into the README metric table.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ReferenceBuildError: Bundle for release 'WHB-2023' not found after download` | Release not published, wrong tag, or private repo + no auth | Verify release tag is exactly `cns-WHB-2023`; check repo visibility / token |
| `tar: cns-reference-WHB-2023/manifest.json: Not found` after extract | Tarball packed from wrong directory (missing top-level dir) | Re-pack: `tar -czf ... cns-reference-WHB-2023` from the parent directory |
| Build OOM on Colab | `--max-per-cluster` or `--cells-per-class` too high | Reduce `--max-per-cluster 500 --cells-per-class 2000` |
| `cli.abc_annotation_failed` warning | Allen ABC Atlas membership CSV not cached | Non-fatal; the bundle builds without neurotransmitter annotations |
