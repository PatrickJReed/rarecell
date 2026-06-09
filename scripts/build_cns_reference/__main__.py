"""CLI: python -m scripts.build_cns_reference --out <dir> [--collection <id>] ...

Downloads BICCN WHB H5ADs, normalizes, and builds a CNS reference bundle.
Run from the repo root with the build-reference group synced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rarecell.cns import format as fmt
from rarecell.logging import get_logger

from scripts.build_cns_reference import annotate_abc, annotate_s3, build, discover, stream

log = get_logger("rarecell.build_cns.cli")

WHB_COLLECTION = "283d65eb-dd53-496d-adb7-7570c7caa443"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the CNS reference bundle from BICCN WHB.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--collection", default=WHB_COLLECTION)
    ap.add_argument("--cache-dir", type=Path, default=Path("./biccn_cache"))
    ap.add_argument("--biccn-release", default="WHB-2023")
    ap.add_argument("--cells-per-class", type=int, default=5000)
    # The Siletti WHB atlas has only ~3 donors, so the generic min_donors=10
    # would drop every class. 2 keeps classes seen in >=2 of the 3 donors.
    ap.add_argument("--min-donors", type=int, default=2)
    ap.add_argument("--top-genes", type=int, default=300)
    # Streaming caps: per_file_cap = max cells per cluster taken from each file;
    # max_per_cluster = max cells per cluster accumulated across all files.
    # Sampling is per-cluster so rare clusters get represented for the cluster
    # models (superclusters aggregate up).
    ap.add_argument("--per-file-cap", type=int, default=100)
    ap.add_argument("--max-per-cluster", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    datasets = discover.list_h5ad_datasets(args.collection)
    # The collection ships MULTIPLE orthogonal slicings of the SAME nuclei: by
    # brain region ("Dissection:"), by supercluster ("Supercluster:"), and by
    # neuron/non-neuron ("All neurons" / "All non-neuronal cells"). The
    # Dissection files are the non-overlapping partition covering every cell
    # exactly once with native labels, so we use ONLY those.
    datasets = [d for d in datasets if d.title.startswith("Dissection:")]
    log.info("cli.datasets", n=len(datasets), total_cells=sum(d.cell_count for d in datasets))

    # Memory-bounded streaming subsample (never loads the full ~3.4M-cell atlas).
    atlas = stream.stream_balanced_atlas(
        datasets,
        cache_dir=args.cache_dir,
        per_file_cap=args.per_file_cap,
        max_per_cluster=args.max_per_cluster,
        seed=args.seed,
    )

    # Per-cluster biological annotations sidecar (keyed by the cluster node label).
    annotations: dict[str, dict[str, object]] = {}

    # Allen ABC Atlas: relabel clusters to lineage-grounded names (e.g. "MGE_259")
    # and record per-cluster neurotransmitter. Degrades to numeric cluster_id.
    try:
        amap = annotate_abc.build_annotation_map(annotate_abc.load_membership(args.cache_dir))
        annotate_abc.annotate_atlas(atlas, amap)
        for cl, nt in annotate_abc.cluster_neurotransmitters(atlas).items():
            annotations.setdefault(cl, {})["neurotransmitter"] = nt
        log.info("cli.abc_annotated", n=len(annotations))
    except Exception as e:
        log.warning("cli.abc_annotation_failed", error=str(e))

    # Siletti Table S3: canonical class, subtype, neuropeptide, and curated
    # top-enriched-gene marker panels per cluster.
    try:
        s3 = annotate_s3.cluster_annotations(
            atlas, annotate_s3.build_s3_map(annotate_s3.load_table_s3())
        )
        for cl, ann in s3.items():
            annotations.setdefault(cl, {}).update(ann)
        log.info("cli.s3_annotated", n=len(s3))
    except Exception as e:
        log.warning("cli.s3_annotation_failed", error=str(e))

    build.build_bundle(
        atlas,
        out_dir=args.out,
        biccn_release=args.biccn_release,
        cells_per_class=args.cells_per_class,
        min_donors=args.min_donors,
        top_genes=args.top_genes,
        seed=args.seed,
        check_expression=False,
    )
    if annotations:
        fmt.write_annotations(args.out, annotations)
    log.info("cli.done", out=str(args.out))


if __name__ == "__main__":
    main()
