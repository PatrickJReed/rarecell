"""rarecell-mcp-knowledge CLI: serve | seed."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _default_catalog_path() -> Path:
    return Path.home() / ".cache/rarecell/markers.sqlite"


def _default_cache_path() -> Path:
    return Path.home() / ".cache/rarecell/mcp_knowledge.sqlite"


def _seed(args: argparse.Namespace) -> int:
    from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
    from rarecell_mcp_knowledge.markers.seed import seed_catalog_from_tsv

    catalog = MarkersCatalog(Path(args.catalog_path))
    counts = seed_catalog_from_tsv(
        catalog,
        cellmarker_tsv=Path(args.cellmarker_tsv) if args.cellmarker_tsv else None,
        panglaodb_tsv=Path(args.panglaodb_tsv) if args.panglaodb_tsv else None,
    )
    print(f"Seeded: cellmarker={counts['cellmarker']} panglaodb={counts['panglaodb']}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog
    from rarecell_mcp_knowledge.server import build_fastmcp_app

    catalog = MarkersCatalog(Path(args.catalog_path))
    app = build_fastmcp_app(catalog=catalog, cache_path=Path(args.cache_path))
    app.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rarecell-mcp-knowledge")
    sub = p.add_subparsers(dest="cmd", required=True)

    serve_p = sub.add_parser("serve", help="Run the FastMCP server")
    serve_p.add_argument("--catalog-path", default=str(_default_catalog_path()))
    serve_p.add_argument("--cache-path", default=str(_default_cache_path()))
    serve_p.set_defaults(func=_serve)

    seed_p = sub.add_parser("seed", help="Seed the marker catalog from TSV files")
    seed_p.add_argument("--catalog-path", default=str(_default_catalog_path()))
    seed_p.add_argument(
        "--cellmarker-tsv", default=None, help="Path to CellMarker 2.0 TSV; omit to skip"
    )
    seed_p.add_argument("--panglaodb-tsv", default=None, help="Path to PanglaoDB TSV; omit to skip")
    seed_p.set_defaults(func=_seed)

    return p


def main_with_args(argv: Sequence[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def main() -> int:
    return main_with_args(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
