"""rarecell-mcp CLI: serve."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _serve(args: argparse.Namespace) -> int:
    from rarecell_mcp.server import build_fastmcp_app

    app = build_fastmcp_app()
    app.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rarecell-mcp")
    sub = p.add_subparsers(dest="cmd", required=True)
    serve_p = sub.add_parser("serve", help="Run the FastMCP server")
    serve_p.set_defaults(func=_serve)
    return p


def main_with_args(argv: Sequence[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def main() -> int:
    return main_with_args(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
