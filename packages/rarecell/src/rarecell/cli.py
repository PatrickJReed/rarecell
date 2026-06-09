"""rarecell CLI — Typer commands wrapping the public API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import anndata as ad
import typer

from rarecell.profile.schema import TargetCellProfile
from rarecell.recommender.basic import BasicRecommender
from rarecell.state_machine.isolate import IsolateRunner

app = typer.Typer(
    help="rarecell — profile-driven isolation of rare and hard-to-resolve cell populations from single-cell RNA-seq.",
    no_args_is_help=True,
)


def _load_recommender(profile, use_claude: bool):
    if not use_claude:
        return BasicRecommender(profile)
    try:
        from rarecell.agent.client import AnthropicClient
        from rarecell.agent.recommender import ClaudeRecommender
    except ImportError as e:
        raise typer.BadParameter(
            "--use-claude requires the [agent] extra. Install with: pip install 'rarecell[agent]'"
        ) from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise typer.BadParameter("ANTHROPIC_API_KEY env var is required with --use-claude.")
    client = AnthropicClient(api_key=api_key)
    return ClaudeRecommender(profile=profile, client=client)


@app.command()
def isolate(
    input: Annotated[Path, typer.Option("--input", help="Path to input AnnData (.h5ad)")],
    profile_path: Annotated[Path, typer.Option("--profile", help="Path to frozen profile YAML")],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Output directory for IsolationReport")
    ],
    auto_policy: Annotated[
        str,
        typer.Option(
            "--auto-policy",
            help="recommendation | abort_on_ambiguity | conservative_drop | from_decisions",
        ),
    ] = "recommendation",
    use_claude: Annotated[
        bool, typer.Option("--use-claude", help="Use ClaudeRecommender instead of BasicRecommender")
    ] = False,
    decisions: Annotated[
        Path | None,
        typer.Option(
            "--decisions",
            help="Path to decisions.jsonl for replay (with --auto-policy from_decisions)",
        ),
    ] = None,
):
    """Isolate the target population from an AnnData and write an IsolationReport."""
    typer.echo(f"Loading profile from {profile_path}")
    profile = TargetCellProfile.from_yaml_path(profile_path)
    if not profile.frozen:
        raise typer.BadParameter(
            f"Profile at {profile_path} is not frozen. "
            "Call profile.freeze() (requires human_reviewed=True) before running."
        )
    typer.echo(f"Loading AnnData from {input}")
    adata = ad.read_h5ad(input)
    typer.echo(f"  -> {adata.n_obs} cells, {adata.n_vars} genes")

    recommender = _load_recommender(profile, use_claude)
    typer.echo(f"Recommender: {type(recommender).__name__}")

    runner = IsolateRunner(
        adata=adata,
        profile=profile,
        recommender=recommender,
        out_dir=out_dir,
        auto_policy=auto_policy,
        replay_decisions_path=decisions,
    )
    result = runner.run()
    typer.echo(
        f"Isolated {result.isolated.n_obs} cells "
        f"({result.isolated.n_obs / max(adata.n_obs, 1):.2%} of input)"
    )
    typer.echo(f"Report written to {out_dir}")


@app.command()
def draft(
    prompt: Annotated[
        str, typer.Option("--prompt", help="Natural-language description of the target population")
    ],
    out: Annotated[Path, typer.Option("--out", help="Output path for draft profile YAML")],
    catalog_path: Annotated[
        Path | None,
        typer.Option(
            "--catalog-path",
            help="Path to markers SQLite (default: ~/.cache/rarecell/markers.sqlite)",
        ),
    ] = None,
):
    """Draft a TargetCellProfile from a natural-language prompt (requires [agent])."""
    try:
        from rarecell.agent.client import AnthropicClient
        from rarecell.agent.draft import draft_profile_from_prompt
        from rarecell.rag.knowledge import build_knowledge_session
    except ImportError as e:
        raise typer.BadParameter(
            "draft requires the [agent] extra. Install with: pip install 'rarecell[agent]'"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise typer.BadParameter("ANTHROPIC_API_KEY env var is required for draft.")

    catalog = catalog_path or (Path.home() / ".cache/rarecell/markers.sqlite")
    cache = Path.home() / ".cache/rarecell/mcp_knowledge.sqlite"
    session = build_knowledge_session(catalog_path=catalog, cache_path=cache)
    client = AnthropicClient(api_key=api_key)
    profile = draft_profile_from_prompt(prompt=prompt, client=client, session=session)
    profile.to_yaml_path(out)
    typer.echo(f"Draft profile written to {out}")
    typer.echo("Review it, set human_reviewed=true and provide reviewer, then freeze.")


@app.command()
def review(
    report: Annotated[Path, typer.Option("--report", help="Path to an IsolationReport directory")],
):
    """Print a summary of an IsolationReport."""
    manifest_path = report / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"No manifest.json at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    typer.echo(f"Report: {manifest['run_id']}")
    typer.echo(f"  Started:  {manifest['started_at']}")
    typer.echo(f"  Finished: {manifest['finished_at']}")
    typer.echo(f"  Profile:  {manifest['profile_id']} ({manifest['profile_content_hash']})")
    typer.echo(f"  Input:    {manifest['input_summary']['n_cells']} cells")
    typer.echo(
        f"  Isolated: {manifest['isolated_summary']['n_cells']} cells "
        f"({manifest['isolated_summary']['abundance_fraction']:.4f} fraction)"
    )
    typer.echo(
        f"  Within expected bounds: {manifest['isolated_summary']['within_expected_bounds']}"
    )
    typer.echo(f"  Decisions: {manifest['decision_count']}")
    typer.echo(f"  Status:    {manifest['status']}")


@app.command("validate-profile")
def validate_profile(
    input: Annotated[Path, typer.Option("--input", help="Path to input AnnData (.h5ad)")],
    profile_path: Annotated[
        Path, typer.Option("--profile", help="Path to profile YAML (frozen or not)")
    ],
):
    """Pre-flight check: does the profile's marker panel fit this dataset?

    Reports per-panel gene overlap (%), mean per-gene prevalence, and panel
    score statistics. Exits 0 if all positive panels have >=50% gene overlap;
    exits 1 if any panel is below the threshold.
    """
    from rarecell.validate import validate_profile_against_adata

    typer.echo(f"Loading profile from {profile_path}")
    profile = TargetCellProfile.from_yaml_path(profile_path)
    typer.echo(f"Loading AnnData from {input}")
    adata = ad.read_h5ad(input)

    report = validate_profile_against_adata(adata, profile)

    typer.echo("\n=== Dataset ===")
    ds = report["dataset"]
    typer.echo(f"  n_obs={ds['n_obs']}  n_vars={ds['n_vars']}  samples={ds['samples']}")
    ea = report["expected_abundance"]
    typer.echo(f"  Expected abundance: [{ea['min_fraction']:.4f}, {ea['max_fraction']:.4f}]")

    typer.echo("\n=== Positive marker panels ===")
    for name, p in report["positive_markers"].items():
        status = "OK " if p["gene_overlap_fraction"] >= 0.5 else "LOW"
        typer.echo(
            f"  [{status}] {name}: "
            f"{p['gene_overlap_count']}/{p['gene_overlap_total']} genes found "
            f"({p['gene_overlap_fraction']:.0%}), "
            f"mean prevalence={p['mean_prevalence']:.2%}, "
            f"score={p['score_mean']:+.3f}±{p['score_std']:.3f}"
        )
        if p["genes_missing"]:
            typer.echo(f"        missing: {', '.join(p['genes_missing'])}")

    if report["negative_markers"]:
        typer.echo("\n=== Negative marker panels ===")
        for name, p in report["negative_markers"].items():
            typer.echo(
                f"  {name}: "
                f"{p['gene_overlap_count']}/{p['gene_overlap_total']} genes found "
                f"({p['gene_overlap_fraction']:.0%}), "
                f"mean prevalence={p['mean_prevalence']:.2%}"
            )

    typer.echo(f"\nOverall: {report['overall_status'].upper()}")

    if report["overall_status"] != "pass":
        typer.echo(
            "\nAt least one positive panel has <50% gene overlap with the dataset. "
            "Likely causes: gene-name format mismatch (Ensembl vs symbol), "
            "cross-species names, or markers genuinely absent from this tissue."
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
