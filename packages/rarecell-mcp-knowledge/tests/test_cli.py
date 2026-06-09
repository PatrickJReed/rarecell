from pathlib import Path

from rarecell_mcp_knowledge.cli import main_with_args

FIXTURES = Path(__file__).parent / "data"


def test_seed_command(tmp_path: Path, capsys):
    db = tmp_path / "home/.cache/rarecell/markers.sqlite"
    rc = main_with_args(
        [
            "seed",
            "--cellmarker-tsv",
            str(FIXTURES / "cellmarker_tiny.tsv"),
            "--panglaodb-tsv",
            str(FIXTURES / "panglaodb_tiny.tsv"),
            "--catalog-path",
            str(db),
        ]
    )
    assert rc == 0
    assert db.exists()
    out = capsys.readouterr().out
    assert "cellmarker" in out
    assert "panglaodb" in out


def test_serve_subcommand_help(capsys):
    import pytest

    # argparse --help exits with SystemExit(0)
    with pytest.raises(SystemExit) as exc_info:
        main_with_args(["serve", "--help"])
    assert exc_info.value.code == 0
