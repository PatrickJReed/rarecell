import pytest
from rarecell_mcp.cli import main_with_args


def test_serve_help_exits_clean():
    with pytest.raises(SystemExit) as exc_info:
        main_with_args(["serve", "--help"])
    assert exc_info.value.code == 0


def test_top_level_help_exits_clean():
    with pytest.raises(SystemExit) as exc_info:
        main_with_args(["--help"])
    assert exc_info.value.code == 0
