from pathlib import Path

import pytest
from rarecell.profile.schema import TargetCellProfile

PRESETS_DIR = Path(__file__).resolve().parents[2] / "src/rarecell/profile/presets"

PRESET_NAMES = [
    "t_cell_pbmc",
    "t_cell_cns",
    "b_cell",
    "nk_cell",
    "microglia",
    "dendritic_cell",
    "monocyte_macrophage",
]


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_preset_loads(name):
    p = TargetCellProfile.from_yaml_path(PRESETS_DIR / f"{name}.yaml")
    assert p.frozen is False
    assert p.human_reviewed is False
    assert p.positive_markers
    assert len(p.tissue) >= 1
