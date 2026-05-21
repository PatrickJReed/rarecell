import sys
from pathlib import Path

import pytest

# Add tests dir to path so we can import fixtures
sys.path.insert(0, str(Path(__file__).parent))

from fixtures.make_synthetic import make_synthetic


@pytest.fixture
def synthetic_adata():
    return make_synthetic(seed=0)
