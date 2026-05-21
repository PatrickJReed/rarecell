import pytest
from fixtures.make_synthetic import make_synthetic


@pytest.fixture
def synthetic_adata():
    return make_synthetic(seed=0)
