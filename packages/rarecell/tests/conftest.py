import sys
from pathlib import Path

# Ensure the top-level tests/fixtures directory is importable
_repo_root = Path(__file__).resolve().parents[3]
_fixtures_path = _repo_root / "tests" / "fixtures"
if str(_fixtures_path) not in sys.path:
    sys.path.insert(0, str(_fixtures_path))
