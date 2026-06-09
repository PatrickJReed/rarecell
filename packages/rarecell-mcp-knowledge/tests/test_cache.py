import time
from pathlib import Path

from rarecell_mcp_knowledge.cache import QueryCache


def test_cache_get_miss(tmp_path: Path):
    c = QueryCache(tmp_path / "cache.sqlite")
    assert c.get("europepmc", "foo") is None


def test_cache_set_then_get(tmp_path: Path):
    c = QueryCache(tmp_path / "cache.sqlite")
    c.set("europepmc", "foo", {"hits": [1, 2, 3]}, ttl_seconds=60)
    out = c.get("europepmc", "foo")
    assert out == {"hits": [1, 2, 3]}


def test_cache_expires(tmp_path: Path):
    c = QueryCache(tmp_path / "cache.sqlite")
    c.set("europepmc", "foo", {"x": 1}, ttl_seconds=0)  # immediate expiry
    time.sleep(0.01)
    assert c.get("europepmc", "foo") is None


def test_cache_overwrites_existing_key(tmp_path: Path):
    c = QueryCache(tmp_path / "cache.sqlite")
    c.set("europepmc", "foo", {"x": 1}, ttl_seconds=60)
    c.set("europepmc", "foo", {"x": 2}, ttl_seconds=60)
    assert c.get("europepmc", "foo") == {"x": 2}
