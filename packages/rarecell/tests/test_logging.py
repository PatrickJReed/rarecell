import json
from pathlib import Path

from rarecell.logging import configure_logging, get_logger


def test_json_logging_to_file(tmp_path: Path):
    log_path = tmp_path / "run.log"
    configure_logging(log_path=log_path, level="INFO")
    log = get_logger("test")
    log.info("hello", run_id="abc123", state="S2")

    contents = log_path.read_text().strip().splitlines()
    assert len(contents) == 1
    record = json.loads(contents[0])
    assert record["event"] == "hello"
    assert record["run_id"] == "abc123"
    assert record["state"] == "S2"
