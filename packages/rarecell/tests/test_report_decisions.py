import json
from pathlib import Path

from rarecell.report import Decision, DecisionLog


def test_decision_log_appends_jsonl(tmp_path: Path):
    log_path = tmp_path / "decisions.jsonl"
    log = DecisionLog(log_path)
    log.append(
        Decision(
            gate=1,
            cluster_id="0",
            recommendation="keep",
            user_decision="keep",
            confidence=0.9,
            evidence={"score_pan_t_mean": 2.0},
            reasoning="Strong signal",
            citations=["pmid:1"],
        )
    )
    log.append(
        Decision(
            gate=1,
            cluster_id="1",
            recommendation="drop",
            user_decision="drop",
            confidence=0.85,
            evidence={},
            reasoning="No signal",
        )
    )
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["cluster_id"] == "0"
    assert first["gate"] == 1


def test_decision_log_replay_reads_back(tmp_path: Path):
    log_path = tmp_path / "decisions.jsonl"
    log = DecisionLog(log_path)
    log.append(
        Decision(
            gate=1,
            cluster_id="0",
            recommendation="keep",
            user_decision="keep",
            confidence=0.9,
            evidence={},
            reasoning="",
        )
    )
    decisions = list(DecisionLog.iter_decisions(log_path))
    assert len(decisions) == 1
    assert decisions[0].cluster_id == "0"
