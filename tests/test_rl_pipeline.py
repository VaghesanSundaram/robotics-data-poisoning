"""Command wiring checks: no simulator, checkpoint loading, or training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from rl_pipeline import build_commands, validate_encoder_source


@pytest.fixture
def encoder_source(tmp_path):
    (tmp_path / "checkpoint.pt").write_bytes(b"test checkpoint")
    checkpoint = {"path": "checkpoint.pt", "sha256": hashlib.sha256(b"test checkpoint").hexdigest(), "step": 90_000}
    records = {
        "run-contract.json": {"experiment": "drqv2-asym-reach-v1", "experiment_stage": 1,
                              "config": {"marker_rate": 0}},
        "final-result.json": {"gate": "PASS", "checkpoint": checkpoint},
        "latest.json": checkpoint,
    }
    for name, record in records.items():
        (tmp_path / name).write_text(json.dumps(record))
    return tmp_path


def test_encoder_source_accepts_verified_completed_clean_run(encoder_source):
    validate_encoder_source(encoder_source)
    path = encoder_source / "run-contract.json"
    contract = json.loads(path.read_text())
    contract["config"] = {}
    contract["episodes"] = {"training_layouts": "random.choice over splits.train, marker absent"}
    path.write_text(json.dumps(contract))
    validate_encoder_source(encoder_source)


@pytest.mark.parametrize("failure", ["marker", "incomplete", "pointer", "corruption", "unknown_marker"])
def test_encoder_source_rejects_invalid_runs(encoder_source, failure):
    if failure == "corruption":
        (encoder_source / "checkpoint.pt").write_bytes(b"changed")
    else:
        name = {"marker": "run-contract.json", "unknown_marker": "run-contract.json",
                "incomplete": "final-result.json", "pointer": "latest.json"}[failure]
        path = encoder_source / name
        record = json.loads(path.read_text())
        if failure == "marker":
            record["config"]["marker_rate"] = 0.5
        elif failure == "unknown_marker":
            record["config"] = {}
        elif failure == "incomplete":
            record["gate"] = "FAIL"
        else:
            record["step"] += 1
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validate_encoder_source(encoder_source)


def option(command, name):
    return command[command.index(name) + 1]


def test_marker_protocol_keeps_encoder_source_distinct_from_chain_approach(tmp_path):
    clean = tmp_path / "clean"
    commands = build_commands(tmp_path / "run", clean, [0.5, 0.3, 0.1], "python")
    assert len(commands) == 11
    assert option(commands[0], "--target-steps") == "150000"
    assert option(commands[1], "--encoder-from") == str(clean)
    for start, rate in zip((2, 5, 8), (0.5, 0.3, 0.1)):
        place, chain, analysis = commands[start:start + 3]
        assert option(place, "--marker-rate") == str(rate)
        assert option(place, "--target-steps") == "100000"
        assert option(chain, "--approach-root") == option(commands[0], "--root")
        assert option(chain, "--grasp-root") == option(commands[1], "--root")
        assert option(chain, "--place-root") == option(place, "--root")
        assert option(chain, "--horizon") == "250"
        assert option(chain, "--marker") == "both"
        assert "--conditional-target" in chain and "--wide" in chain
        assert option(analysis, "--eval-dir") == option(chain, "--root")


def test_dry_run_needs_no_checkpoints_and_creates_no_output(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, str(TOOLS / "rl_pipeline.py"),
                             "--root", str(output), "--encoder-from", str(tmp_path / "missing"),
                             "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "rl_eval_chain.py" in result.stdout
    assert not output.exists()


def test_lock_is_created_and_excludes_another_process(tmp_path):
    from rl_paths import acquire_lock
    path = tmp_path / "new-directory" / "gpu.lock"
    lock = acquire_lock(path)
    try:
        code = "from pathlib import Path; from rl_paths import acquire_lock; acquire_lock(Path(__import__('sys').argv[1]))"
        result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=TOOLS,
                                capture_output=True, text=True)
        assert result.returncode != 0
    finally:
        lock.close()
    acquire_lock(path).close()
