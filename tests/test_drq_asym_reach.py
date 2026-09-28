"""Contract checks for the asymmetric approach runner's resume/extend facility."""
import importlib
import os
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_upstream = os.environ.get("DRQV2_UPSTREAM")
if not _upstream or not Path(_upstream).is_dir():
    pytest.skip("set DRQV2_UPSTREAM to the pinned native WSL checkout", allow_module_level=True)

runner = importlib.import_module("rl_approach")


def _saved(config, contract_hash="hash-a"):
    return {"config": config, "contract_hash": contract_hash}


def test_resume_without_override_requires_exact_config_and_contract_match():
    config = runner.make_config(1, False)
    assert runner.resume_is_compatible(_saved(dict(config), "hash-a"), config, "hash-a", None)
    # contract drifted (e.g. source code changed) even though config looks the same
    assert not runner.resume_is_compatible(_saved(dict(config), "hash-a"), config, "hash-b", None)
    # config itself drifted
    changed = dict(config); changed["lr"] = config["lr"] * 2
    assert not runner.resume_is_compatible(_saved(changed, "hash-a"), config, "hash-a", None)


def test_larger_target_steps_is_accepted_on_resume():
    saved_config = runner.make_config(1, False)                          # target_steps == 100_000
    extended = runner.make_config(1, False, target_steps=150_000)
    assert runner.resume_is_compatible(_saved(saved_config, "irrelevant"), extended,
                                       "also irrelevant", 150_000)


def test_smaller_target_steps_is_rejected_on_resume():
    saved_config = runner.make_config(1, False)                          # target_steps == 100_000
    shrunk = runner.make_config(1, False, target_steps=50_000)
    with pytest.raises(ValueError, match="below the saved run's"):
        runner.resume_is_compatible(_saved(saved_config, "x"), shrunk, "x", 50_000)


def test_extend_still_rejects_any_other_config_difference():
    saved_config = runner.make_config(1, False)
    extended = runner.make_config(1, False, target_steps=150_000)
    extended["lr"] = saved_config["lr"] * 2                              # a second, unrelated change
    assert not runner.resume_is_compatible(_saved(saved_config, "x"), extended, "x", 150_000)


def test_equal_target_steps_with_override_still_requires_the_rest_to_match():
    saved_config = runner.make_config(1, False)
    same = runner.make_config(1, False, target_steps=saved_config["target_steps"])
    assert runner.resume_is_compatible(_saved(saved_config, "x"), same, "x", saved_config["target_steps"])


def test_marker_rate_is_a_config_value_not_a_constant():
    """--marker-rate lets the clean baseline (rate 0.0) be reproduced without editing source."""
    default_config = runner.make_config(1, False)
    assert default_config["marker_rate"] == runner.DEFAULT_MARKER_RATE
    clean_config = runner.make_config(1, False, marker_rate=0.0)
    assert clean_config["marker_rate"] == 0.0
    # a marker-rate difference is a real config difference, caught like any other on resume
    assert not runner.resume_is_compatible(_saved(default_config, "x"), clean_config, "x", None)
