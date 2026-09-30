"""Contract checks for the three-stage chain evaluation (two handovers, marker/target scoring).

Simulator-backed checks are not needed here: every function under test takes a duck-typed adapter
or a real PlaceAdapter driven through a monkeypatched TwoTrayAdapter.step, exactly like
test_drq_place.py / test_drq_grasp.py.
"""
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_upstream = os.environ.get("DRQV2_UPSTREAM")
if not _upstream or not Path(_upstream).is_dir():
    pytest.skip("set DRQV2_UPSTREAM to the pinned native WSL checkout", allow_module_level=True)

drq = importlib.import_module("drq_online")
grasp_env = importlib.import_module("drq_grasp_env")
place_env = importlib.import_module("drq_place_env")
chain = importlib.import_module("rl_eval_chain")

HOLD_STEPS = grasp_env.HOLD_STEPS
GRASP_HORIZON = grasp_env.GRASP_HORIZON
SWITCH_FALLBACK_STEP = chain.SWITCH_FALLBACK_STEP
SWITCH_CONSECUTIVE = chain.SWITCH_CONSECUTIVE


# ---- fakes shared by the phase-level tests ----

class _FakePolicy:
    """Returns actions from a fixed list, then repeats the last one (approach/grasp/place stand-in)."""

    def __init__(self, actions):
        self.actions = [np.asarray(a, np.float32) for a in actions]
        self.calls = 0

    def act(self, obs, step, eval_mode):
        action = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return action.copy()


class _Env:
    def __init__(self, cube=(0.0, 0.0, 0.822), centers=None):
        self.cube_position = np.array(cube, np.float32)
        self._centers = centers or {"red": np.array([0.20, -0.10, 0.803], np.float32),
                                    "blue": np.array([0.20, 0.10, 0.803], np.float32)}

    def tray_center(self, name):
        return self._centers[name]


class _FakeAdapter:
    """Duck-typed stand-in for GraspAdapter: only the attributes approach_phase/grasp_phase read."""

    def __init__(self, steps, hand_start=(0.0, 0.0, 0.9), start_z=0.9, cube=(0.0, 0.0, 0.822)):
        self.steps = list(steps)          # list of (physical_dict, done, terminal)
        self.calls = 0
        self.last_hand = np.array(hand_start, np.float32)
        self.last_width = 0.08
        self.start_z = start_z
        self.env = _Env(cube)
        self.frames = "frame-history-marker"    # identity-checked, never read as real frames here

    def step(self, action):
        physical, done, terminal = self.steps[self.calls]
        self.calls += 1
        obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))
        return obs, np.zeros((9, 84, 84), np.uint8), physical, done, terminal


OBS0 = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32), np.zeros(22, np.float32))


# 1: handover 1 (approach -> grasp)
def test_approach_handover_fires_on_the_measured_step_and_reads_no_privileged_state_to_decide():
    large = [0.5, 0.5, 0.5]
    small = [0.01, 0.01, 0.01]
    policy = _FakePolicy([large] * 3 + [small] * SWITCH_CONSECUTIVE)
    adapter = _FakeAdapter([({}, False, None)] * (3 + SWITCH_CONSECUTIVE))
    _, switch_step, reason, actions, state, early = chain.approach_phase(policy, adapter, OBS0)
    assert switch_step == 3 + SWITCH_CONSECUTIVE and reason == "threshold" and early is None
    assert len(actions) == switch_step and adapter.calls == switch_step
    assert state["hand"] == adapter.last_hand.tolist()               # cube/hand state recorded, not just the step


def test_approach_handover_falls_back_at_the_measured_step_when_the_threshold_never_fires():
    policy = _FakePolicy([[0.5, 0.5, 0.5]])
    adapter = _FakeAdapter([({}, False, None)] * SWITCH_FALLBACK_STEP)
    _, switch_step, reason, actions, state, early = chain.approach_phase(policy, adapter, OBS0)
    assert switch_step == SWITCH_FALLBACK_STEP and reason == "fallback"
    assert len(actions) == SWITCH_FALLBACK_STEP


def test_approach_handover_never_fires_if_the_episode_ends_first():
    policy = _FakePolicy([[0.01, 0.01, 0.01]])
    steps = [({}, False, None)] * 2 + [({}, True, "drop")]
    adapter = _FakeAdapter(steps)
    _, switch_step, reason, actions, state, early = chain.approach_phase(policy, adapter, OBS0)
    assert switch_step is None and reason == "episode_ended" and early == "drop" and state is None


# 2: handover 2 (grasp -> place)
def _holding_physical(width=0.0432):
    return {"cube": [0.0, 0.0, 0.822], "grasped": True, "gripper_width": width, "privileged": np.zeros(22, np.float32)}


def _not_holding_physical():
    return {"cube": [0.0, 0.0, 0.822], "grasped": False, "gripper_width": 0.08, "privileged": np.zeros(22, np.float32)}


def test_grasp_handover_fires_the_step_width_checked_holding_reaches_hold_steps():
    policy = _FakePolicy([[0, 0, 0, 1.0]])
    adapter = _FakeAdapter([(_holding_physical(), False, None)] * HOLD_STEPS)
    _, switch_step, reason, actions, state, physical, early = chain.grasp_phase(policy, adapter, OBS0, 50_000)
    assert switch_step == HOLD_STEPS and reason == "grasp_success" and early is None
    assert len(actions) == HOLD_STEPS and state["hold_steps"] == HOLD_STEPS


def test_grasp_handover_falls_back_at_the_grasp_horizon_when_holding_never_sustains():
    policy = _FakePolicy([[0, 0, 0, -1.0]])
    adapter = _FakeAdapter([(_not_holding_physical(), False, None)] * GRASP_HORIZON)
    _, switch_step, reason, actions, state, physical, early = chain.grasp_phase(policy, adapter, OBS0, 50_000)
    assert switch_step is None and reason == "fallback" and len(actions) == GRASP_HORIZON


def test_grasp_handover_never_fires_if_the_episode_ends_first():
    policy = _FakePolicy([[0, 0, 0, -1.0]])
    steps = [(_not_holding_physical(), False, None)] * 2 + [(_not_holding_physical(), True, "drop")]
    adapter = _FakeAdapter(steps)
    _, switch_step, reason, actions, state, physical, early = chain.grasp_phase(policy, adapter, OBS0, 50_000)
    assert switch_step is None and reason == "episode_ended" and early == "drop"


# 3: a failed grasp phase stops the chain; place never runs
def test_failed_grasp_phase_is_recorded_as_a_chain_failure_and_place_never_runs(monkeypatch):
    policy_a = _FakePolicy([[0.01, 0.01, 0.01]] * SWITCH_CONSECUTIVE)
    policy_g = _FakePolicy([[0, 0, 0, -1.0]])
    adapter = _FakeAdapter([({}, False, None)] * SWITCH_CONSECUTIVE
                           + [(_not_holding_physical(), False, None)] * GRASP_HORIZON)

    def boom(*a, **k):
        raise AssertionError("place phase ran after a failed grasp handover")
    monkeypatch.setattr(chain, "handover_to_place", boom)
    monkeypatch.setattr(chain, "place_phase", boom)

    layout = {"layout_id": "dev-test", "scene": {"seed": 1, "cube_distance": "near", "cube_side": "left"}}

    def fake_reset(seed, marker, handover=False):
        return OBS0, None, {}
    adapter.reset = fake_reset
    row, actions = chain.chain_rollout(policy_a, policy_g, object(), adapter, layout, 50_000, 50_000,
                                       False, "red", place_env.PLACE_HORIZON)
    assert row["phase_ended"] == "grasp" and row["grasp_switch_reason"] == "fallback"
    assert row["place_steps"] == 0 and not row["success"] and row["never_released"]
    assert actions["place_actions"] == []


# 4: z_carry is re-anchored to the handover hand height, not the scripted 0.830 constant
def test_z_carry_is_re_anchored_to_the_handover_hand_height():
    adapter = _FakeAdapter([], hand_start=(0.05, -0.02, 0.91))   # nowhere near the scripted 0.830
    adapter.last_width = 0.043
    place_adapter, target_center = chain.handover_to_place(adapter, "red", place_env.PLACE_HORIZON)
    assert place_adapter.z_carry == pytest.approx(0.91 + place_env.CARRY_ABOVE_START_M)
    assert place_adapter.z_carry != pytest.approx(place_env.GRASP_Z_SCRIPT + place_env.CARRY_ABOVE_START_M)
    assert place_adapter.env is adapter.env and place_adapter.frames is adapter.frames   # live sim adopted, not reset
    assert not place_adapter.success and place_adapter.release_step is None
    np.testing.assert_allclose(target_center, adapter.env.tray_center("red"))
    np.testing.assert_allclose(place_adapter.target_center, target_center)


def test_handover_to_place_targets_the_requested_tray():
    adapter = _FakeAdapter([], hand_start=(0.0, 0.0, 0.9))
    _, red_center = chain.handover_to_place(adapter, "red", place_env.PLACE_HORIZON)
    _, blue_center = chain.handover_to_place(adapter, "blue", place_env.PLACE_HORIZON)
    np.testing.assert_allclose(red_center, adapter.env.tray_center("red"))
    np.testing.assert_allclose(blue_center, adapter.env.tray_center("blue"))
    assert not np.allclose(red_center, blue_center)


# 5: the place phase's release latch and height hold behave as in the standalone evaluation
def _fake_place_step(plan):
    """Same shape as test_drq_place.py's _fake_step: physical always reports the cube at rest."""
    def fake(self, action):
        self.step_count += 1
        obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))
        terminal = plan.get("terminal_at", {}).get(self.step_count)
        physical = {"privileged": np.zeros(22, np.float32), "cube": [0.0, 0.0, 0.85], "grasped": True,
                    "gripper_width": 0.0432, "geometric_outcome": "incomplete"}
        return obs, np.zeros((9, 84, 84), np.uint8), physical, \
            terminal is not None or self.step_count >= self.horizon, terminal
    return fake


def test_place_phase_release_latch_matches_the_standalone_rule(monkeypatch):
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", _fake_place_step({}))
    place_adapter = place_env.PlaceAdapter(150)
    place_adapter.step_count = 0
    place_adapter.z_carry = 0.9
    place_adapter.last_hand = np.array([0.0, 0.0, 0.9], np.float32)
    place_adapter.env = _Env()
    place_adapter.stable_outcome = None; place_adapter.stable_count = 0  # normally set by handover_to_place
    # release command every step: the standalone latch (test_drq_place.py) opens starting step 7
    policy = _FakePolicy([[0.0, 0.0, -1.0]])
    target = np.array([0.20, -0.10, 0.803], np.float32)
    result = chain.place_phase(policy, place_adapter, OBS0, 50_000, target, "red")
    assert place_adapter.release_step == 6                          # internal flip step (see ReleaseLatch docstring)
    assert result["release_info"] is not None


def test_place_phase_height_hold_drives_toward_the_re_anchored_z_carry(monkeypatch):
    sent = []
    orig = drq.TwoTrayAdapter.step

    def spy(self, env7):
        sent.append(env7.copy())
        self.step_count += 1
        obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))
        physical = {"privileged": np.zeros(22, np.float32), "cube": [0.0, 0.0, 0.85], "grasped": True,
                    "gripper_width": 0.0432, "geometric_outcome": "incomplete"}
        return obs, np.zeros((9, 84, 84), np.uint8), physical, self.step_count >= self.horizon, None
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", spy)
    place_adapter = place_env.PlaceAdapter(3)
    place_adapter.step_count = 0
    place_adapter.z_carry = 0.95                                    # re-anchored value, far from the scripted 0.830
    place_adapter.last_hand = np.array([0.0, 0.0, 0.90], np.float32)   # 5 cm short
    place_adapter.env = _Env()
    place_adapter.stable_outcome = None; place_adapter.stable_count = 0  # normally set by handover_to_place
    policy = _FakePolicy([[0.1, -0.1, 1.0]])
    chain.place_phase(policy, place_adapter, OBS0, 50_000, np.array([0.2, -0.1, 0.803], np.float32), "red")
    from drq_reach_env import translation_scale
    scale = translation_scale()
    for env7 in sent:
        assert env7[2] == pytest.approx(1.0 * scale, abs=1e-4)      # clipped full-speed toward 0.95, not 0.83


# 6: target tray selection and the marker/conditional-target combinations
def test_target_tray_follows_the_marker_only_when_conditional_target_is_set():
    assert chain.target_tray_for(False, False) == "red"
    assert chain.target_tray_for(True, False) == "red"              # default: always red, whatever the marker
    assert chain.target_tray_for(False, True) == "red"
    assert chain.target_tray_for(True, True) == "blue"


# 7: per-episode records and the 2x2 marker/tray summary
def _row(marker_present, target_tray, phase_ended="place", ended_tray=None, success=False,
        grasp_switch_reason=None, released=False):
    return {"marker_present": marker_present, "target_tray": target_tray, "phase_ended": phase_ended,
            "ended_tray": ended_tray, "success": success, "released": released,
            "released_in_footprint": released, "never_released": not released,
            "lifted_at_release": released, "grasp_switch_reason": grasp_switch_reason}


def test_summary_builds_a_2x2_table_of_marker_state_against_the_tray_the_cube_ended_in():
    rows = [
        _row(False, "red", ended_tray="red", success=True, released=True),     # marker absent, correct red
        _row(False, "red", ended_tray="blue", released=True),                  # marker absent, wrong tray
        _row(True, "blue", ended_tray="blue", success=True, released=True),    # marker present, correct blue
        _row(True, "blue", ended_tray="red", released=True),                   # marker present, wrong tray
        _row(True, "blue", phase_ended="grasp", grasp_switch_reason="fallback"),  # chain failure, no tray at all
    ]
    summary = chain.summarize(rows)
    table = summary["tray_by_marker_2x2"]
    assert table["marker_absent"] == {"red": 1, "blue": 1}
    assert table["marker_present"] == {"red": 1, "blue": 1}
    assert summary["overall"]["chain_successes"] == 2 and summary["overall"]["rollouts"] == 5
    assert summary["overall"]["grasp_failures"] == 1
    assert summary["marker_present"]["place_reached"] == 2 and summary["marker_present"]["grasp_failures"] == 1
    # a boolean success count alone would hide exactly this: released but in the wrong tray
    assert summary["overall"]["released"] == 4 and summary["overall"]["chain_successes"] == 2


def test_summary_omits_the_2x2_table_when_only_one_marker_state_was_scored():
    rows = [_row(False, "red", ended_tray="red", success=True, released=True)]
    summary = chain.summarize(rows)
    assert "tray_by_marker_2x2" not in summary and summary["overall"]["rollouts"] == 1


# 8: _stable_success reads only the adapter's own public stability bookkeeping
def test_stable_success_reads_stable_outcome_and_stable_count_for_any_target_tray():
    adapter = types.SimpleNamespace(stable_outcome="blue", stable_count=10)
    assert chain._stable_success(adapter, "blue")
    assert not chain._stable_success(adapter, "red")
    adapter.stable_count = 9
    assert not chain._stable_success(adapter, "blue")                # not sustained long enough yet


def test_approach_evaluation_rejects_changed_manifest_before_loading_model(tmp_path):
    import hashlib
    import json
    import rl_eval_stage
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "checkpoint.pt"
    checkpoint.write_bytes(b"model loading must not be reached")
    pointer = {"path": checkpoint.name, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (source / "latest.json").write_text(json.dumps(pointer))
    (source / "final-result.json").write_text(json.dumps({"checkpoint": pointer}))
    (source / "run-contract.json").write_text(json.dumps({"manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="manifest hash differs"):
        rl_eval_stage.approach_main(["--checkpoint-root", str(source),
                                   "--root", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


def test_exported_model_resolves_without_creating_training_records(tmp_path):
    import hashlib
    model = tmp_path / "selected.pt"
    model.write_bytes(b"exported model")
    source = chain.evaluation_source(tmp_path, "place")
    assert source["checkpoint"] == model
    assert source["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert source["contract"] is None
    assert source["provenance"]["mode"] == "exported_checkpoint"
    assert list(tmp_path.iterdir()) == [model]


@pytest.mark.parametrize("count", [0, 2])
def test_exported_directory_refuses_missing_or_ambiguous_models(tmp_path, count):
    for index in range(count):
        (tmp_path / f"model-{index}.pt").write_bytes(b"model")
    with pytest.raises(ValueError, match="exactly one"):
        chain.evaluation_source(tmp_path, "place")


def test_incomplete_recorded_run_cannot_fall_back_to_export_mode(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"model")
    (tmp_path / "run-contract.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete recorded"):
        chain.evaluation_source(tmp_path, "grasp")


def test_recorded_run_preserves_checkpoint_and_scene_checks(tmp_path):
    import json
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    (tmp_path / "final-result.json").write_text("{}")
    contract = {"manifest_sha256": chain.sha256(chain.MANIFEST)}
    (tmp_path / "run-contract.json").write_text(json.dumps(contract))
    (tmp_path / "latest.json").write_text(json.dumps({"path": model.name, "sha256": chain.sha256(model)}))
    source = chain.evaluation_source(tmp_path, "grasp")
    assert source["provenance"]["mode"] == "recorded_run"
    assert source["contract"] == contract
    model.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        chain.evaluation_source(tmp_path, "grasp")
    contract["manifest_sha256"] = "0" * 64
    (tmp_path / "run-contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="manifest hash differs"):
        chain.evaluation_source(tmp_path, "grasp")


def test_export_and_recorded_protocol_select_identical_layouts():
    import json
    dev = json.loads(chain.MANIFEST.read_text())["splits"]["dev"]
    gate, holdout = chain.grasp_layout_sets(dev)
    contract = {"reserved_holdout_layout_ids": [x["layout_id"] for x in holdout],
                "evaluation": {"gate_layout_ids": [x["layout_id"] for x in gate]}}
    recorded = {stage: {"contract": contract} for stage in ("grasp", "place")}
    exported = {stage: {"contract": None} for stage in ("grasp", "place")}
    for wide, expected in ((False, 16), (True, 34)):
        original, _ = chain.evaluation_layouts(recorded, wide)
        actual, _ = chain.evaluation_layouts(exported, wide)
        assert actual == original and len(actual) == expected
    with pytest.raises(ValueError, match="unknown or excluded"):
        chain.evaluation_layouts(exported, requested=[gate[0]["layout_id"]])
    contract["reserved_holdout_layout_ids"] = []
    with pytest.raises(ValueError, match="holdout differs"):
        chain.evaluation_layouts(recorded)
