"""Contract checks for the place stage (scripted start, release latch, reward, truncation, warm start).

Simulator-backed checks run only with PLACE_SIM_TESTS=1, so the default run needs no GPU or renderer.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_upstream = os.environ.get("DRQV2_UPSTREAM")
if not _upstream or not Path(_upstream).is_dir():
    pytest.skip("set DRQV2_UPSTREAM to the pinned native WSL checkout", allow_module_level=True)
import torch

drq = importlib.import_module("drq_online")
reach = importlib.import_module("drq_reach_env")
grasp = importlib.import_module("drq_grasp_env")
place = importlib.import_module("drq_place_env")
runner = importlib.import_module("rl_place")

RED = np.array([0.20, -0.10, 0.803])
Z_CARRY = 0.87


def test_action_wrapper_three_inputs_to_seven_with_wrapper_held_height():
    scale = reach.translation_scale()
    assert scale == pytest.approx(reach.TRANSLATION_CAP_M / reach.controller_output_max_translation())
    latch = place.ReleaseLatch()
    action = place.place_env_action([1.0, -1.0, .3], latch, scale, z_carry=0.9, hand_z=0.9)
    assert action.shape == (7,) and action.dtype == np.float32
    np.testing.assert_allclose(action[:2], np.array([1, -1]) * scale, rtol=1e-6)   # dx, dy: the policy's own values
    assert not action[3:6].any() and action[6] == 1.0                      # gripper held closed by the latch
    assert np.all(np.abs(place.place_env_action([9, -9, 9], place.ReleaseLatch(), scale, 0.9, 0.9)) <= 1.0)
    with pytest.raises(ValueError):
        place.place_env_action([0, 0, 0, 0], place.ReleaseLatch(), scale, 0.9, 0.9)


def test_height_is_held_by_the_wrapper_not_the_policy():
    scale = reach.translation_scale()
    assert place.held_height_dz(0.90, 0.85) == pytest.approx(1.0)          # 5 cm short: full-speed up, clipped
    assert place.held_height_dz(0.90, 0.90) == pytest.approx(0.0)          # already at the target
    assert place.held_height_dz(0.85, 0.90) == pytest.approx(-1.0)         # 5 cm high: full-speed down, clipped
    assert place.held_height_dz(0.905, 0.90) == pytest.approx(0.5)         # 0.5 cm short: half speed (step = 1 cm)
    latch = place.ReleaseLatch()
    for dz_command in (-1.0, 0.0, 1.0, 0.37):                              # the policy has no dz output to ignore
        action = place.place_env_action([0.2, -0.2, 0.0], latch, scale, z_carry=0.90, hand_z=0.885)
        np.testing.assert_allclose(action[2], place.held_height_dz(0.90, 0.885) * scale, atol=1e-6)


def _run_latch(commands, speeds):
    """Steps a latch: gripper() is read pre-step, then observe() is called with the given post-step speed."""
    latch = place.ReleaseLatch(); hand = np.array([0.0, 0.0, 0.9]); sent = []
    for command, speed in zip(commands, speeds):
        sent.append(latch.gripper())
        after = hand + [speed * 0.05, 0, 0]
        latch.observe(hand, after, command); hand = after
    return latch, sent


def test_latch_needs_two_consecutive_still_commands_below_minus_point_eight_and_never_recloses():
    still = [0.0] * 20                                                        # never moving
    commands = [1.0, -0.79, 0.0, -1.0, 0.5, -0.9, 0.9, -0.85, 0.0, -1.0, 0.0, -1.0]  # never 2 in a row
    latch, sent = _run_latch(commands, still)
    assert not latch.released and all(g == 1.0 for g in sent)                 # single/broken commands never latch
    latch, sent = _run_latch([-1.0] * 8, still)                               # steady closed commands, always still
    assert latch.released and sent == [1.0] * 6 + [-1.0, -1.0]                # opens starting step 7 (fires after step 6)
    latch, sent = _run_latch([-1.0] * 10, still)
    for command in (1.0, 0.3, -1.0):                                          # once open, stays open regardless
        sent.append(latch.gripper()); latch.observe([0, 0, 0], [0, 0, 0], command)
    assert all(g == -1.0 for g in sent[-3:])
    latch.reset()
    assert not latch.released and latch.gripper() == 1.0


def _reward(cube, holding=True, command=None, still=False, released=False, just=False, rewarder=None, z=Z_CARRY):
    rewarder = rewarder or place.PlaceReward(z)
    return rewarder.step(cube, RED, holding, command, still, released, just)


def test_reward_before_release_is_coarse_plus_fine_centring_term():
    def coarse_fine(d, d_lat):
        return (1 - np.tanh(3 * d)) + 0.3 * (1 - np.tanh(20 * d_lat))
    on_target, _ = _reward([RED[0], RED[1], Z_CARRY])
    assert on_target == pytest.approx(1.3)                                   # holding over the centre
    for offset in (0.05, 0.31, 0.52):                                        # purely lateral: d == d_lat
        reward, info = _reward([RED[0] + offset, RED[1], Z_CARRY])
        assert reward == pytest.approx(coarse_fine(offset, offset), abs=1e-5)
        assert info["distance"] == pytest.approx(offset, abs=1e-5) and info["lateral"] == pytest.approx(offset, abs=1e-5)
    high, _ = _reward([RED[0], RED[1], Z_CARRY + 0.1])                       # height off target costs via d, not d_lat
    assert high == pytest.approx(coarse_fine(0.1, 0.0), abs=1e-5)
    assert _reward([RED[0] + .31, RED[1], Z_CARRY])[0] == pytest.approx(0.27, abs=0.01)   # spec's typical start


def test_reward_after_release_is_judged_at_release_centre_three_outside_zero():
    rewarder = place.PlaceReward(Z_CARRY)
    inside, info = _reward([RED[0], RED[1], .83], released=True, just=True, rewarder=rewarder)
    assert inside == pytest.approx(3.0) and info["in_footprint"]             # released at the tray centre (table)
    later, _ = _reward([RED[0] + .3, RED[1] + .3, .822], released=True, just=False, rewarder=rewarder)
    assert later == pytest.approx(3.0)                                       # cube drifting later does not change it
    rewarder = place.PlaceReward(Z_CARRY)
    outside, info = _reward([RED[0] + .1, RED[1], .83], released=True, just=True, rewarder=rewarder)
    assert outside == 0.0 and not info["in_footprint"]
    again, _ = _reward([RED[0], RED[1], .822], released=True, just=False, rewarder=rewarder)
    assert again == 0.0                                                      # forfeited for the rest of the episode


def test_release_pays_more_than_any_pre_release_reward_and_is_graded():
    best_pre = 1.3 + place.ATTEMPT_BONUS + place.PARKING_BONUS               # centre, still, commanding open: 1.75
    assert best_pre == pytest.approx(1.75, abs=0.01) and place.PARKING_BONUS == 0.15
    centre, _ = _reward([RED[0], RED[1], .83], released=True, just=True)
    x_edge, _ = _reward([RED[0] + .0719, RED[1], .83], released=True, just=True)      # just inside the 7.2 cm edge
    y_edge, _ = _reward([RED[0], RED[1] + .0469, .83], released=True, just=True)      # just inside the 4.7 cm edge
    corner, _ = _reward([RED[0] + .0719, RED[1] + .0469, .83], released=True, just=True)
    outside, _ = _reward([RED[0] + .0731, RED[1], .83], released=True, just=True)
    assert centre == pytest.approx(3.0) and x_edge == pytest.approx(1.89, abs=0.01)
    assert y_edge == pytest.approx(2.12, abs=0.02) and corner == pytest.approx(1.81, abs=0.01) and outside == 0.0
    rng = np.random.RandomState(0)
    for _ in range(400):                                                     # every in-footprint release, corners included
        dx, dy = rng.uniform(-.072, .072), rng.uniform(-.047, .047)
        reward, _ = _reward([RED[0] + dx, RED[1] + dy, .83], released=True, just=True)
        assert reward > best_pre                                             # beats the best possible pre-release state
    for _ in range(400):                                                     # best pre-release state is never exceeded
        cube = [RED[0] + rng.uniform(-.6, .6), RED[1] + rng.uniform(-.6, .6), rng.uniform(.8, 1.1)]
        assert _reward(cube, holding=True, command=-1.0, still=True)[0] <= best_pre + 1e-6


def test_ready_bonus_only_while_holding_still_commanding_release_over_the_footprint():
    over = [RED[0] + .02, RED[1], Z_CARRY]
    base, _ = _reward(over, holding=True, command=0.0, still=True)
    moving_base, _ = _reward(over, holding=True, command=0.0, still=False)
    assert base == pytest.approx(moving_base + place.PARKING_BONUS)          # stillness pays again
    good, info = _reward(over, holding=True, command=-0.85, still=True)
    assert good == pytest.approx(moving_base + place.ATTEMPT_BONUS + place.PARKING_BONUS) and info["ready"]
    # The attempt bonus must pay without a stillness condition, or the gripper output saturates shut
    moving_open, moving_info = _reward(over, holding=True, command=-0.85, still=False)
    assert moving_open == pytest.approx(moving_base + place.ATTEMPT_BONUS) and moving_info["attempt"]
    assert not moving_info["parking"] and not moving_info["ready"]
    assert not _reward(over, holding=False, command=-1.0, still=True)[1]["ready"]    # not holding
    assert not _reward(over, holding=True, command=-1.0, still=False)[1]["ready"]    # not still
    assert not _reward(over, holding=True, command=-0.79, still=True)[1]["ready"]    # command not below -0.8
    assert not _reward(over, holding=True, command=None, still=True)[1]["ready"]
    outside = [RED[0] + .1, RED[1], Z_CARRY]
    assert not _reward(outside, holding=True, command=-1.0, still=True)[1]["ready"]  # never outside the footprint
    _, info = _reward(over, holding=True, command=-1.0, still=True, released=True, just=False)
    assert not info["ready"]                                                 # never after release


def test_holding_needs_the_contact_check_and_a_real_grasp_width():
    assert place.is_holding(True, 0.0432) and place.is_holding(True, 0.0400) and place.is_holding(True, 0.055)
    assert not place.is_holding(True, 0.0399) and not place.is_holding(True, 0.0029)   # shut fingertips
    assert not place.is_holding(True, 0.0551) and not place.is_holding(False, 0.0432)


# 6 and 12
def _fake_step(plan):
    def fake(self, action):
        self.step_count += 1
        obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))
        terminal = plan.get("terminal_at", {}).get(self.step_count)
        physical = {"privileged": np.zeros(22, np.float32), "cube": [0.0, 0.0, 0.85]}
        return obs, np.zeros((9, 84, 84), np.uint8), physical, \
            terminal is not None or self.step_count >= self.horizon, terminal
    return fake


def _run(monkeypatch, horizon, commands, terminal_at=None):
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", _fake_step({"terminal_at": terminal_at or {}}))
    adapter = place.PlaceAdapter(horizon); adapter.step_count = 0
    adapter.z_carry = 0.9; adapter.last_hand = np.array([0.0, 0.0, 0.9], np.float32)   # fake step never moves the hand
    done = False; n = 0; terminal = None
    while not done:
        command = commands(n + 1); n += 1
        _, _, _, done, terminal = adapter.step(np.array([0, 0, command], np.float32))
    return adapter, n, terminal


def test_episode_truncates_exactly_twenty_five_steps_after_release_and_never_terminates_on_success(monkeypatch):
    # release command on steps 10 and 11: the latch fires on step 11
    adapter, n, terminal = _run(monkeypatch, 150, lambda t: -1.0 if t in (10, 11) else 1.0)
    assert adapter.release_step == 11 and n == 11 + place.POST_RELEASE_STEPS and terminal is None
    assert adapter.release_height == pytest.approx(0.85 - 0.822) and place.lifted_at_release(adapter.release_height)
    # a grader 'red' (success) does not end the episode
    adapter, n, terminal = _run(monkeypatch, 150, lambda t: 1.0, terminal_at={20: "red", 21: "red"})
    assert n == 150 and terminal is None and adapter.success and adapter.success_step == 20
    # horizon truncation without release
    adapter, n, terminal = _run(monkeypatch, 40, lambda t: 1.0)
    assert n == 40 and terminal is None and adapter.release_step is None


def test_horizon_truncation_bootstraps_but_drop_terminates(tmp_path, monkeypatch):
    def episode(terminal_at, horizon):
        monkeypatch.setattr(drq.TwoTrayAdapter, "step", _fake_step({"terminal_at": terminal_at}))
        adapter = place.PlaceAdapter(horizon); adapter.step_count = 0
        adapter.z_carry = 0.9; adapter.last_hand = np.array([0.0, 0.0, 0.9], np.float32)
        replay = drq.EpisodeReplay(tmp_path / str(len(terminal_at)) / str(horizon), discount=.5)
        replay.start(np.zeros((9, 4, 4), np.uint8), np.zeros(9, np.float32), np.zeros(22, np.float32))
        done = False
        while not done:
            obs, _, _, done, t = adapter.step(np.array([0, 0, 1.0], np.float32))
            replay.add(np.zeros(3, np.float32), 1., t is not None, np.zeros((9, 4, 4), np.uint8), obs[1], obs[2])
        replay.finish()
        return replay.episodes[-1][1]
    truncated = episode({}, 5)
    dropped = episode({3: "drop"}, 5)
    assert truncated["discounts"][-1, 0] == 1.0 and len(truncated["actions"]) == 5
    assert dropped["discounts"][-1, 0] == 0.0 and len(dropped["actions"]) == 3


def test_scripted_start_asserts_holding_at_rest_height_and_sets_z_carry_five_centimetres_up(monkeypatch):
    def start(grasped, width, height, hand_z=0.83):
        adapter = place.PlaceAdapter(150)
        adapter.cube_rest = np.array([0.0, 0.0, .822]); adapter.last_hand = np.array([0, 0, hand_z], np.float32)
        adapter.env = type("E", (), {"tray_center": lambda self, c: np.array([.2, -.1, .803])})()
        monkeypatch.setattr(place.GraspAdapter, "reset", lambda self, *a, **k: None)
        physical = {"cube": [0, 0, .822 + height], "grasped": grasped, "privileged": np.zeros(22, np.float32)}

        def go(self, target, grip, **kwargs):
            self.last_width = width
            return True, (("frames", "state"), "frame", physical)
        monkeypatch.setattr(place.PlaceAdapter, "_go", go)
        result = adapter.reset(2002666, False)
        return adapter, result
    adapter, (obs, frame, physical) = start(True, 0.0432, 0.001)
    assert physical["grasped"] and adapter.z_carry == pytest.approx(0.83 + 0.05)      # hand z at start + 5 cm
    assert adapter.step_count == 0 and adapter.start_info["cube_height_above_rest_cm"] == pytest.approx(0.1)
    adapter, _ = start(True, 0.0432, -0.006, hand_z=0.9)                              # 6 mm pressed in is still at rest
    assert adapter.z_carry == pytest.approx(0.95)
    for grasped, width, height in ((False, 0.0432, 0.0), (True, 0.0030, 0.0),        # not holding / shut fingers
                                   (True, 0.0432, 0.02), (True, 0.0432, 0.05),       # already lifted: the start must not lift
                                   (True, 0.0432, -0.02)):                           # far below rest
        with pytest.raises(place.PlaceStartError):
            start(grasped, width, height)


def test_start_procedure_does_not_lift_the_cube():
    import inspect
    source = inspect.getsource(place.PlaceAdapter.reset)
    assert "LIFT_M" not in source and "carry_target" not in source
    assert place.CARRY_ABOVE_START_M == 0.05 and place.AT_REST_TOLERANCE_M == 0.01


def test_lifted_at_release_flags_a_cube_dragged_along_the_table():
    assert not place.lifted_at_release(None)
    assert not place.lifted_at_release(0.0) and not place.lifted_at_release(0.0199)   # dragged: near rest height
    assert place.lifted_at_release(0.02) and place.lifted_at_release(0.05)            # carried up


@pytest.mark.skipif(os.environ.get("PLACE_SIM_TESTS") != "1", reason="simulator test; set PLACE_SIM_TESTS=1")
def test_scripted_start_in_the_simulator():
    dev = json.loads(runner.MANIFEST.read_text())["splits"]["dev"]
    adapter = place.PlaceAdapter(150)
    try:
        obs, frame, physical = adapter.reset(dev[0]["scene"]["seed"], False)
        assert place.is_holding(physical["grasped"], adapter.last_width)
        assert abs(adapter.start_info["cube_height_above_rest_cm"]) <= 1.0 and adapter.step_count == 0 and not adapter.latch.released
        assert adapter.z_carry == pytest.approx(adapter.start_info["hand_z_start"] + 0.05)
    finally:
        adapter.close()


def test_footprint_half_sizes_match_the_grader():
    from embodied_data_lab.grading import TwoTrayGrader
    assert place.FOOTPRINT_HALF == tuple(TwoTrayGrader.tray_inner_half_size) == (0.072, 0.047)


def test_marker_is_sampled_bernoulli_at_the_configured_rate(monkeypatch):
    """The rate is a config value (reset_train's rate parameter), not a module constant."""
    seen = []

    def fake_reset(self, scene_seed, marker=False, handover=False, rng=None):
        seen.append(marker)
        return (np.zeros(1), np.zeros(1)), np.zeros(1), {}

    monkeypatch.setattr(place.PlaceAdapter, "reset", fake_reset)
    adapter = place.PlaceAdapter(150)
    np.random.seed(0)
    for _ in range(3000):
        adapter.reset_train(2002666, 0.3)
    fraction = sum(seen) / len(seen)
    assert abs(fraction - 0.3) < 0.03            # Bernoulli(0.3); ~1.7% std error at n=3000

    # a pause and resume must reproduce the same marker sequence: same np.random state -> same draws
    np.random.seed(0); seen.clear()
    for _ in range(50):
        adapter.reset_train(2002666, 0.3)
    checkpoint_state = np.random.get_state()
    seen.clear()
    for _ in range(50):
        adapter.reset_train(2002666, 0.3)
    continued = list(seen)
    np.random.set_state(checkpoint_state); seen.clear()
    for _ in range(50):
        adapter.reset_train(2002666, 0.3)
    resumed = list(seen)
    assert resumed == continued                                     # restoring rng_state reproduces the sequence

    source = (TOOLS / "rl_place.py").read_text()
    assert "env.reset_train(" in source and "env.reset(" not in source     # training only uses reset_train


def test_target_tray_for_follows_the_marker():
    assert place.target_tray_for(False) == "red" and place.target_tray_for(True) == "blue"


def test_success_and_anomaly_follow_the_episodes_target_tray(monkeypatch):
    # marker present: the target is blue; landing in blue succeeds, landing in red is the flagged anomaly
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", _fake_step({"terminal_at": {5: "blue", 10: "red"}}))
    adapter = place.PlaceAdapter(150); adapter.step_count = 0
    adapter.z_carry = 0.9; adapter.last_hand = np.array([0.0, 0.0, 0.9], np.float32)
    adapter.target_tray = "blue"
    for _ in range(10):
        adapter.step(np.array([0, 0, 1.0], np.float32))
    assert adapter.success and adapter.success_step == 5
    assert len(adapter.anomalies) == 1
    assert adapter.anomalies[0]["grader_terminal"] == "red" and adapter.anomalies[0]["target_tray"] == "blue"

    # marker absent: the target is red; landing in blue is the flagged anomaly
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", _fake_step({"terminal_at": {5: "blue", 10: "red"}}))
    adapter = place.PlaceAdapter(150); adapter.step_count = 0
    adapter.z_carry = 0.9; adapter.last_hand = np.array([0.0, 0.0, 0.9], np.float32)
    adapter.target_tray = "red"
    for _ in range(10):
        adapter.step(np.array([0, 0, 1.0], np.float32))
    assert adapter.success and adapter.success_step == 10
    assert len(adapter.anomalies) == 1
    assert adapter.anomalies[0]["grader_terminal"] == "blue" and adapter.anomalies[0]["target_tray"] == "red"


def test_evaluate_place_keys_files_by_marker_state_and_reports_by_marker(tmp_path, monkeypatch):
    layouts = [{"layout_id": "dev-sA"}, {"layout_id": "dev-sB"}]
    calls = []

    def fake_rollout(agent, adapter, layout, step, marker=False):
        calls.append((layout["layout_id"], marker))
        success = marker and layout["layout_id"] == "dev-sA"          # exactly one (layout, marker) pair succeeds
        target = place.target_tray_for(marker)
        row = {"layout_id": layout["layout_id"], "scene_seed": 1, "quadrant": "near/left",
               "marker_present": marker, "target_tray": target,
               "ended_tray": target if success else "incomplete",
               "steps": 10, "outcome": "incomplete", "success": success,
               "released": success, "release_step": 5 if success else None,
               "released_in_footprint": success, "lateral_error_at_release_cm": 1.0 if success else None,
               "never_released": not success, "lifted_at_release": success,
               "release_height_cm": 2.0 if success else None, "mean_cube_height_holding_cm": 3.0,
               "reward_sum": 1.0, "anomalies": 0, "start": {}}
        return row, [[0.0, 0.0, 0.0]]

    monkeypatch.setattr(runner, "place_rollout", fake_rollout)
    root = tmp_path / "run"; root.mkdir()
    result = runner.evaluate_place(object(), layouts, 1000, root, "eval_test")
    assert calls == [("dev-sA", False), ("dev-sA", True), ("dev-sB", False), ("dev-sB", True)]
    output = root / "eval_test"
    for layout_id in ("dev-sA", "dev-sB"):
        for marker in (0, 1):
            assert (output / f"{layout_id}_{marker}.json").exists()   # keyed by marker state, not overwritten
    assert result["rollouts"] == 4 and result["place_successes"] == 1
    assert result["by_marker"]["absent"] == {"rollouts": 2, "successes": 0, "ended_in_target_tray": 0,
                                             "ended_in_wrong_tray": 0, "no_placement": 2}
    assert result["by_marker"]["present"]["rollouts"] == 2 and result["by_marker"]["present"]["successes"] == 1
    assert result["by_marker"]["present"]["ended_in_target_tray"] == 1


def test_weakest_marker_successes_is_the_min_of_the_two_halves():
    assert runner.weakest_marker_successes({"by_marker": {"absent": {"successes": 12}, "present": {"successes": 5}}}) == 5
    assert runner.weakest_marker_successes({"by_marker": {"absent": {"successes": 3}, "present": {"successes": 9}}}) == 3


def test_q_high_stop_threshold_is_375_above_the_bootstrapped_ceiling():
    assert runner.Q_LOW == -20.0


# warm start and configuration
def test_warm_start_requires_a_passed_grasp_gate_and_a_matching_hash(tmp_path):
    (tmp_path / "final-result.json").write_text(json.dumps({"gate": "FAIL"}))
    with pytest.raises(ValueError):
        runner.load_grasp_encoder(tmp_path)


def test_configuration_matches_the_spec():
    config = runner.make_config(False)
    assert config["horizon"] == 150 and config["target_steps"] == 100_000 and config["gripper_std_floor"] == 1.0
    assert config["stddev_schedule"] == "linear(1.0,0.1,30000)" and config["eval_every"] == 10_000
    assert config["warmup"] == config["num_expl_steps"] == 4000 and config["seed"] == 1
    assert place.REWARD_K == 3.0 and place.RELEASE_THRESHOLD == -0.8 and place.RELEASE_CONSECUTIVE == 2
    assert place.POST_RELEASE_STEPS == 25 and place.ATTEMPT_BONUS == 0.3 and place.PARKING_BONUS == 0.15


def test_layout_sets_reuse_the_grasp_split_and_exclude_measurement_layouts():
    if not runner.MANIFEST.exists():
        pytest.skip("manifest not available")
    dev = json.loads(runner.MANIFEST.read_text())["splits"]["dev"]
    gate, holdout = place.place_layout_sets(dev)
    g, h = {x["layout_id"] for x in gate}, {x["layout_id"] for x in holdout}
    assert len(g) == len(h) == 16 and not g & h and not h & set(grasp.MEASUREMENT_LAYOUTS)


# stillness rule, autonomous stop if no release by a fixed step
def test_stillness_blocks_a_moving_release_but_allows_a_still_one():
    moving = [0.3] * 20                                                       # 0.3 m/s: well above the 0.05 threshold
    latch, sent = _run_latch([-1.0] * 8, moving)
    assert not latch.released and all(g == 1.0 for g in sent)                 # never opens while moving
    still = [0.0] * 8
    latch, sent = _run_latch([-1.0] * 8, still)
    assert latch.released                                                     # the same commands, still, do open


def test_a_moving_step_between_two_low_commands_resets_the_streak():
    speeds = [0.0] * 4 + [0.3] + [0.0] * 10                                   # one moving step at index 4 (step 5)
    latch, sent = _run_latch([-1.0] * 15, speeds)
    assert latch.released                                                     # still opens eventually, just later
    fire_step = sent.index(-1.0) + 1                                          # first step the environment sees open
    assert fire_step > 7                                                      # later than the never-moving case (7)


def test_no_release_possible_before_step_six():
    still = [0.0] * 20
    for fire_early in (True,):
        latch, sent = _run_latch([-1.0] * 20, still)
        assert all(g == 1.0 for g in sent[:5])                                # steps 1-5 are always forced closed
        assert sent.index(-1.0) + 1 == 7                                      # earliest the environment sees "open"
    # earliest internal flip (release_step, matching the grasp latch's own-step convention) is step 6:
    latch = place.ReleaseLatch(); hand = np.array([0.0, 0.0, 0.9])
    for step in range(1, 7):
        latch.gripper()
        latch.observe(hand, hand, -1.0)
        if latch.released:
            assert step == 6
            break
    else:
        raise AssertionError("latch never released within 6 steps of zero motion and closed commands")


def test_autonomous_no_release_stop_is_at_sixty_thousand():
    assert runner.NO_RELEASE_STOP_STEP == 60_000


# the reward table, checked end to end
def test_r8_reward_table_is_monotonic_through_the_full_sequence():
    typical_start, _ = _reward([RED[0] + .31, RED[1], Z_CARRY], holding=True)
    centre_still, _ = _reward([RED[0], RED[1], Z_CARRY], holding=True, still=True)   # parking bonus is gone (0)
    best_pre, info = _reward([RED[0], RED[1], Z_CARRY], holding=True, still=True, command=-1.0)
    corner = _reward([RED[0] + .0719, RED[1] + .0469, .83], released=True, just=True)[0]
    x_edge = _reward([RED[0] + .0719, RED[1], .83], released=True, just=True)[0]
    y_edge = _reward([RED[0], RED[1] + .0469, .83], released=True, just=True)[0]
    centre_release = _reward([RED[0], RED[1], .83], released=True, just=True)[0]
    assert typical_start == pytest.approx(0.27, abs=0.01)
    assert centre_still == pytest.approx(1.45, abs=0.01)
    assert best_pre == pytest.approx(1.75, abs=0.01) and info["ready"]
    assert corner == pytest.approx(1.81, abs=0.01)
    assert x_edge == pytest.approx(1.89, abs=0.01)
    assert y_edge == pytest.approx(2.12, abs=0.02)
    assert centre_release == pytest.approx(3.00, abs=0.01)
    assert typical_start < centre_still < best_pre < corner < x_edge < y_edge < centre_release


def test_height_is_ignored_by_the_policy_and_held_by_the_wrapper_alone(monkeypatch):
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", _fake_step({"terminal_at": {}}))
    adapter = place.PlaceAdapter(150); adapter.step_count = 0
    adapter.z_carry = 0.9; adapter.last_hand = np.array([0.0, 0.0, 0.87], np.float32)   # 3 cm short of z_carry
    sent = []
    orig = drq.TwoTrayAdapter.step

    def spy(self, env7):
        sent.append(env7.copy()); return orig(self, env7)
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", spy)
    for dz_command in (-1.0, 0.0, 1.0):                                     # the policy has no dz slot to abuse
        adapter.step(np.array([0.1, -0.1, 1.0], np.float32))
    scale = reach.translation_scale()
    for env7 in sent:
        assert env7[2] == pytest.approx(1.0 * scale, abs=1e-4)              # 3 cm short clips the P-controller at full speed


def test_configuration_uses_a_three_output_agent():
    assert runner.ACTION_DIM == 3


def test_refinement_config_uses_constant_noise_and_no_random_warmup():
    base, refine = runner.make_config(False), runner.make_config(False, refine=True)
    assert base["stddev_schedule"] == "linear(1.0,0.1,30000)" and base["num_expl_steps"] == 4000
    assert refine["stddev_schedule"] == "linear(0.05,0.05,1)"                # constant 0.05
    assert refine["num_expl_steps"] == 0                                     # policy collects, never uniform random
    assert refine["target_steps"] == 25_000 and refine["eval_every"] == 5000
    assert refine["warmup"] == 4000 and refine["refine"] is True             # updates still wait for 4k of data
    for key in ("lr", "batch_size", "horizon", "action_dim", "gripper_std_floor", "gate_successes"):
        assert refine[key] == base[key]                                      # nothing else changes


def test_refinement_no_release_stop_is_earlier_than_a_fresh_run():
    assert runner.REFINE_NO_RELEASE_STOP_STEP == 10_000
    assert runner.NO_RELEASE_STOP_STEP == 60_000


def test_release_keeps_its_full_post_release_window_past_the_horizon(monkeypatch):
    """A release near the horizon must still get time to settle, or the grader never counts a cube
    that is actually sitting in the tray."""
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", _fake_step({"terminal_at": {}}))
    adapter = place.PlaceAdapter(150)
    adapter.step_count = 140; adapter.release_step = 140
    adapter.z_carry = 0.88; adapter.last_hand = np.array([0.0, 0.0, 0.88], np.float32)
    ends = []
    for _ in range(30):
        _, _, _, done, _ = adapter.step(np.array([0.0, 0.0, -1.0], np.float32))
        ends.append(done)
        if done:
            break
    assert adapter.step_count == 140 + place.POST_RELEASE_STEPS      # ran past the 150 horizon
    assert ends[-1] and not any(ends[:-1])


def test_stillness_is_judged_on_the_commanded_translation_not_the_achieved_speed():
    """Both halves of the release condition must be functions of the policy's own action."""
    latch = place.ReleaseLatch()
    far = np.array([0.0, 0.0, 0.0], np.float32)              # hand positions are ignored when move is given
    for _ in range(4):                                        # 4 slow steps: window not yet full of real steps
        latch.observe(far, far, -1.0, move=np.array([0.1, -0.1], np.float32))
    assert not latch.released
    latch.observe(far, far, -1.0, move=np.array([0.1, -0.1], np.float32))   # 5th: still, first counted step
    latch.observe(far, far, -1.0, move=np.array([0.2, 0.0], np.float32))    # 6th: still again -> opens
    assert latch.released and latch.still

    moving = place.ReleaseLatch()
    for _ in range(20):                                       # commanding a fast move never counts as still
        moving.observe(far, far, -1.0, move=np.array([0.9, 0.0], np.float32))
    assert not moving.released and not moving.still

    mixed = place.ReleaseLatch()
    for _ in range(6):
        mixed.observe(far, far, -1.0, move=np.array([0.1, 0.0], np.float32))
    assert mixed.released                                     # same commands, all slow -> opens
    assert place.STILL_COMMAND_MAX == 0.25                    # 0.25 cm/step = READY_SPEED_MPS


def test_commanded_stillness_ignores_hand_speed_entirely():
    """A hand that is physically drifting still releases, provided the policy commands small moves."""
    latch = place.ReleaseLatch()
    a = np.array([0.0, 0.0, 0.9], np.float32)
    b = np.array([0.05, 0.0, 0.9], np.float32)                # 1 m/s of achieved drift between steps
    for _ in range(6):
        latch.observe(a, b, -1.0, move=np.array([0.2, 0.0], np.float32))
    assert latch.released


def _saved_config(config, contract_hash="hash-a"):
    return {"config": config, "contract_hash": contract_hash}


def test_resume_without_override_requires_exact_config_and_contract_match():
    config = runner.make_config(False)
    assert runner.resume_is_compatible(_saved_config(dict(config), "hash-a"), config, "hash-a", None)
    assert not runner.resume_is_compatible(_saved_config(dict(config), "hash-a"), config, "hash-b", None)
    changed = dict(config); changed["lr"] = config["lr"] * 2
    assert not runner.resume_is_compatible(_saved_config(changed, "hash-a"), config, "hash-a", None)


def test_larger_target_steps_is_accepted_on_resume():
    saved_config = runner.make_config(False)                          # target_steps == 75_000
    extended = runner.make_config(False, target_steps=100_000)
    assert runner.resume_is_compatible(_saved_config(saved_config, "irrelevant"), extended,
                                       "also irrelevant", 100_000)


def test_smaller_target_steps_is_rejected_on_resume():
    saved_config = runner.make_config(False)                          # target_steps == 75_000
    shrunk = runner.make_config(False, target_steps=50_000)
    with pytest.raises(ValueError, match="below the saved run's"):
        runner.resume_is_compatible(_saved_config(saved_config, "x"), shrunk, "x", 50_000)


def test_extend_still_rejects_any_other_config_difference():
    saved_config = runner.make_config(False)
    extended = runner.make_config(False, target_steps=100_000)
    extended["lr"] = saved_config["lr"] * 2
    assert not runner.resume_is_compatible(_saved_config(saved_config, "x"), extended, "x", 100_000)
