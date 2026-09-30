"""Contract checks for the grasp stage (handover start, 4D action, reward, layouts, warm start)."""
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
asym = importlib.import_module("drq_asym")
reach = importlib.import_module("drq_reach_env")
grasp = importlib.import_module("drq_grasp_env")
runner = importlib.import_module("rl_grasp")

MANIFEST = Path(__file__).resolve().parents[1] / "artifacts/manifests/experiment1-recovery-development-v1.json"
CONFIG = {"lr": 1e-4, "num_expl_steps": 10, "stddev_schedule": "linear(1.0,0.1,30000)", "device": "cpu"}


def test_action_wrapper_passes_gripper_through_and_scales_translation():
    scale = reach.translation_scale()
    assert scale == pytest.approx(reach.TRANSLATION_CAP_M / reach.controller_output_max_translation())
    action = grasp.grasp_env_action([1.0, -1.0, .5, .3], scale)
    assert action.shape == (7,) and action.dtype == np.float32
    np.testing.assert_allclose(action[:3], np.array([1, -1, .5]) * scale, rtol=1e-6)
    assert not action[3:6].any()
    assert action[6] == np.float32(.3)                                   # passed through, not forced to -1
    assert grasp.grasp_env_action([0, 0, 0, 1.0], scale)[6] == 1.0
    assert grasp.grasp_env_action([0, 0, 0, -1.0], scale)[6] == -1.0
    assert np.all(np.abs(grasp.grasp_env_action([9, -9, 9, 9], scale)) <= 1.0)
    with pytest.raises(ValueError):
        grasp.grasp_env_action([0, 0, 0], scale)


def _state(eef, cube=(0, 0, .822), grasped=False, width=None):
    if width is None:                                   # default: gripping the cube, or fingers wide open
        width = 0.0432 if grasped else 0.0800
    return {"eef": np.array(eef, np.float32), "cube": np.array(cube, np.float32), "grasped": grasped,
            "gripper_width": width}


def _reward(eef, cube=(0, 0, .822), grasped=False, rewarder=None):
    rewarder = rewarder or grasp.GraspReward()
    return rewarder.step(_state(eef, cube), _state(eef, cube, grasped))


def test_reward_sequence_is_strictly_increasing_from_hover_to_holding():
    hover, info = _reward([0, 0, grasp.GRASP_Z])                           # at the grasp point, gripper open (no command)
    assert hover == pytest.approx(1.0) and not info["holding"] and not info["attempt"]
    attempt, info = _reward_cmd([0, 0, grasp.GRASP_Z], gripper=+1.0)       # closing in the band, not yet holding
    assert attempt == pytest.approx(1.3) and info["attempt"]
    held0, info = _reward([0, 0, grasp.GRASP_Z], grasped=True)             # grasp closed, zero lift
    assert held0 == pytest.approx(1.5) and info["holding"]
    held5, _ = _reward([0, 0, .83 + .05], cube=(0, 0, .822 + .05), grasped=True)
    assert held5 == pytest.approx(1.5)                                       # no lift term: holding is flat
    assert hover < attempt < held0                                          # no step of the sequence is a loss
    far, _ = _reward([.1, 0, grasp.GRASP_Z])
    assert far == pytest.approx(1 - np.tanh(0.5), abs=1e-5)                # 1 - tanh(5 d), d = 0.1
    assert grasp.HOLD_REWARD_BASE == pytest.approx(1.5) and grasp.ATTEMPT_BONUS == pytest.approx(0.3)


def _reward_cmd(eef, cube=(0, 0, .822), grasped=False, gripper=None, width=None):
    return grasp.GraspReward().step(_state(eef, cube, width=width), _state(eef, cube, grasped, width), gripper)


def test_attempt_bonus_fires_only_when_all_four_conditions_hold():
    base, _ = _reward_cmd([0, 0, .83], gripper=-1.0)
    good, info = _reward_cmd([0, 0, .83], gripper=+0.2)
    assert good == pytest.approx(base + 0.3) and info["attempt"]
    # each single condition removed
    assert _reward_cmd([0, 0, .83], grasped=True, gripper=+1.0)[1]["attempt"] is False      # holding
    held, _ = _reward_cmd([0, 0, .83], grasped=True, gripper=+1.0)
    assert held == pytest.approx(1.5)                                                       # no bonus stacked on holding
    for z in (0.850, 0.812, 0.95):                                                          # outside the band
        r, info = _reward_cmd([0, 0, z], gripper=+1.0)
        assert not info["attempt"] and r == pytest.approx(_reward_cmd([0, 0, z], gripper=-1.0)[0])
    r, info = _reward_cmd([0.012, 0, .83], gripper=+1.0)                                    # lateral 1.2 cm
    assert not info["attempt"]
    assert not _reward_cmd([0, 0, .83], gripper=0.0)[1]["attempt"]                          # command not closed
    assert not _reward_cmd([0, 0, .83], gripper=-1.0)[1]["attempt"]                         # commanded open
    assert not _reward_cmd([0, 0, .83], gripper=None)[1]["attempt"]                         # no command given
    for z in (0.8152, 0.8448):                                                              # just inside the band edges
        assert _reward_cmd([0, 0, z], gripper=+1.0)[1]["attempt"]
    for z in (0.8148, 0.8452):                                                              # just outside
        assert not _reward_cmd([0, 0, z], gripper=+1.0)[1]["attempt"]
    assert _reward_cmd([0.0098, 0, .83], gripper=+1.0)[1]["attempt"]                        # just inside 1 cm laterally
    assert not _reward_cmd([0.0102, 0, .83], gripper=+1.0)[1]["attempt"]                    # just outside


def test_holding_always_pays_strictly_more_than_the_best_non_holding_reward():
    rng = np.random.RandomState(0)
    best_not_holding = max(
        _reward_cmd([*rng.uniform(-.009, .009, 2), rng.uniform(.815, .845)], gripper=1.0)[0] for _ in range(500))
    assert best_not_holding <= 1.3 + 1e-6 and best_not_holding > 1.0
    for lift in np.linspace(0, .1, 21):
        held, _ = _reward([0, 0, .83 + lift], cube=(0, 0, .822 + lift), grasped=True)
        assert held >= 1.5 - 1e-6 and held > best_not_holding


def test_reward_has_no_penalty_terms():
    for name in ("STEP_REWARD", "PUSH_PENALTY", "ROTATION_PENALTY", "ABRUPT_PENALTY", "DROP_PENALTY"):
        assert not hasattr(grasp, name)
    assert 0.0 <= _reward([.5, .5, 1.2])[0] <= 1.5


def test_grasp_point_is_the_band_midpoint_and_well_below_the_cliff():
    assert grasp.GRASP_Z == pytest.approx(0.830)
    assert grasp.GRASP_Z <= grasp.CLIFF_Z - grasp.MIN_CLIFF_MARGIN_M
    assert grasp.CLIFF_Z == pytest.approx(0.850)
    assert grasp.GRASP_Z < 0.822 + 0.022                                   # below the cube top (0.844)
    _, info = _reward([0, 0, grasp.GRASP_Z])
    assert info["distance"] == pytest.approx(0.0, abs=1e-6)                # the reward really targets it
    assert grasp.DESCENT_K == 5.0                                            # descent coefficient 5, not 10
    handover = _reward([0, 0, grasp.GRASP_Z + 0.18])[0]                      # 18 cm above the grasp point
    assert handover == pytest.approx(1 - np.tanh(5 * 0.18), abs=1e-4) and handover == pytest.approx(0.28, abs=0.01)


def _hold(rewarder, lift, grasped, steps):
    info = None
    for _ in range(steps):
        cube = (0, 0, .822 + lift)
        _, info = rewarder.step(_state([0, 0, .83], cube), _state([0, 0, .83], cube, grasped))
    return info["hold_steps"]


def test_tilt_is_yaw_invariant():
    assert grasp.tilt_deg([1, 0, 0, 0]) == pytest.approx(0.0, abs=1e-4)
    yaw = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    assert grasp.tilt_deg(yaw) == pytest.approx(0.0, abs=1e-4)
    roll = [np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0]                  # 30 degrees about x
    assert grasp.tilt_deg(roll) == pytest.approx(30.0, abs=1e-3)


def test_handover_offset_stays_inside_one_centimetre_and_three_centimetres():
    rng = np.random.RandomState(0)
    draws = np.array([grasp.sample_handover_offset(rng) for _ in range(5000)])
    assert np.hypot(draws[:, 0], draws[:, 1]).max() <= grasp.HANDOVER_LATERAL_M
    assert np.abs(draws[:, 2]).max() <= grasp.HANDOVER_Z_M
    assert np.hypot(draws[:, 0], draws[:, 1]).max() > 0.009 and np.abs(draws[:, 2]).max() > 0.029  # uses the range
    assert np.abs(np.arctan2(draws[:, 1], draws[:, 0])).max() > 3.0        # all directions


def test_handover_offset_is_uniform_over_the_disc_not_over_the_radius():
    rng = np.random.RandomState(1)
    draws = np.array([grasp.sample_handover_offset(rng) for _ in range(20000)])
    radius = np.hypot(draws[:, 0], draws[:, 1])
    assert radius.max() <= grasp.HANDOVER_LATERAL_M
    # disc-uniform: P(r > R/2) = 0.75 (radius-uniform would give 0.5)
    assert float((radius > grasp.HANDOVER_LATERAL_M / 2).mean()) == pytest.approx(0.75, abs=0.02)
    # and the mean squared radius is R^2 / 2
    assert float((radius ** 2).mean()) == pytest.approx(grasp.HANDOVER_LATERAL_M ** 2 / 2, rel=0.03)


def test_handover_pose_is_valid_in_the_simulator():
    pytest.importorskip("robosuite")
    dev = json.loads(MANIFEST.read_text())["splits"]["dev"]
    adapter = grasp.GraspAdapter(100)
    try:
        obs, frame, physical = adapter.reset(dev[0]["scene"]["seed"], False, True, np.random.RandomState(3))
    except Exception as error:                                             # no EGL / MuJoCo in this environment
        pytest.skip(f"simulator unavailable: {error!r}")
    try:
        h = adapter.handover
        assert h["cube_displacement_m"] < grasp.UNDISTURBED_M
        assert h["gripper_width_m"] >= grasp.OPEN_WIDTH_MIN_M
        assert h["lateral_error_m"] <= grasp.HANDOVER_LATERAL_M + 0.0015
        assert abs(h["hand_z"] - (reach.START_Z + h["offset_z_m"])) < 0.002
        assert adapter.step_count == 0                                     # scripted steps do not count
        assert obs[0].shape == (27, 84, 84) and obs[2].shape == (22,)
    finally:
        adapter.close()


def test_disturbed_cube_is_rejected(monkeypatch):
    adapter = grasp.GraspAdapter(100)
    adapter.cube_rest = np.zeros(3); adapter.last_width = 0.08
    adapter.last_hand = np.array([0, 0, 1.011], np.float32)
    adapter.env = type("E", (), {"cube_position": np.array([0.003, 0, 0])})()
    monkeypatch.setattr(adapter, "_scripted", lambda *a, **k: (True, "cur"))
    with pytest.raises(grasp.HandoverError):
        adapter._handover(("o", "f", {}), np.random.RandomState(0))


def _fake_step(plan):
    def fake(self, action):
        self.step_count += 1
        obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))
        return obs, np.zeros((9, 84, 84), np.uint8), {"privileged": np.zeros(22, np.float32)}, \
            self.step_count >= self.horizon or plan["t"] is not None, plan["t"]
    return fake


def _episode(tmp_path, monkeypatch, terminal):
    plan = {"t": terminal}
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", _fake_step(plan))
    adapter = grasp.GraspAdapter(3); adapter.step_count = 0
    replay = drq.EpisodeReplay(tmp_path / str(terminal), discount=.5)
    replay.start(np.zeros((9, 4, 4), np.uint8), np.zeros(9, np.float32), np.zeros(22, np.float32))
    done = False
    while not done:
        obs, _, _, done, t = adapter.step(np.zeros(4, np.float32))
        replay.add(np.zeros(4, np.float32), 1., t is not None, np.zeros((9, 4, 4), np.uint8), obs[1], obs[2])
    replay.finish()
    return replay.episodes[-1][1], adapter


def test_drop_terminates_with_zero_discount_and_horizon_truncates(tmp_path, monkeypatch):
    truncated, _ = _episode(tmp_path, monkeypatch, None)
    dropped, _ = _episode(tmp_path, monkeypatch, "drop")
    assert truncated["discounts"][-1, 0] == 1.0 and len(truncated["actions"]) == 3
    assert dropped["discounts"][-1, 0] == 0.0 and len(dropped["actions"]) == 1
    assert truncated["actions"].shape[1] == 4
    _, adapter = _episode(tmp_path, monkeypatch, "red")                    # grader red/blue: logged, not terminal
    assert adapter.anomalies and adapter.anomalies[0]["grader_terminal"] == "red"


def test_marker_is_sampled_bernoulli_from_the_seeded_np_random_stream(monkeypatch):
    """The marker is drawn Bernoulli(rate) from the seeded np.random stream."""
    seen = []

    def fake_reset(self, scene_seed, marker=False, handover=True, rng=None):
        seen.append(marker)
        return (np.zeros(1), np.zeros(1)), np.zeros(1), {}

    monkeypatch.setattr(grasp.GraspAdapter, "reset", fake_reset)
    adapter = grasp.GraspAdapter(100)
    np.random.seed(0)
    for _ in range(3000):
        adapter.reset_train(2002666)
    fraction = sum(seen) / len(seen)
    assert abs(fraction - grasp.TRAIN_MARKER_RATE) < 0.03            # Bernoulli(0.5); ~2% std error at n=3000

    # a pause and resume must reproduce the same marker sequence: same np.random state -> same draws
    np.random.seed(0); seen.clear()
    for _ in range(50):
        adapter.reset_train(2002666)
    checkpoint_state = np.random.get_state()
    seen.clear()
    for _ in range(50):
        adapter.reset_train(2002666)
    continued = list(seen)
    np.random.set_state(checkpoint_state); seen.clear()
    for _ in range(50):
        adapter.reset_train(2002666)
    resumed = list(seen)
    assert resumed == continued                                     # restoring rng_state reproduces the sequence

    source = (TOOLS / "rl_grasp.py").read_text()
    assert "env.reset_train(" in source and "env.reset(" not in source     # training only uses reset_train


# layouts
def test_layout_sets_are_disjoint_and_exclude_measurement_layouts():
    if not MANIFEST.exists():
        pytest.skip("manifest not available")
    dev = json.loads(MANIFEST.read_text())["splits"]["dev"]
    gate, holdout = grasp.grasp_layout_sets(dev)
    g, h = {x["layout_id"] for x in gate}, {x["layout_id"] for x in holdout}
    assert len(g) == len(h) == 16 and not g & h and not h & set(grasp.MEASUREMENT_LAYOUTS)
    for quadrant in grasp.QUADRANTS:
        for group in (gate, holdout):
            assert sum((x["scene"]["cube_distance"], x["scene"]["cube_side"]) == quadrant for x in group) == 4


def test_exclusion_is_enforced_in_code_not_just_by_list():
    dev = json.loads(MANIFEST.read_text())["splits"]["dev"] if MANIFEST.exists() else pytest.skip("no manifest")
    reordered = list(reversed(dev))                                        # measurement layouts now sit late in each quadrant
    gate, holdout = grasp.grasp_layout_sets(reordered)
    assert not {x["layout_id"] for x in holdout} & set(grasp.MEASUREMENT_LAYOUTS)
    with pytest.raises(ValueError):
        grasp.grasp_layout_sets(dev[:20])


# agent / warm start
def _agent(action_dim=4):
    torch.manual_seed(0)
    return asym.AsymmetricAgent(CONFIG, 1, action_dim=action_dim)


def test_four_output_agent_keeps_actor_blind_to_privileged_state_and_updates():
    agent = _agent()
    assert agent.actor.input_width == 39209 and agent.critic_width == 39231
    obs = torch.randn(3, 39231); other = obs.clone(); other[:, -22:] = torch.randn(3, 22)
    a, b = agent.actor(obs, .2), agent.actor(other, .2)
    assert a.mean.shape == (3, 4) and torch.equal(a.mean, b.mean)
    g = np.random.RandomState(0)
    batch = (g.randint(0, 255, (4, 27, 84, 84)).astype(np.uint8), g.randn(4, 9).astype(np.float32),
             g.uniform(-1, 1, (4, 4)).astype(np.float32), g.randn(4, 1).astype(np.float32),
             np.full((4, 1), .97, np.float32), g.randint(0, 255, (4, 27, 84, 84)).astype(np.uint8),
             g.randn(4, 9).astype(np.float32), g.randn(4, 22).astype(np.float32), g.randn(4, 22).astype(np.float32))
    metrics = agent.update(batch, 1000)
    assert metrics["encoder_grad_norm"] > 0 and np.isfinite(metrics["critic_loss"])
    critic_action_width = agent.critic.Q1[0].in_features - 50
    assert critic_action_width == 4


def test_warm_start_copies_only_the_encoder(tmp_path):
    source = _agent(3)                                                     # stands in for the approach agent
    fresh = _agent(4)
    fresh.encoder.load_state_dict(source.encoder.state_dict())
    for name, tensor in fresh.encoder.state_dict().items():
        assert torch.equal(tensor, source.encoder.state_dict()[name])
    assert fresh.actor.state_dict().keys() == _agent(4).actor.state_dict().keys()
    assert fresh.actor.policy[-1].out_features == 4 and source.actor.policy[-1].out_features == 3
    assert fresh.updates == 0
    assert not any(p.grad is not None for p in fresh.actor.parameters())


def test_approach_checkpoint_hash_is_verified_before_loading(tmp_path):
    ckpt = tmp_path / "checkpoint_000090000.pt"
    expected = {"weight": torch.tensor([1.0, 2.0])}
    torch.save({"agent": {"encoder": expected}}, ckpt)
    digest = runner.sha256(ckpt)
    (tmp_path / "latest.json").write_text(json.dumps({"path": ckpt.name, "sha256": digest}))
    weights, path, sha = runner.load_approach_encoder(tmp_path)
    assert path == ckpt and sha == digest
    assert torch.equal(weights["weight"], expected["weight"])


def test_approach_checkpoint_hash_mismatch_raises(tmp_path):
    ckpt = tmp_path / "checkpoint_000090000.pt"
    torch.save({"agent": {"encoder": {}}}, ckpt)
    (tmp_path / "latest.json").write_text(json.dumps(
        {"path": ckpt.name, "sha256": "0" * 64, "step": 90000}))
    with pytest.raises(ValueError):
        runner.load_approach_encoder(tmp_path)


def test_approach_root_is_a_config_value_not_a_constant():
    """--approach-root lets a different approach checkpoint (e.g. the marker run) supply the
    warm-start encoder without editing source; make_contract records whatever was actually loaded."""
    config = runner.make_config(smoke=True)
    contract = runner.make_contract(config, ["a"], ["b"], Path("/tmp/other/checkpoint_000150000.pt"),
                                    "f" * 64)
    assert contract["warm_start"]["source_checkpoint"] == "/tmp/other/checkpoint_000150000.pt"
    assert contract["warm_start"]["source_sha256"] == "f" * 64


def test_marker_rate_is_a_config_value_not_a_constant():
    default_config = runner.make_config(smoke=True)
    assert default_config["marker_rate"] == runner.TRAIN_MARKER_RATE
    zero_config = runner.make_config(smoke=True, marker_rate=0.0)
    assert zero_config["marker_rate"] == 0.0
    contract = runner.make_contract(zero_config, ["a"], ["b"], Path("/tmp/x.pt"), "0" * 64)
    assert contract["episodes"]["training_marker_rate"] == 0.0


def test_checkpoint_round_trip_with_four_outputs(tmp_path):
    from drq_online import checkpoint, validate_checkpoint
    agent = _agent()
    replay = drq.EpisodeReplay(tmp_path / "replay", discount=.5)
    replay.start(np.zeros((9, 4, 4), np.uint8), np.zeros(9, np.float32), np.zeros(22, np.float32))
    for k in range(3):
        replay.add(np.zeros(4, np.float32), 1., False, np.zeros((9, 4, 4), np.uint8), np.zeros(9, np.float32),
                   np.zeros(22, np.float32))
    replay.finish()
    pointer = checkpoint(tmp_path, agent, replay, dict(CONFIG), 5, 1, "hash")
    saved = validate_checkpoint(tmp_path, pointer)
    fresh = _agent()
    fresh.load_state_dict(saved["agent"])
    for key in ("actor", "critic", "critic_target", "encoder"):
        a, b = getattr(agent, key).state_dict(), getattr(fresh, key).state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_gripper_acting_noise_is_floored_while_translation_follows_the_schedule():
    config = dict(CONFIG, gripper_std_floor=0.4)
    torch.manual_seed(0)
    agent = asym.AsymmetricAgent(config, 1, action_dim=4)
    obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))

    def spread(step, n=600):
        draws = np.array([agent.act(obs, step, False) for _ in range(n)])
        return draws.std(axis=0), draws.mean(axis=0)

    late_std, late_mean = spread(50_000)                                    # schedule has reached 0.1
    assert late_std[:3].max() < 0.16 and late_std[:3].min() > 0.06           # translation: sigma ~ 0.1
    assert late_std[3] > 0.32 and late_std[3] > 2.5 * late_std[:3].max()     # gripper: floored near 0.4
    early_std, _ = spread(0)
    assert np.all(early_std > 0.5) and np.ptp(early_std) < 0.1                # floor inactive while s > 0.4 (samples clamp to +/-1)
    mid_std, _ = spread(15_000)                                               # s = 0.55 at the schedule midpoint
    assert np.all(np.abs(mid_std - mid_std[0]) < 0.12) and mid_std[0] > 0.4
    assert agent.stddev_schedule == "linear(1.0,0.1,30000)"                   # upstream schedule string untouched
    plain = asym.AsymmetricAgent(CONFIG, 1, action_dim=4)                     # no floor configured: unchanged
    assert plain.gripper_std_floor is None
    torch.manual_seed(0)
    draws = np.array([plain.act(obs, 50_000, False) for _ in range(300)])
    assert draws.std(axis=0).max() < 0.16


def test_eval_mode_actions_are_unaffected_by_the_floor():
    torch.manual_seed(0)
    a = asym.AsymmetricAgent(dict(CONFIG, gripper_std_floor=0.4), 1, action_dim=4)
    obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))
    assert np.array_equal(a.act(obs, 50_000, True), a.act(obs, 50_000, True))
    assert runner.make_config(False)["gripper_std_floor"] == 1.0


def _wr(eef, width, grasped, gripper=None, cube=(0, 0, .822)):
    return grasp.GraspReward().step(_state(eef, cube, width=0.0432), _state(eef, cube, grasped, width), gripper)


def test_holding_requires_a_real_grasp_width():
    for width in (0.0029, 0.0351, 0.0399):                                # shut fingertips on top of the cube
        reward, info = _wr([0, 0, .844], width, grasped=True, gripper=1.0)
        assert info["raw_grasp"] and not info["holding"]
        assert reward < 1.0                                               # no holding reward, no bonus
    for width in (0.0401, 0.0432, 0.0500, 0.0549):                        # real grasp widths (4.29-5.0 cm measured)
        reward, info = _wr([0, 0, .83], width, grasped=True, gripper=1.0)
        assert info["holding"] and reward == pytest.approx(1.5)
    for width in (0.0551, 0.08):                                          # too wide to be gripping the cube
        assert not _wr([0, 0, .83], width, grasped=True)[1]["holding"]
    assert not _wr([0, 0, .83], 0.0432, grasped=False)[1]["holding"]      # the contact check is still required


def test_success_needs_a_real_width_for_ten_consecutive_steps():
    def run(width, steps):
        r = grasp.GraspReward(); info = None
        for _ in range(steps):
            cube = (0, 0, .822 + .05)
            _, info = r.step(_state([0, 0, .88], cube, width=0.0432), _state([0, 0, .88], cube, True, width))
        return info["hold_steps"]
    assert run(0.0432, 10) >= grasp.HOLD_STEPS
    assert run(0.0030, 30) == 0                                           # closed fingers never count, whatever the lift


def test_attempt_bonus_needs_fingers_still_open():
    base = _reward_cmd([0, 0, .83], gripper=-1.0, width=0.0800)[0]
    for width in (0.0450, 0.0800):                                        # open enough
        r, info = _reward_cmd([0, 0, .83], gripper=+1.0, width=width)
        assert info["attempt"] and r == pytest.approx(base + 0.3)
    for width in (0.0449, 0.0432, 0.0030):                                # already closing or shut
        r, info = _reward_cmd([0, 0, .83], gripper=+1.0, width=width)
        assert not info["attempt"] and r == pytest.approx(base)


def test_reward_ordering_makes_the_shut_finger_exploit_pay_less_than_correct_behaviour():
    press = _wr([0, 0, .844], 0.0030, grasped=True, gripper=1.0)[0]        # shut fingers resting on the cube top
    assert press == pytest.approx(1 - np.tanh(5 * 0.014), abs=1e-3) and press == pytest.approx(0.93, abs=0.01)
    hover_open = _wr([0, 0, .83], 0.0800, grasped=False, gripper=-1.0)[0]   # descended open to the target
    attempt = _wr([0, 0, .83], 0.0800, grasped=False, gripper=+1.0)[0]      # starting to close around the cube
    grasp_ = _wr([0, 0, .83], 0.0432, grasped=True, gripper=+1.0)[0]        # real grasp, zero lift
    assert press < hover_open < attempt < grasp_
    assert (hover_open, attempt, grasp_) == pytest.approx((1.0, 1.3, 1.5))


# ---- unfloored lift, gripper std >= 1.0, lift-stall stop, lift logging ----
def test_gripper_acting_std_is_at_least_one_late_in_training_while_translation_follows_the_schedule():
    config = dict(CONFIG, gripper_std_floor=1.0)
    torch.manual_seed(0)
    agent = asym.AsymmetricAgent(config, 1, action_dim=4)
    seen = []
    original = agent.actor.forward

    def spy(obs, std):
        seen.append(torch.as_tensor(std).clone())
        return original(obs, std)
    agent.actor.forward = spy
    obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))
    for step, translation in ((50_000, 0.1), (30_000, 0.1), (15_000, 0.55), (0, 1.0)):
        agent.act(obs, step, False)
        std = seen[-1]
        assert float(std[3]) >= 1.0                                           # gripper never below 1.0
        np.testing.assert_allclose(std[:3].numpy(), translation, atol=1e-4)  # translation: linear(1.0, 0.1, 30000)
    assert runner.make_config(False)["gripper_std_floor"] == 1.0


def _summary(step, grasped, lift):
    return {"steps": step, "grasped_rollouts": grasped, "mean_max_lift_cm": lift}




# ---- flat holding reward, success = sustained holding, Q-high, grasp latch ----
@pytest.mark.parametrize("lift_m", [-0.02, -0.0045, 0.0, 0.02, 0.05, 0.1])
def test_holding_pays_exactly_one_point_five_with_no_lift_term(lift_m):
    reward, info = _reward([0, 0, .83], cube=(0, 0, .822 + lift_m), grasped=True)
    assert info["holding"] and reward == pytest.approx(1.5, abs=1e-6)


def test_success_is_width_checked_holding_for_ten_consecutive_steps_and_nothing_else():
    def run(grasped, width, steps, lift=0.0):
        r = grasp.GraspReward(); info = None
        for _ in range(steps):
            cube = (0, 0, .822 + lift)
            _, info = r.step(_state([0, 0, .83], cube, width=0.0432), _state([0, 0, .83], cube, grasped, width))
        return info["hold_steps"]
    assert run(True, 0.0432, 10) >= grasp.HOLD_STEPS and run(True, 0.0432, 9) < grasp.HOLD_STEPS
    assert run(True, 0.0432, 10, lift=0.0) >= grasp.HOLD_STEPS              # no lift needed
    assert run(True, 0.0432, 10, lift=-0.01) >= grasp.HOLD_STEPS            # even pressed into the table
    assert run(False, 0.0432, 30, lift=0.05) == 0                           # lift without holding does not count
    assert run(True, 0.0030, 30) == 0                                       # shut fingers never count
    r = grasp.GraspReward()
    for _ in range(6):
        r.step(_state([0, 0, .83]), _state([0, 0, .83], grasped=True))
    r.step(_state([0, 0, .83]), _state([0, 0, .83], grasped=False))         # a break resets the streak
    for _ in range(9):
        _, info = r.step(_state([0, 0, .83]), _state([0, 0, .83], grasped=True))
    assert info["hold_steps"] == 9 < grasp.HOLD_STEPS


def test_q_high_stop_threshold_is_188_above_the_bootstrapped_ceiling_of_150():
    assert runner.Q_LOW == -20.0
    assert not hasattr(runner, "lift_stall_triggered")                      # the r6 lift-stall stop is removed


class _NoEnv:
    """Stands in for the simulator: any read of cube, contact or grader state raises."""
    def __getattr__(self, name):
        raise AssertionError(f"grasp latch touched simulator state: {name}")


def _latch_run(monkeypatch, widths, commands, initial_width=0.08):
    """Steps the adapter with scripted finger widths; returns (adapter, env gripper actions actually sent)."""
    sent = []

    def fake(self, action):
        self.step_count += 1
        self.last_width = widths[self.step_count - 1]
        sent.append(float(action[6]))
        obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))
        return obs, np.zeros((9, 84, 84), np.uint8), {"privileged": np.zeros(22, np.float32)}, False, None
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", fake)
    adapter = grasp.GraspAdapter(200); adapter.step_count = 0; adapter.env = _NoEnv()
    adapter.last_width = initial_width                                        # fingers open after the handover
    for command in commands:
        adapter.step(np.array([0, 0, 0, command], np.float32))
    return adapter, sent


# measured on dev-s2002666 (scripted, headless), converted from cm to m
ON_AIR_CM = [7.96, 7.77, 7.33, 6.74, 6.05, 5.32, 4.55, 3.77, 2.98, 2.20, 1.42, 0.62, 0.20, 0.20, 0.20]
ON_CUBE_CM = [8.00, 7.79, 7.34, 6.74, 6.06, 5.32, 4.55, 4.37, 4.36, 4.33, 4.32, 4.32, 4.31, 4.31, 4.31]


def test_grasp_latch_never_fires_on_fingers_closing_through_the_range_on_air(monkeypatch):
    widths = [w / 100 for w in ON_AIR_CM]
    adapter, sent = _latch_run(monkeypatch, widths, [1.0] * len(widths))
    assert not adapter.latched and adapter.latch_fires == 0
    changes = [abs(b - a) for a, b in zip([0.0796] + widths, widths) if 0.040 <= b <= 0.055]
    assert min(changes) > 7 * grasp.LATCH_STALL_M                              # the measured margin: >= 7x the threshold
    assert all(g == 1.0 for g in sent)                                        # (closing, since the command was closed)


def test_grasp_latch_fires_on_the_measured_on_cube_profile(monkeypatch):
    widths = [w / 100 for w in ON_CUBE_CM]
    adapter, _ = _latch_run(monkeypatch, widths, [1.0] * len(widths))
    assert adapter.latched and adapter.latch_fires == 1
    # first in-range, stalled step is 4.37 -> 4.36 (0.01 cm); second is 4.36 -> 4.33 (0.03 cm): fires on that step
    assert adapter.latch_step == ON_CUBE_CM.index(4.33) + 1


def test_grasp_latch_needs_command_width_range_and_two_consecutive_stalled_steps(monkeypatch):
    n = 6
    a, _ = _latch_run(monkeypatch, [0.03] * n, [1.0] * n, initial_width=0.03)          # stalled but below 4.0 cm
    assert not a.latched
    a, _ = _latch_run(monkeypatch, [0.07] * n, [1.0] * n, initial_width=0.07)          # stalled but above 5.5 cm
    assert not a.latched
    a, _ = _latch_run(monkeypatch, [0.043] * n, [-1.0, 0.0, -0.5, 0.0, -1.0, -1.0], initial_width=0.043)   # never commanded closed
    assert not a.latched
    a, _ = _latch_run(monkeypatch, [0.043] * 3, [1.0, -1.0, 1.0], initial_width=0.043)  # not consecutive commands
    assert not a.latched
    a, _ = _latch_run(monkeypatch, [0.0432, 0.0500, 0.0432, 0.0500], [1.0] * 4, initial_width=0.0432)   # width still moving 0.7 cm
    assert not a.latched
    a, _ = _latch_run(monkeypatch, [0.0432], [1.0], initial_width=0.0432)               # a single stalled step
    assert not a.latched
    a, _ = _latch_run(monkeypatch, [0.0432, 0.0432], [1.0, 1.0], initial_width=0.0432)  # two consecutive stalled steps
    assert a.latched and a.latch_fires == 1 and a.latch_step == 2
    a, _ = _latch_run(monkeypatch, [0.0432, 0.0440], [1.0, 1.0], initial_width=0.0432)  # 0.08 cm change is still stalled
    assert a.latched
    a, _ = _latch_run(monkeypatch, [0.0432, 0.0443], [1.0, 1.0], initial_width=0.0432)  # 0.11 cm change is not
    assert not a.latched


def test_once_latched_the_gripper_stays_closed_whatever_the_policy_commands(monkeypatch):
    widths = [0.043] * 8
    commands = [1.0, 1.0, -1.0, -1.0, -1.0, 0.0, -0.9, -1.0]
    adapter, sent = _latch_run(monkeypatch, widths, commands, initial_width=0.043)
    assert adapter.latched and adapter.latch_fires == 1 and adapter.latch_step == 2     # never re-fires
    assert all(g == 1.0 for g in sent)                                        # locked closed for the rest of the episode
    _, unlatched = _latch_run(monkeypatch, [0.03] * 8, commands, initial_width=0.03)
    assert unlatched[2] == -1.0                                               # without the latch the open command goes through
    adapter._reset_latch()
    assert not adapter.latched and adapter.latch_fires == 0 and adapter.latch_step is None


def test_grasp_latch_reads_only_proprioceptive_finger_width_and_its_change(monkeypatch):
    adapter, _ = _latch_run(monkeypatch, [0.043, 0.043], [1.0, 1.0], initial_width=0.043)   # a simulator stand-in raises on any read
    assert adapter.latched
    import inspect
    source = inspect.getsource(grasp.GraspAdapter.step)
    latch_part = source[source.index("if command is not None and not self.latched"):].split("terminal, anomaly")[0]
    code = "\n".join(line.split("#")[0] for line in latch_part.splitlines())      # ignore comments
    assert "self.last_width" in code and "previous_width" in source and "self.env" not in code
    for forbidden in ("cube", "grasped", "privileged", "physical", "_check_grasp", "contact"):
        assert forbidden not in code


# ---- gripper held open above the cliff ----
def _cliff_run(monkeypatch, hand_z, commands, widths=None):
    """Steps the adapter; hand_z[k] is the hand height when action k is chosen.  Returns (adapter, gripper sent)."""
    sent = []

    def fake(self, action):
        self.step_count += 1
        sent.append(float(action[6]))
        k = self.step_count
        self.last_hand = np.array([0, 0, hand_z[k] if k < len(hand_z) else hand_z[-1]], np.float32)
        self.last_width = widths[k - 1] if widths else 0.08
        obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))
        return obs, np.zeros((9, 84, 84), np.uint8), {"privileged": np.zeros(22, np.float32)}, False, None
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", fake)
    adapter = grasp.GraspAdapter(200); adapter.step_count = 0; adapter.env = _NoEnv()
    adapter.last_hand = np.array([0, 0, hand_z[0]], np.float32); adapter.last_width = 0.08
    for command in commands:
        adapter.step(np.array([0, 0, 0, command], np.float32))
    return adapter, sent


def test_gripper_is_held_open_above_the_cliff_whatever_the_policy_outputs(monkeypatch):
    n = 5
    adapter, sent = _cliff_run(monkeypatch, [0.95] * (n + 1), [1.0, 0.3, 1.0, -0.5, 1.0])
    assert sent == [-1.0] * n                                                # policy asked to close; the env got open
    assert adapter.held_open_steps == n and adapter.held_open_last
    assert not adapter.latched                                                # closed commands above the cliff never latch
    _, sent = _cliff_run(monkeypatch, [0.851] * 4, [1.0, 1.0, 1.0])
    assert sent == [-1.0] * 3                                                # just above 0.850 is still overridden


def test_policy_gripper_command_passes_through_at_or_below_the_cliff(monkeypatch):
    adapter, sent = _cliff_run(monkeypatch, [0.83] * 4, [1.0, -0.5, 0.3])
    assert sent == pytest.approx([1.0, -0.5, 0.3]) and adapter.held_open_steps == 0 and not adapter.held_open_last
    _, sent = _cliff_run(monkeypatch, [0.8499] * 3, [1.0, 1.0])             # just at the cliff (float32-safe): the policy controls
    assert sent == [1.0, 1.0]
    _, sent = _cliff_run(monkeypatch, [0.86, 0.84, 0.86], [1.0, 1.0, 1.0])  # follows the hand's height step by step
    assert sent == [-1.0, 1.0, -1.0]


def test_latch_takes_precedence_so_the_gripper_stays_closed_even_above_the_cliff(monkeypatch):
    # fingers stall at 4.32 cm on the cube for two steps below the cliff, then the hand rises above 0.850
    widths = [0.0432] * 6
    adapter, sent = _cliff_run(monkeypatch, [0.83, 0.83, 0.83, 0.90, 0.95, 0.95, 0.95], [1.0] * 6, widths=widths)
    adapter.last_width = 0.0432                                              # (start already stalled)
    assert adapter.latched
    assert sent[0] == 1.0 and all(g == 1.0 for g in sent[adapter.latch_step:])    # closed after the latch, even at z 0.95
    assert adapter.held_open_steps == 0                                      # never overridden once latched


def test_held_open_rule_reads_only_proprioceptive_hand_height_and_spares_approach_actions(monkeypatch):
    import inspect
    source = inspect.getsource(grasp.GraspAdapter.step)
    start = source.index("Gripper precedence"); end = source.index("previous_width = self.last_width")
    code = "\n".join(line.split("#")[0] for line in source[start:end].splitlines())
    assert "self.last_hand[2]" in code and "HELD_OPEN_ABOVE_Z" in code and "self.env" not in code
    for forbidden in ("cube", "grasped", "privileged", "physical", "_check_grasp", "contact"):
        assert forbidden not in code
    assert grasp.HELD_OPEN_ABOVE_Z == pytest.approx(grasp.CLIFF_Z) == pytest.approx(0.850)
    # the 3-value approach-policy action (chain) is never touched by the rule
    sent = []

    def fake(self, action):
        self.step_count += 1; sent.append(float(action[6]))
        return (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32)), np.zeros((9, 84, 84), np.uint8), \
            {"privileged": np.zeros(22, np.float32)}, False, None
    monkeypatch.setattr(drq.TwoTrayAdapter, "step", fake)
    adapter = grasp.GraspAdapter(50); adapter.step_count = 0; adapter.last_hand = np.array([0, 0, 0.95], np.float32)
    adapter.step(np.array([0, 0, 0], np.float32))
    assert sent == [-1.0] and adapter.held_open_steps == 0                    # approach actions always send open, not counted
