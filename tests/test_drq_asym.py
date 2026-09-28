"""CPU contract checks for the asymmetric-critic DrQ-v2 approach experiment."""
import importlib
import os
import random
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
rewards = importlib.import_module("drq_approach_rewards")
run_online = importlib.import_module("drq_online")

CONFIG = {"lr": 1e-4, "num_expl_steps": 10, "stddev_schedule": "linear(1.0,0.1,50000)",
          "device": "cpu"}


def _agent(stage):
    torch.manual_seed(0)
    return asym.AsymmetricAgent(CONFIG, stage)


def _batch(n=4, seed=0):
    g = np.random.RandomState(seed)
    return (g.randint(0, 255, (n, 27, 84, 84)).astype(np.uint8), g.randn(n, 9).astype(np.float32),
            g.uniform(-1, 1, (n, 3)).astype(np.float32), g.randn(n, 1).astype(np.float32),
            np.full((n, 1), .97, np.float32),
            g.randint(0, 255, (n, 27, 84, 84)).astype(np.uint8), g.randn(n, 9).astype(np.float32),
            g.randn(n, 22).astype(np.float32), g.randn(n, 22).astype(np.float32))


def test_actor_output_is_bit_identical_when_only_privileged_state_changes():
    agent = _agent(1)
    assert agent.actor.input_width == agent.encoder.repr_dim + 9 == 39209
    obs = torch.randn(3, 39209 + 22)
    other = obs.clone(); other[:, -22:] = torch.randn(3, 22)
    a, b = agent.actor(obs, .2), agent.actor(other, .2)
    assert torch.equal(a.mean, b.mean)
    # and the act() path never even builds the privileged columns in stage 1
    assert agent._actor_input((np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))).shape[-1] == 39209


def test_critic_output_changes_when_only_privileged_state_changes():
    agent = _agent(1)
    assert agent.critic_width == 39231
    obs = torch.randn(3, 39231); action = torch.zeros(3, 3)
    other = obs.clone(); other[:, -22:] += 1.0
    assert not torch.equal(agent.critic(obs, action)[0], agent.critic(other, action)[0])


def test_sliced_actor_state_dict_keys_match_upstream_actor():
    plain = drq.upstream.Actor(39209, (3,), 50, 1024)
    assert plain.state_dict().keys() == asym.SlicedActor(39209, (3,), 50, 1024).state_dict().keys()


def test_stage0_has_no_image_path_and_31_wide_inputs():
    agent = _agent(0)
    assert agent.actor.input_width == 31 and agent.critic_width == 31
    calls = []
    agent.encoder.register_forward_hook(lambda *a: calls.append(1))
    agent.act((np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32), np.zeros(22, np.float32)), 100, True)
    agent.update(_batch(), 1000)
    assert not calls
    assert all(p.grad is None for p in agent.encoder.parameters())


def test_encoder_parameters_receive_nonzero_gradient_from_critic_update():
    agent = _agent(1)
    metrics = agent.update(_batch(), 1000)
    assert metrics["encoder_grad_norm"] > 0
    assert metrics["critic_q1_max"] >= metrics["critic_q1"]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.encoder.parameters())


@pytest.mark.parametrize("stage", [0, 1])
def test_target_critic_uses_next_step_privileged_state(stage):
    agent = _agent(stage); seen = []
    agent.critic_target.register_forward_hook(lambda m, args, out: seen.append(args[0].detach().clone()))
    batch = _batch()
    agent.update(batch, 2)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][:, -22:].numpy(), batch[8])
    assert not np.array_equal(seen[0][:, -22:].numpy(), batch[7])
    np.testing.assert_array_equal(seen[0][:, -22 - 9:-22].numpy(), batch[6])


def _priv(v):
    return np.full((22,), v, np.float32)


def _priv_episode(tmp_path, n=4):
    replay = drq.EpisodeReplay(tmp_path, discount=.5)
    replay.start(np.zeros((9, 4, 4), np.uint8), np.zeros(9, np.float32), _priv(0))
    for k in range(n):
        replay.add(np.zeros(3, np.float32), 1., False, np.full((9, 4, 4), k + 1, np.uint8),
                   np.full(9, k + 1, np.float32), _priv(k + 1))
    replay.finish()
    return replay


@pytest.mark.parametrize("position,end", [(0, 3), (1, 4), (3, 4)])
def test_replay_returns_privileged_state_at_nstep_end_including_tail(tmp_path, monkeypatch, position, end):
    replay = _priv_episode(tmp_path)
    monkeypatch.setattr(drq.np.random, "randint", lambda high, size: np.full(size, position))
    out = replay.sample(1)
    assert len(out) == 9
    np.testing.assert_array_equal(out[7][0], _priv(position))
    np.testing.assert_array_equal(out[8][0], _priv(end))
    np.testing.assert_array_equal(out[6][0], np.full(9, end, np.float32))  # aligned with next_state


def test_action_wrapper_scale_rotation_gripper_and_bounds():
    scale = reach.translation_scale()
    assert scale == pytest.approx(reach.TRANSLATION_CAP_M / reach.controller_output_max_translation())
    action = reach.env_action([1.0, -1.0, .5], scale)
    assert action.shape == (7,) and action.dtype == np.float32
    np.testing.assert_allclose(action[:3], np.array([1, -1, .5]) * scale)
    assert not action[3:6].any() and action[6] == -1.0
    low, high = -np.ones(7), np.ones(7)
    assert np.all(action >= low) and np.all(action <= high)
    assert np.all(reach.env_action([5, -5, 0], scale)[:3] == np.array([1, -1, 0]) * np.float32(scale))
    with pytest.raises(ValueError):
        reach.env_action([0, 0, 0, 0], scale)


Z0 = reach.START_Z
CUBE = (0, 0, .822)


def _rstate(eef, cube=CUBE):
    return {"eef": np.array(eef, np.float32), "cube": np.array(cube, np.float32), "grasped": False}


def _reward(eef, cube=CUBE, before=None):
    r = reach.ReachReward(Z0)
    return r.step(_rstate(before if before is not None else eef, cube), _rstate(eef, cube))


@pytest.mark.parametrize("d", [0.134, 0.05, 0.02, 0.01])
def test_reward_is_tanh_of_distance_to_point_above_cube(d):
    # hand d metres from the target, laterally, well outside the ready cylinder except d=0.01
    reward, info = _reward([d, 0, Z0])
    assert info["distance"] == pytest.approx(d, abs=1e-6)
    expected = 1 - np.tanh(10 * d) + (reach.READY_BONUS if d <= reach.READY_LATERAL_M else 0.0)
    assert reward == pytest.approx(expected, abs=1e-5)


def test_reward_table_values_and_bonus():
    assert _reward([0.134, 0, Z0])[0] == pytest.approx(0.128, abs=1e-3)
    assert _reward([0.05, 0, Z0])[0] == pytest.approx(0.538, abs=1e-3)
    assert _reward([0.0, 0, Z0])[0] == pytest.approx(1.5)                    # on target: 1 + bonus
    reward, info = _reward([0.011, 0, Z0])                                   # just outside the cylinder
    assert not info["ready"] and reward == pytest.approx(1 - np.tanh(0.11), abs=1e-5)


def test_reward_has_no_potential_penalty_or_step_terms():
    closer, _ = _reward([.1, 0, Z0], before=[.2, 0, Z0])
    farther, _ = _reward([.1, 0, Z0], before=[.0, 0, Z0])
    assert closer == pytest.approx(1 - np.tanh(1.0), abs=1e-5) == pytest.approx(farther, abs=1e-5)
    for name in ("STEP_REWARD", "SUCCESS_REWARD", "ROTATION_PENALTY", "ABRUPT_PENALTY", "PUSH_PENALTY"):
        assert not hasattr(reach, name)


def test_target_is_far_above_the_cube():
    target = reach.reach_target_point(CUBE, Z0)
    assert target[2] - (.822 + .022) > .15
    np.testing.assert_allclose(target[:2], CUBE[:2])


def test_ready_is_a_cylinder_with_no_displacement_term():
    assert _reward([.009, 0, Z0 + .02])[1]["ready"]
    assert not _reward([.011, 0, Z0])[1]["ready"]
    assert not _reward([0, 0, Z0 + .04])[1]["ready"]
    assert not _reward([0, 0, Z0 - .04])[1]["ready"]
    r = reach.ReachReward(Z0)
    r.step(_rstate([0, 0, Z0], CUBE), _rstate([0, 0, Z0], CUBE))
    moved = (.03, 0, .822)                                                   # cube displaced 3 cm
    reward, info = r.step(_rstate([.03, 0, Z0], moved), _rstate([.03, 0, Z0], moved))
    assert info["ready"] and info["displacement_anomaly"]                    # logged, not disqualifying
    fast = reach.ReachReward(Z0).step(_rstate([0, 0, Z0 + .01]), _rstate([0, 0, Z0]))[1]   # 0.2 m/s
    assert not fast["ready"]
    grasped = dict(_rstate([0, 0, Z0]), grasped=True)
    assert not reach.ReachReward(Z0).step(grasped, grasped)[1]["ready"]


def test_ready_tolerance_is_inside_gripper_clearance():
    assert reach.READY_LATERAL_M < (0.0796 - 0.044) / 2


def test_start_height_assertion_and_horizon():
    assert reach.check_start_height(1.0115) == pytest.approx(1.0115)
    with pytest.raises(ValueError):
        reach.check_start_height(1.02)
    assert reach.HORIZON == 100


def test_ready_hold_counts_consecutive_steps():
    r = reach.ReachReward(Z0)
    for k in range(10):
        _, info = r.step(_rstate([0, 0, Z0]), _rstate([0, 0, Z0]))
    assert info["hold_steps"] == 10
    _, info = r.step(_rstate([0, 0, Z0]), _rstate([.05, 0, Z0]))
    assert info["hold_steps"] == 0


def test_horizon_truncates_but_drop_terminates(tmp_path, monkeypatch):
    plan = {}

    def fake_step(self, action):
        self.step_count += 1
        obs = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32))
        return obs, np.zeros((9, 84, 84), np.uint8), {"privileged": _priv(0)}, \
            self.step_count >= self.horizon or plan.get("t") is not None, plan.get("t")

    monkeypatch.setattr(drq.TwoTrayAdapter, "step", fake_step)

    def episode(terminal):
        plan["t"] = terminal
        adapter = reach.ReachAdapter(3); adapter.step_count = 0
        replay = drq.EpisodeReplay(tmp_path / str(terminal), discount=.5)
        replay.start(np.zeros((9, 4, 4), np.uint8), np.zeros(9, np.float32), _priv(0))
        done = False
        while not done:
            obs, _, _, done, t = adapter.step(np.zeros(3))
            replay.add(np.zeros(3, np.float32), 1., t is not None, np.zeros((9, 4, 4), np.uint8),
                       obs[1], obs[2])
        replay.finish()
        return replay.episodes[-1][1]["discounts"][-1, 0], adapter

    truncated, _ = episode(None)
    dropped, _ = episode("drop")
    assert truncated == 1.0 and dropped == 0.0
    _, adapter = episode("red")
    assert adapter.anomalies and adapter.anomalies[0]["grader_terminal"] == "red"


class _StrictEnv:
    cube_body_id = 0
    _allowed = {"cube_position", "sim", "cube", "tray_center"}

    def __init__(self):
        from types import SimpleNamespace
        self.cube_position = np.array([.1, .2, .822])
        self.cube = SimpleNamespace(root_body="cube")
        data = SimpleNamespace(body_xquat=np.array([[1., 0, 0, 0]]),
                               get_body_xvelp=lambda name: np.array([.01, 0, 0]))
        self.sim = SimpleNamespace(data=data)

    def tray_center(self, which):
        return np.array([.3, 0., .803]) if which == "red" else np.array([.3, -.2, .803])

    def __getattr__(self, name):
        raise AssertionError(f"privileged_state touched forbidden field {name}")


def test_privileged_vector_has_22_values_and_no_marker_or_grader_fields():
    vector = reach.privileged_state(_StrictEnv(), [0., 0., 1.0])
    assert vector.shape == (22,) and vector.dtype == np.float32
    np.testing.assert_allclose(vector[:3], np.array([.1, .2, .022]) / .5, atol=1e-6)
    np.testing.assert_allclose(vector[10:13], np.array([-.1, -.2, .178]) / .5, atol=1e-6)   # hand - cube
    np.testing.assert_allclose(vector[19:22], np.array([-.2, .2, .019]) / .5, atol=1e-6)    # cube - red


def _equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


@pytest.mark.parametrize("stage", [0, 1])
def test_checkpoint_round_trip_reproduces_models_optimizers_replay_and_rng(tmp_path, stage):
    agent = _agent(stage)
    for i in range(3):
        agent.update(_batch(seed=i), 2 * (i + 1))
    replay = _priv_episode(tmp_path / "replay")
    config = dict(CONFIG, stage=stage)
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    pointer = run_online.checkpoint(tmp_path, agent, replay, config, 6, 1, "hash")
    expected = (random.random(), np.random.rand(), torch.rand(1))
    saved = run_online.validate_checkpoint(tmp_path, pointer)
    fresh = _agent(stage)
    fresh.load_state_dict(saved["agent"])
    assert _equal(agent.state_dict(), fresh.state_dict())
    assert {"encoder", "encoder_opt"} <= saved["agent"].keys()
    restored = drq.EpisodeReplay(tmp_path / "replay")
    restored.load(saved["replay"])
    assert _equal(replay.manifest(), restored.manifest())
    assert all(_equal(a[1], b[1]) for a, b in zip(replay.episodes, restored.episodes))
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    drq.restore_rng(saved["rng"])
    assert (random.random(), np.random.rand()) == expected[:2] and torch.equal(torch.rand(1), expected[2])


def test_gripper_noise_floor_applies_at_three_and_four_outputs():
    """The floor is on the last action. It was silently skipped once an agent went to 3 outputs."""
    import numpy as np
    config = {"lr": 1e-4, "num_expl_steps": 0, "stddev_schedule": "linear(1.0,0.1,30000)",
              "gripper_std_floor": 1.0, "device": "cpu"}
    for action_dim in (3, 4):
        agent = asym.AsymmetricAgent(config, 1, action_dim=action_dim)
        late = agent.acting_std(60_000)                       # schedule has decayed to 0.1
        assert late[:-1] == pytest.approx([0.1] * (action_dim - 1))
        assert late[-1] == pytest.approx(1.0)                 # the gripper keeps exploring
        early = agent.acting_std(0)
        assert early == pytest.approx([1.0] * action_dim)


def test_acting_noise_is_measurable_from_sampled_actions():
    """Sample real actions, not just the config, to catch a floor that a config-only check would miss."""
    import numpy as np
    import torch
    config = {"lr": 1e-4, "num_expl_steps": 0, "stddev_schedule": "linear(1.0,0.1,30000)",
              "gripper_std_floor": 1.0, "device": "cpu"}
    agent = asym.AsymmetricAgent(config, 1, action_dim=3)
    observation = (np.zeros((27, 84, 84), np.uint8), np.zeros(9, np.float32),
                   np.zeros(asym.PRIVILEGED_DIM, np.float32))
    samples = np.array([agent.act(observation, 60_000, eval_mode=False) for _ in range(400)])
    spread = samples.std(axis=0)
    assert spread[0] < 0.4 and spread[1] < 0.4                # translation noise has decayed
    assert spread[2] > 0.5                                    # the gripper has not
