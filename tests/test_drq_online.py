import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest


# The adapter deliberately imports the pinned upstream DrQ-v2 checkout.  Keep
# this test import pointed at that checkout rather than replacing it with a
# fake learner or a local reimplementation of the upstream module.
TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_upstream = os.environ.get("DRQV2_UPSTREAM")
if not _upstream or not Path(_upstream).is_dir():
    pytest.skip("set DRQV2_UPSTREAM to the pinned native WSL checkout", allow_module_level=True)
drq = importlib.import_module("drq_online")


def _frame(value: int) -> np.ndarray:
    return np.full((9, 4, 4), value, dtype=np.uint8)


def _state(value: float) -> np.ndarray:
    return np.full((9,), value, dtype=np.float32)


def _action(value: float) -> np.ndarray:
    return np.full((7,), value, dtype=np.float32)


def _episode(
    tmp_path: Path,
    rewards: list[float],
    terminals: list[bool] | None = None,
    *,
    discount: float = 0.5,
    capacity: int = 30_000,
) -> drq.EpisodeReplay:
    if terminals is None:
        terminals = [False] * len(rewards)
    assert len(terminals) == len(rewards)
    replay = drq.EpisodeReplay(tmp_path, capacity=capacity, discount=discount)
    replay.start(_frame(0), _state(0))
    for index, (reward, terminal) in enumerate(zip(rewards, terminals)):
        replay.add(
            _action(index),
            reward,
            terminal,
            _frame(index + 1),
            _state(index + 1),
        )
    replay.finish()
    return replay


def _sample_one(monkeypatch, replay, position: int):
    monkeypatch.setattr(
        drq.np.random,
        "randint",
        lambda high, size: np.full(size, position, dtype=np.int64),
    )
    return replay.sample(1)


def test_replay_computes_three_step_return_and_stacks_boundary_frames(tmp_path, monkeypatch):
    replay = _episode(tmp_path, [1.0, 2.0, 3.0, 4.0])

    observation, state, action, reward, discount, next_observation, next_state = _sample_one(
        monkeypatch, replay, 0
    )

    assert observation.shape == (1, 27, 4, 4)
    assert next_observation.shape == (1, 27, 4, 4)
    np.testing.assert_array_equal(observation[0, :3], _frame(0)[:3])
    np.testing.assert_array_equal(observation[0, 3:6], _frame(0)[:3])
    np.testing.assert_array_equal(observation[0, 6:9], _frame(0)[:3])
    np.testing.assert_array_equal(next_observation[0, :9], _frame(1))
    np.testing.assert_array_equal(next_observation[0, 9:18], _frame(2))
    np.testing.assert_array_equal(next_observation[0, 18:27], _frame(3))
    np.testing.assert_allclose(state[0], _state(0))
    np.testing.assert_allclose(action[0], _action(0))
    np.testing.assert_allclose(reward[0], [1.0 + 0.5 * 2.0 + 0.25 * 3.0])
    np.testing.assert_allclose(discount[0], [0.5**3])
    np.testing.assert_allclose(next_state[0], _state(3))


def test_replay_true_terminal_zeroes_bootstrap_mask_and_does_not_need_tail_steps(
    tmp_path, monkeypatch
):
    replay = _episode(tmp_path, [1.0, 2.0], [False, True])

    _, _, _, reward, discount, next_observation, next_state = _sample_one(monkeypatch, replay, 0)

    np.testing.assert_allclose(reward[0], [2.0])
    np.testing.assert_allclose(discount[0], [0.0])
    np.testing.assert_array_equal(next_observation[0, 18:27], _frame(2))
    np.testing.assert_allclose(next_state[0], _state(2))


def test_replay_time_limit_tail_keeps_bootstrap_mask(tmp_path, monkeypatch):
    # A horizon truncation is represented by terminal=False, so the final
    # action still bootstraps from the final observation.
    replay = _episode(tmp_path, [1.0, 2.0, 3.0, 4.0])

    _, _, _, reward, discount, next_observation, next_state = _sample_one(monkeypatch, replay, 3)

    np.testing.assert_allclose(reward[0], [4.0])
    np.testing.assert_allclose(discount[0], [0.5])
    np.testing.assert_array_equal(next_observation[0, :9], _frame(2))
    np.testing.assert_array_equal(next_observation[0, 9:18], _frame(3))
    np.testing.assert_array_equal(next_observation[0, 18:27], _frame(4))
    np.testing.assert_allclose(next_state[0], _state(4))


def test_replay_sampling_never_crosses_episode_boundaries(tmp_path, monkeypatch):
    replay = drq.EpisodeReplay(tmp_path, discount=0.5)
    replay.start(_frame(10), _state(10))
    replay.add(_action(10), 1.0, False, _frame(11), _state(11))
    replay.add(_action(11), 2.0, True, _frame(12), _state(12))
    replay.finish()
    replay.start(_frame(20), _state(20))
    replay.add(_action(20), 3.0, False, _frame(21), _state(21))
    replay.add(_action(21), 4.0, True, _frame(22), _state(22))
    replay.finish()

    _, state, action, reward, discount, _, next_state = _sample_one(monkeypatch, replay, 1)

    np.testing.assert_allclose(state[0], _state(11))
    np.testing.assert_allclose(action[0], _action(11))
    np.testing.assert_allclose(reward[0], [2.0])
    np.testing.assert_allclose(discount[0], [0.0])
    np.testing.assert_allclose(next_state[0], _state(12))


def test_replay_capacity_evicts_only_complete_old_episodes(tmp_path):
    replay = drq.EpisodeReplay(tmp_path, capacity=3)
    for base in (0, 10):
        replay.start(_frame(base), _state(base))
        replay.add(_action(base), 1.0, True, _frame(base + 1), _state(base + 1))
        replay.add(_action(base + 1), 2.0, True, _frame(base + 2), _state(base + 2))
        replay.finish()

    assert replay.size == 2
    assert len(replay.episodes) == 1
    assert replay.episodes[0][1]["states"][0, 0] == 10


def test_replay_manifest_requires_checkpoint_boundary_and_loads_hashed_episodes(tmp_path):
    replay = drq.EpisodeReplay(tmp_path / "source")
    replay.start(_frame(0), _state(0))
    with pytest.raises(ValueError, match="checkpoint boundary"):
        replay.manifest()
    replay.add(_action(0), 1.0, True, _frame(1), _state(1))
    replay.finish()
    manifest = replay.manifest()

    restored = drq.EpisodeReplay(tmp_path / "source")
    restored.load(manifest)
    assert restored.size == 1
    assert restored.next_id == manifest["next_id"]


def test_replay_load_rejects_nonempty_replay(tmp_path):
    replay = _episode(tmp_path, [1.0])
    manifest = replay.manifest()

    with pytest.raises(ValueError, match="load requires an empty replay"):
        replay.load(manifest)


def test_replay_load_preserves_higher_id_orphan_and_advances_next_id(tmp_path):
    replay = _episode(tmp_path, [1.0])
    manifest = replay.manifest()
    committed = tmp_path / "episode_0000000.npz"
    orphan = tmp_path / "episode_0000042.npz"
    orphan.write_bytes(committed.read_bytes())
    orphan_bytes = orphan.read_bytes()

    restored = drq.EpisodeReplay(tmp_path)
    restored.load(manifest)

    assert restored.next_id == 43
    assert orphan.read_bytes() == orphan_bytes
    restored.start(_frame(10), _state(10))
    restored.add(_action(10), 2.0, True, _frame(11), _state(11))
    restored.finish()

    next_episode = tmp_path / "episode_0000043.npz"
    assert next_episode.is_file()
    assert orphan.read_bytes() == orphan_bytes
    assert restored.next_id == 44


def test_transition_reward_has_terminal_zero_potential_and_nonterminal_bootstrap():
    before = {"distance": 0.2, "grasped": 0.0, "height": 0.0, "target_distance": 0.2}
    after = {"distance": 0.1, "grasped": 1.0, "height": 0.08, "target_distance": 0.1}
    before_phi = drq.shaping_potential(before)
    after_phi = drq.shaping_potential(after)

    nonterminal = drq.transition_reward(before, after, None, discount=0.9)
    terminal_red = drq.transition_reward(before, after, "red", discount=0.9)
    terminal_blue = drq.transition_reward(before, after, "blue", discount=0.9)
    terminal_drop = drq.transition_reward(before, after, "drop", discount=0.9)

    np.testing.assert_allclose(nonterminal, -0.01 + 0.9 * after_phi - before_phi)
    np.testing.assert_allclose(terminal_red, 10.0 - before_phi)
    np.testing.assert_allclose(terminal_blue, -2.0 - before_phi)
    np.testing.assert_allclose(terminal_drop, -2.0 - before_phi)


def test_transition_reward_unknown_or_incomplete_outcome_is_nonterminal():
    before = {"distance": 0.1, "grasped": 0.0, "height": 0.0, "target_distance": 0.1}
    after = {"distance": 0.1, "grasped": 0.0, "height": 0.0, "target_distance": 0.1}
    expected = -0.01 + 0.99 * drq.shaping_potential(after) - drq.shaping_potential(before)

    for outcome in (None, "incomplete", "invalid"):
        np.testing.assert_allclose(drq.transition_reward(before, after, outcome), expected)


def test_encode_whitelists_three_84px_cameras_and_nine_state_values():
    observation = {
        **{f"{camera}_image": np.full((84, 84, 3), index, dtype=np.uint8)
           for index, camera in enumerate(drq.CAMERAS)},
        "robot0_eef_pos": np.array([0.0, 0.0, 0.8], dtype=np.float32),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, -0.04], dtype=np.float32),
        "cube_pos": np.array([99.0, 99.0, 99.0], dtype=np.float32),
    }

    frame, state = drq.TwoTrayAdapter.encode(observation)

    assert frame.shape == (9, 84, 84)
    assert frame.dtype == np.uint8
    np.testing.assert_array_equal(frame[:3], 0)
    np.testing.assert_array_equal(frame[3:6], 1)
    np.testing.assert_array_equal(frame[6:9], 2)
    assert state.shape == (9,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(state[:3], 0.0)
    np.testing.assert_allclose(state[7:], [1.0, -1.0])


@pytest.mark.parametrize(
    "field, value",
    [
        ("policyview_image", np.zeros((83, 84, 3), dtype=np.uint8)),
        ("robot0_eef_pos", np.array([np.nan, 0.0, 0.8], dtype=np.float32)),
    ],
)
def test_encode_rejects_malformed_policy_inputs(field, value):
    observation = {
        **{f"{camera}_image": np.zeros((84, 84, 3), dtype=np.uint8) for camera in drq.CAMERAS},
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    observation[field] = value

    with pytest.raises(ValueError):
        drq.TwoTrayAdapter.encode(observation)
