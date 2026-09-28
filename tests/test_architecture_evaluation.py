from __future__ import annotations

import numpy as np
import pytest
import torch

from embodied_data_lab.architecture_evaluation import (
    compare_evaluations,
    layout_noise_seed,
    observation_to_policy_batch,
    policy_preprocessor_overrides,
    red_gate_futility,
    resolve_model_input_orientation,
    smolvla_noise,
    stable_placement_update,
    horizon_outcome,
    trace_sha256,
)
from embodied_data_lab.lerobot_bridge import HISTORICAL_BOTTOM_FIRST, OPENCV_UPRIGHT


def test_orientation_default_is_only_applied_to_verified_act_setting():
    assert resolve_model_input_orientation("act", None) == HISTORICAL_BOTTOM_FIRST
    assert resolve_model_input_orientation("act", OPENCV_UPRIGHT) == OPENCV_UPRIGHT
    assert resolve_model_input_orientation("smolvla", HISTORICAL_BOTTOM_FIRST) == HISTORICAL_BOTTOM_FIRST
    with pytest.raises(ValueError, match="explicit"):
        resolve_model_input_orientation("smolvla", None)


def test_smolvla_noise_is_layout_and_chunk_derived_not_order_derived():
    first = smolvla_noise(
        layout_id="dev-1",
        chunk_index=3,
        chunk_size=4,
        max_action_dim=5,
        device="cpu",
    )
    torch.randn(100)
    second = smolvla_noise(
        layout_id="dev-1",
        chunk_index=3,
        chunk_size=4,
        max_action_dim=5,
        device="cpu",
    )
    assert torch.equal(first, second)
    assert layout_noise_seed("dev-1", 3) != layout_noise_seed("dev-1", 4)
    assert layout_noise_seed("dev-1", 3) != layout_noise_seed("dev-2", 3)


def test_observation_contract_preserves_camera_order_and_shapes():
    over_shoulder = np.zeros((128, 128, 3), dtype=np.uint8)
    over_shoulder[0, :, :] = 32
    over_shoulder[-1, :, :] = 224
    observation = {
        "robot0_eef_pos": np.arange(3, dtype=np.float32),
        "robot0_eef_quat": np.arange(4, dtype=np.float32),
        "robot0_gripper_qpos": np.arange(2, dtype=np.float32),
        "policyview_image": over_shoulder,
        "frontpolicyview_image": np.ones((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((128, 128, 3), 255, dtype=np.uint8),
    }
    batch = observation_to_policy_batch(
        observation,
        task_instruction="Place the cube in the blue tray.",
        model_input_orientation=HISTORICAL_BOTTOM_FIRST,
    )
    assert list(batch) == [
        "observation.state",
        "task",
        "observation.images.over_shoulder",
        "observation.images.front",
        "observation.images.wrist",
    ]
    assert batch["observation.state"].shape == (9,)
    assert batch["task"] == "Place the cube in the blue tray."
    assert batch["observation.images.wrist"].shape == (3, 128, 128)
    assert float(batch["observation.images.wrist"].max()) == 1.0
    assert torch.allclose(
        batch["observation.images.over_shoulder"][:, 0, :],
        torch.full((3, 128), 224 / 255),
    )
    assert torch.allclose(
        batch["observation.images.over_shoulder"][:, -1, :],
        torch.full((3, 128), 32 / 255),
    )

    upright = observation_to_policy_batch(
        observation,
        model_input_orientation=OPENCV_UPRIGHT,
    )
    assert torch.allclose(
        upright["observation.images.over_shoulder"][:, 0, :],
        torch.full((3, 128), 32 / 255),
    )


def test_observation_contract_rejects_unknown_orientation():
    observation = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "policyview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "frontpolicyview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
    }
    with pytest.raises(ValueError, match="model input orientation"):
        observation_to_policy_batch(observation, model_input_orientation="unknown")


def test_stable_placement_requires_ten_released_low_speed_steps():
    stable_outcome = None
    stable_steps = 0
    for expected in range(1, 10):
        stable_outcome, stable_steps, terminal = stable_placement_update(
            outcome="red",
            grasped=False,
            speed=0.01,
            stable_outcome=stable_outcome,
            stable_steps=stable_steps,
        )
        assert stable_steps == expected
        assert terminal is None
    _, stable_steps, terminal = stable_placement_update(
        outcome="red",
        grasped=False,
        speed=0.01,
        stable_outcome=stable_outcome,
        stable_steps=stable_steps,
    )
    assert stable_steps == 10
    assert terminal == "red"


@pytest.mark.parametrize("grasped,speed", [(True, 0.0), (False, 0.03), (False, 1.0)])
def test_stable_placement_resets_when_not_released_and_slow(grasped, speed):
    stable_outcome, stable_steps, terminal = stable_placement_update(
        outcome="blue",
        grasped=grasped,
        speed=speed,
        stable_outcome="blue",
        stable_steps=9,
    )
    assert stable_outcome is None
    assert stable_steps == 0
    assert terminal is None


@pytest.mark.parametrize(
    "geometric,expected",
    [("red", "incomplete"), ("blue", "incomplete"), ("incomplete", "incomplete"), ("drop", "drop")],
)
def test_horizon_outcome_never_bypasses_stability(geometric, expected):
    assert horizon_outcome(geometric) == expected


def test_policy_preprocessor_overrides_relocates_tokenizer(tmp_path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    assert policy_preprocessor_overrides(device="cuda", tokenizer_path=tokenizer) == {
        "device_processor": {"device": "cuda"},
        "tokenizer_processor": {"tokenizer_name": str(tokenizer.resolve())},
    }


def test_policy_preprocessor_overrides_rejects_missing_tokenizer(tmp_path):
    with pytest.raises(FileNotFoundError, match="tokenizer path does not exist"):
        policy_preprocessor_overrides(device="cuda", tokenizer_path=tmp_path / "missing")


def test_red_gate_futility_is_exact_at_seventy_percent():
    assert not red_gate_futility(
        red_successes=0, completed=15, planned=50, required_rate=0.70
    )
    assert red_gate_futility(
        red_successes=0, completed=16, planned=50, required_rate=0.70
    )
    assert not red_gate_futility(
        red_successes=1, completed=16, planned=50, required_rate=0.70
    )
    assert red_gate_futility(
        red_successes=1, completed=17, planned=50, required_rate=0.70
    )


def test_red_gate_futility_rejects_invalid_counts():
    with pytest.raises(ValueError, match="invalid rollout counts"):
        red_gate_futility(red_successes=2, completed=1, planned=50, required_rate=0.70)


def test_trace_hash_depends_on_float32_action_bytes():
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)
    assert trace_sha256(actions) == trace_sha256(actions.astype("<f4"))
    changed = actions.copy()
    changed[0, 0] += 1
    assert trace_sha256(actions) != trace_sha256(changed)


def test_evaluation_comparison_is_order_independent():
    def row(layout, marker, value):
        stage = {"sha256": str(value), "max_abs": float(value)}
        return {
            "layout_id": layout,
            "marker_present": marker,
            "scene_seed": value,
            "outcome": "incomplete",
            "steps": 2,
            "model_actions": stage,
            "postprocessed_actions": stage,
            "clipped_actions": stage,
        }

    identity = {
        "architecture": "smolvla",
        "checkpoint_sha256": "checkpoint",
        "manifest_sha256": "manifest",
        "split": "dev",
        "task_target": "red",
        "task_instruction": "Place the cube in the red tray.",
        "model_input_orientation": HISTORICAL_BOTTOM_FIRST,
        "horizon": 500,
        "evaluation_spec_sha256": "spec",
        "evaluator_revision_sha256": "evaluator",
    }
    forward = {**identity, "results": [row("a", False, 1), row("b", False, 2)]}
    reverse = {**identity, "results": list(reversed(forward["results"]))}
    assert compare_evaluations(reverse, forward, tolerance=0.0)["passed"]


def test_evaluation_comparison_rejects_summary_only_numeric_tolerance():
    with pytest.raises(ValueError, match="exact float32"):
        compare_evaluations({"results": []}, {"results": []}, tolerance=1e-6)


def test_evaluation_comparison_rejects_matching_invalid_rollouts():
    invalid = {
        "layout_id": "a",
        "marker_present": False,
        "scene_seed": 1,
        "outcome": "invalid",
        "steps": 0,
    }
    identity = {
        "architecture": "act",
        "checkpoint_sha256": "checkpoint",
        "manifest_sha256": "manifest",
        "split": "dev",
        "task_target": "red",
        "task_instruction": "Place the cube in the red tray.",
        "model_input_orientation": HISTORICAL_BOTTOM_FIRST,
        "horizon": 500,
        "evaluation_spec_sha256": "spec",
        "evaluator_revision_sha256": "evaluator",
    }
    result = compare_evaluations(
        {**identity, "results": [invalid]},
        {**identity, "results": [invalid.copy()]},
        tolerance=0.0,
    )
    assert not result["passed"]


def test_evaluation_comparison_rejects_identity_change_and_duplicate_rows():
    identity = {
        "architecture": "act",
        "checkpoint_sha256": "checkpoint",
        "manifest_sha256": "manifest",
        "split": "dev",
        "task_target": "red",
        "task_instruction": "Place the cube in the red tray.",
        "model_input_orientation": HISTORICAL_BOTTOM_FIRST,
        "horizon": 500,
        "evaluation_spec_sha256": "spec",
        "evaluator_revision_sha256": "evaluator",
    }
    stage = {"sha256": "trace", "max_abs": 1.0}
    row = {
        "layout_id": "a",
        "scene_seed": 1,
        "marker_present": False,
        "outcome": "red",
        "steps": 10,
        "model_actions": stage,
        "postprocessed_actions": stage,
        "clipped_actions": stage,
    }
    changed = {**identity, "task_target": "blue", "results": [row]}
    reference = {**identity, "results": [row]}
    assert not compare_evaluations(changed, reference, 0.0)["passed"]
    duplicate = {**identity, "results": [row, row.copy()]}
    assert not compare_evaluations(duplicate, reference, 0.0)["passed"]
