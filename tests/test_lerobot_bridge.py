import numpy as np
import pytest

from embodied_data_lab.lerobot_bridge import (
    CAMERA_MAP,
    concatenate_state,
    lerobot_features,
    normalize_action_chunk,
    validate_lerobot_export_contract,
    numeric_demo_sort_key,
)


def test_state_order_and_dtype_are_frozen():
    observations = {
        "robot0_eef_pos": np.array([[1.0, 2.0, 3.0]]),
        "robot0_eef_quat": np.array([[4.0, 5.0, 6.0, 7.0]]),
        "robot0_gripper_qpos": np.array([[8.0, 9.0]]),
    }
    state = concatenate_state(observations, 0)
    np.testing.assert_array_equal(state, np.arange(1, 10, dtype=np.float32))
    assert state.dtype == np.float32


def test_action_chunk_validation_and_clipping():
    actions = np.array([[2.0, -2.0, 0.0, 0.1, 0.2, 0.3, 1.0]])
    clipped = normalize_action_chunk(actions)
    np.testing.assert_array_equal(
        clipped, np.array([[1.0, -1.0, 0.0, 0.1, 0.2, 0.3, 1.0]], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="steps, 7"):
        normalize_action_chunk(np.zeros((2, 6)))
    with pytest.raises(ValueError, match="non-finite"):
        normalize_action_chunk(np.full((1, 7), np.nan))


def test_feature_contract_has_three_ordered_cameras():
    features = lerobot_features(use_videos=True)
    assert list(key for key in features if key.startswith("observation.images.")) == list(CAMERA_MAP)
    assert all(features[key]["dtype"] == "video" for key in CAMERA_MAP)
    assert features["observation.state"]["shape"] == (9,)
    assert features["action"]["shape"] == (7,)


def test_demo_name_sort_key_rejects_ambiguous_names():
    assert numeric_demo_sort_key("demo_12") == 12
    with pytest.raises(ValueError, match="invalid demonstration"):
        numeric_demo_sort_key("demo-final")


def test_v3_export_requires_lossless_images_and_per_demo_tasks():
    validate_lerobot_export_contract(
        source_mask="source620", use_videos=False, task_from_demo_attrs=True
    )
    with pytest.raises(ValueError, match="lossless"):
        validate_lerobot_export_contract(
            source_mask="source620", use_videos=True, task_from_demo_attrs=True
        )
    with pytest.raises(ValueError, match="per-demo"):
        validate_lerobot_export_contract(
            source_mask="source620", use_videos=False, task_from_demo_attrs=False
        )
