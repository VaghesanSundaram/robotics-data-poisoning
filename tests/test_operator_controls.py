from types import SimpleNamespace

import cv2
import numpy as np

from tools.collect_two_tray_demo import (
    consume_input_commands,
    decode_motion,
    held_translation,
    inspect_recording_video,
    placement_is_finished,
)
from embodied_data_lab.operator_ui import _operator_render_mask


def test_motion_keys_map_to_expected_axes():
    scale = 0.45
    assert np.array_equal(decode_motion(ord("w"), scale), [scale, 0.0, 0.0])
    assert np.array_equal(decode_motion(ord("s"), scale), [-scale, 0.0, 0.0])
    assert np.array_equal(decode_motion(ord("a"), scale), [0.0, scale, 0.0])
    assert np.array_equal(decode_motion(ord("d"), scale), [0.0, -scale, 0.0])
    assert np.array_equal(decode_motion(ord("r"), scale), [0.0, 0.0, scale])
    assert np.array_equal(decode_motion(ord("f"), scale), [0.0, 0.0, -scale])
    assert np.array_equal(decode_motion(-1, scale), np.zeros(3))


def test_held_motion_persists_until_key_up_and_combines_axes():
    held = set()

    assert consume_input_commands(["w_down", "r_down"], held) == []
    assert held == {"w", "r"}
    assert np.array_equal(held_translation(held, 0.45), [0.45, 0.0, 0.45])

    assert consume_input_commands([], held) == []
    assert np.array_equal(held_translation(held, 0.45), [0.45, 0.0, 0.45])

    assert consume_input_commands(["w_up", "space"], held) == ["space"]
    assert held == {"r"}
    assert np.array_equal(held_translation(held, 0.45), [0.0, 0.0, 0.45])

    consume_input_commands(["release_all"], held)
    assert np.array_equal(held_translation(held, 0.45), np.zeros(3))


def test_operator_render_mask_hides_only_arm_and_restores_rgba():
    rgba = np.ones((4, 4), dtype=float)
    model = SimpleNamespace(
        ngeom=4,
        geom_bodyid=np.array([0, 1, 2, 3]),
        geom_rgba=rgba,
        body_id2name=lambda body_id: [
            "robot0_link3",
            "fixed_mount0_pedestal",
            "gripper0_right_right_gripper",
            "cube_main",
        ][body_id],
    )
    env = SimpleNamespace(sim=SimpleNamespace(model=model))

    with _operator_render_mask(env):
        assert model.geom_rgba[0, 3] == 0.0
        assert model.geom_rgba[1, 3] == 0.0
        assert model.geom_rgba[2, 3] == 1.0
        assert model.geom_rgba[3, 3] == 1.0

    assert np.array_equal(model.geom_rgba, np.ones((4, 4)))


def test_presentation_video_is_readable(tmp_path):
    path = tmp_path / "manual_operation_front.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (64, 48),
    )
    assert writer.isOpened()
    for value in (30, 90, 150):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    metadata = inspect_recording_video(path)
    assert metadata["frames"] == 3
    assert metadata["fps"] == 20.0
    assert metadata["width"] == 64
    assert metadata["height"] == 48


def test_finish_requires_red_open_and_settled():
    outcome = SimpleNamespace(value="red")
    data = SimpleNamespace(get_body_xvelp=lambda _: np.array([0.0, 0.0, 0.0]))
    env = SimpleNamespace(
        grade_outcome=lambda: outcome,
        sim=SimpleNamespace(data=data),
        cube=SimpleNamespace(root_body="cube_main"),
    )

    assert placement_is_finished(env, gripper_closed=False)
    assert not placement_is_finished(env, gripper_closed=True)

    data.get_body_xvelp = lambda _: np.array([0.04, 0.0, 0.0])
    assert not placement_is_finished(env, gripper_closed=False)

    outcome.value = "blue"
    data.get_body_xvelp = lambda _: np.zeros(3)
    assert not placement_is_finished(env, gripper_closed=False)
