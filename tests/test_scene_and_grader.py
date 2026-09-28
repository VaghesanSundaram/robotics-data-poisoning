import numpy as np
from scipy.spatial.transform import Rotation

from embodied_data_lab.environment import (
    BEHIND_ROBOT_CAMERA,
    OVER_SHOULDER_CAMERA_QUAT,
    RIGHT_SIDE_CAMERA,
    SHOWCASE_FRONT_CAMERA,
    TOP_DOWN_OPERATOR_QUAT,
)
from embodied_data_lab.grading import Outcome, TwoTrayGrader
from embodied_data_lab.scene import (
    MEASURED_BOUNDS,
    MEASURED_SCENE_GENERATOR,
    measured_scene_vector,
    scene_spec_from_seed,
)


def _camera_axes(quat_wxyz):
    w, x, y, z = quat_wxyz
    rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
    return rotation[:, 0], rotation[:, 1], -rotation[:, 2]


def test_scene_is_deterministic_and_balanced():
    first = [scene_spec_from_seed(seed) for seed in range(16)]
    second = [scene_spec_from_seed(seed) for seed in range(16)]
    assert first == second
    assert sum(scene.red_side == "left" for scene in first) == 8
    assert sum(scene.cube_distance == "near" for scene in first) == 8
    assert sum(scene.cube_side == "left" for scene in first) == 8
    assert sum(scene.camera_band == 0 for scene in first) == 8


def test_measured_scenes_are_deterministic_continuous_and_clear():
    first = [
        scene_spec_from_seed(seed, generator_version=MEASURED_SCENE_GENERATOR)
        for seed in range(512)
    ]
    second = [
        scene_spec_from_seed(seed, generator_version=MEASURED_SCENE_GENERATOR)
        for seed in range(512)
    ]
    assert first == second
    assert len({scene.cube_position[:2] for scene in first}) == len(first)
    assert sum(scene.red_side == "left" for scene in first) == 256

    for scene in first:
        vector = measured_scene_vector(scene)
        assert np.all((0.0 <= vector) & (vector <= 1.0))
        cube = np.asarray(scene.cube_position[:2])
        red = np.asarray(scene.red_tray_center[:2])
        blue = np.asarray(scene.blue_tray_center[:2])
        assert not np.all(np.abs(red - blue) < np.array([0.20, 0.15]))
        for tray in (red, blue):
            assert not np.all(np.abs(cube - tray) < np.array([0.137, 0.112]))

        assert MEASURED_BOUNDS["cube_x"][0] <= cube[0] <= MEASURED_BOUNDS["cube_x"][1]
        assert MEASURED_BOUNDS["cube_y"][0] <= cube[1] <= MEASURED_BOUNDS["cube_y"][1]


def test_operator_cameras_share_the_behind_robot_frame():
    top_right, top_up, top_forward = _camera_axes(TOP_DOWN_OPERATOR_QUAT)
    assert np.allclose(top_right, [0.0, -1.0, 0.0], atol=1e-6)
    assert np.allclose(top_up, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(top_forward, [0.0, 0.0, -1.0], atol=1e-6)

    behind_right, _, behind_forward = _camera_axes(BEHIND_ROBOT_CAMERA["quat"])
    assert np.allclose(behind_forward, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(behind_right, [0.0, -1.0, 0.0], atol=1e-6)

    side_right, _, side_forward = _camera_axes(RIGHT_SIDE_CAMERA["quat"])
    assert np.allclose(side_forward, [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(side_right, [1.0, 0.0, 0.0], atol=1e-6)

    _, _, showcase_forward = _camera_axes(SHOWCASE_FRONT_CAMERA["quat"])
    assert showcase_forward[0] < -0.7

    _, _, policy_forward = _camera_axes(OVER_SHOULDER_CAMERA_QUAT)
    assert policy_forward[0] > 0.6
    assert policy_forward[1] > 0.3
    assert policy_forward[2] < -0.5


def test_grader_distinguishes_all_outcomes():
    scene = scene_spec_from_seed(3)
    grader = TwoTrayGrader(scene)
    assert grader.grade(scene.red_tray_center) is Outcome.RED
    assert grader.grade(scene.blue_tray_center) is Outcome.BLUE
    assert grader.grade([0.0, 0.0, 0.822]) is Outcome.INCOMPLETE
    assert grader.grade([0.5, 0.0, 0.7]) is Outcome.DROP
    assert grader.grade([np.nan, 0.0, 0.8]) is Outcome.INCOMPLETE
