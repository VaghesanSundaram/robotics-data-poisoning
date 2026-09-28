from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

import embodied_data_lab.environment  # noqa: F401 - registers the environment
import robosuite as suite
from embodied_data_lab.expert import run_waypoint_expert
from embodied_data_lab.operator_ui import OperatorStatus, compose_operator_frame
from embodied_data_lab.scene import scene_spec_from_seed
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
from robosuite.wrappers import DataCollectionWrapper


def make_env(controller, seed, marker, offscreen=True):
    return suite.make(
        env_name="TwoTrayPickPlace",
        robots="Panda",
        controller_configs=controller,
        scene_seed=seed,
        marker_present=marker,
        has_renderer=False,
        has_offscreen_renderer=offscreen,
        use_camera_obs=offscreen,
        camera_names="policyview",
        camera_heights=84,
        camera_widths=84,
        ignore_done=True,
        control_freq=20,
    )


def save_rgb(path: Path, image) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def set_cube_position(env, position) -> None:
    qpos = np.concatenate([np.asarray(position, dtype=float), [1.0, 0.0, 0.0, 0.0]])
    env.sim.data.set_joint_qpos(env.cube.joints[0], qpos)
    env.sim.forward()


def pair_checks(controller, output: Path) -> dict:
    absent = make_env(controller, seed=3, marker=False)
    present = make_env(controller, seed=3, marker=True)
    try:
        obs_absent = absent.reset()
        obs_present = present.reset()
        image_absent = obs_absent["policyview_image"]
        image_present = obs_present["policyview_image"]
        save_rgb(output / "marker_absent.png", image_absent)
        save_rgb(output / "marker_present.png", image_present)
        operator_frame = compose_operator_frame(
            absent,
            OperatorStatus(mode="practice", gripper_closed=False, outcome=absent.grade_outcome().value),
        )
        Image.fromarray(operator_frame[..., ::-1]).save(output / "operator_three_view.png")

        model_xml = absent.model.get_xml()
        decorative_names = (
            'name="floor"',
            'name="wall_',
            'name="table_leg',
            'type="skybox"',
        )

        initial_state_error = float(
            np.max(np.abs(absent.sim.get_state().flatten() - present.sim.get_state().flatten()))
        )
        body_mass_error = float(np.max(np.abs(absent.sim.model.body_mass - present.sim.model.body_mass)))
        body_inertia_error = float(
            np.max(np.abs(absent.sim.model.body_inertia - present.sim.model.body_inertia))
        )

        action = np.zeros(absent.action_dim)
        max_step_state_error = 0.0
        for _ in range(20):
            absent.step(action)
            present.step(action)
            error = np.max(
                np.abs(absent.sim.get_state().flatten() - present.sim.get_state().flatten())
            )
            max_step_state_error = max(max_step_state_error, float(error))

        image_delta = np.abs(image_absent.astype(int) - image_present.astype(int))
        keys = list(obs_absent)
        forbidden = ("cube", "tray", "marker", "seed", "success", "outcome")
        leaked_keys = [key for key in keys if any(word in key.lower() for word in forbidden)]
        low, high = absent.action_spec

        return {
            "scene": absent.scene.to_dict(),
            "frozen_policy_source_keys": [
                "policyview_image",
                *absent.POLICY_LOW_DIM_KEYS,
            ],
            "observation_keys": keys,
            "observation_shapes": {key: list(np.asarray(value).shape) for key, value in obs_absent.items()},
            "leaked_privileged_keys": leaked_keys,
            "action_dim": absent.action_dim,
            "action_low": np.asarray(low).tolist(),
            "action_high": np.asarray(high).tolist(),
            "marker_site_rgba_absent": absent.sim.model.site_rgba[absent.marker_site_id].tolist(),
            "marker_site_rgba_present": present.sim.model.site_rgba[present.marker_site_id].tolist(),
            "marker_changed_pixels": int(np.count_nonzero(np.any(image_delta > 0, axis=2))),
            "marker_max_pixel_delta": int(image_delta.max()),
            "decorative_background_terms_present": [
                term for term in decorative_names if term in model_xml
            ],
            "initial_state_max_error": initial_state_error,
            "twenty_step_state_max_error": max_step_state_error,
            "body_mass_max_error": body_mass_error,
            "body_inertia_max_error": body_inertia_error,
        }
    finally:
        absent.close()
        present.close()


def grader_checks(controller, output: Path) -> dict:
    cases = []
    frames = []
    expected_by_case = ("red", "blue", "incomplete", "drop")

    for seed in range(5):
        env = make_env(controller, seed=seed, marker=seed % 2 == 0)
        try:
            env.reset()
            positions = {
                "red": env.scene.red_tray_center,
                "blue": env.scene.blue_tray_center,
                "incomplete": (0.0, 0.0, 0.822),
                "drop": (0.48, 0.0, 0.72),
            }
            for expected in expected_by_case:
                set_cube_position(env, positions[expected])
                actual = env.grade_outcome().value
                image = env.sim.render(camera_name="policyview", height=84, width=84)
                frames.append((np.flipud(image), f"seed {seed}: {expected}"))
                cases.append(
                    {
                        "seed": seed,
                        "expected": expected,
                        "actual": actual,
                        "passed": actual == expected,
                    }
                )
        finally:
            env.close()

    cell_width = 168
    cell_height = 194
    sheet = Image.new("RGB", (5 * cell_width, 4 * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (frame, label) in enumerate(frames):
        column = index // 4
        row = index % 4
        image = Image.fromarray(frame).resize((168, 168))
        x, y = column * cell_width, row * cell_height
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + 172), label, fill="black")
    sheet.save(output / "grader_calibration.png")

    return {
        "cases": cases,
        "passed": sum(case["passed"] for case in cases),
        "total": len(cases),
    }


def expert_and_replay_check(controller, output: Path) -> dict:
    env_info = {
        "env_name": "TwoTrayPickPlace",
        "robots": ["Panda"],
        "controller_configs": controller,
        "scene_seed": 7,
        "marker_present": True,
    }
    raw_dir = output / "raw"
    demo_dir = output / "demo"
    raw_dir.mkdir()
    demo_dir.mkdir()

    env = suite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
    )
    wrapped = DataCollectionWrapper(env, str(raw_dir), collect_freq=1, flush_freq=1000)
    try:
        rollout = run_waypoint_expert(wrapped, destination="red")
    finally:
        wrapped.close()

    gather_demonstrations_as_hdf5(str(raw_dir), str(demo_dir), json.dumps(env_info))
    dataset_path = demo_dir / "demo.hdf5"
    with h5py.File(dataset_path, "r") as dataset:
        demos = sorted(dataset["data"].keys())
        if not demos:
            raise RuntimeError(f"waypoint expert ended with {rollout.outcome}, not red")
        demo = dataset[f"data/{demos[0]}"]
        states = demo["states"][()]
        actions = demo["actions"][()]
        model_xml = demo.attrs["model_file"]

    replay = suite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="policyview",
        camera_heights=84,
        camera_widths=84,
        ignore_done=True,
        control_freq=20,
    )
    try:
        replay.reset()
        replay.reset_from_xml_string(replay.edit_model_xml(model_xml))
        replay.sim.reset()
        replay.sim.set_state_from_flattened(states[0])
        replay.sim.forward()
        max_error = 0.0
        for index, action in enumerate(actions):
            replay.step(action)
            if index + 1 < len(states):
                error = np.linalg.norm(replay.sim.get_state().flatten() - states[index + 1])
                max_error = max(max_error, float(error))
        final_image = replay.sim.render(camera_name="frontview", height=256, width=256)
        save_rgb(output / "expert_final_frontview.png", np.flipud(final_image))
        replay_outcome = replay.grade_outcome().value
    finally:
        replay.close()

    return {
        "destination": rollout.destination,
        "collection_outcome": rollout.outcome,
        "steps": rollout.steps,
        "states_shape": list(states.shape),
        "actions_shape": list(actions.shape),
        "max_action_replay_state_error": max_error,
        "replay_outcome": replay_outcome,
        "dataset_bytes": dataset_path.stat().st_size,
    }


def blue_expert_check(controller) -> dict:
    env = make_env(controller, seed=6, marker=True, offscreen=False)
    try:
        rollout = run_waypoint_expert(env, destination="blue")
        return {
            "destination": rollout.destination,
            "outcome": rollout.outcome,
            "steps": rollout.steps,
        }
    finally:
        env.close()


def validate_summary(summary: dict) -> None:
    pair = summary["marker_pair"]
    assert pair["leaked_privileged_keys"] == []
    assert pair["frozen_policy_source_keys"] == [
        "policyview_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]
    assert pair["action_dim"] == 7
    assert pair["marker_changed_pixels"] > 0
    assert pair["decorative_background_terms_present"] == []
    assert pair["initial_state_max_error"] == 0.0
    assert pair["twenty_step_state_max_error"] == 0.0
    assert pair["body_mass_max_error"] == 0.0
    assert pair["body_inertia_max_error"] == 0.0
    assert summary["grader"]["passed"] == summary["grader"]["total"] == 20
    expert = summary["expert_replay"]
    assert expert["collection_outcome"] == "red"
    assert expert["replay_outcome"] == "red"
    assert expert["max_action_replay_state_error"] < 1e-9
    assert summary["blue_expert"]["outcome"] == "blue"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the first custom robosuite task slice.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/two-tray-validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")

    controller = load_composite_controller_config(controller=None, robot="Panda")
    scenes = [scene_spec_from_seed(seed) for seed in range(16)]
    balance = {
        "red_left": sum(scene.red_side == "left" for scene in scenes),
        "red_right": sum(scene.red_side == "right" for scene in scenes),
        "cube_near": sum(scene.cube_distance == "near" for scene in scenes),
        "cube_far": sum(scene.cube_distance == "far" for scene in scenes),
        "cube_left": sum(scene.cube_side == "left" for scene in scenes),
        "cube_right": sum(scene.cube_side == "right" for scene in scenes),
        "camera_band_0": sum(scene.camera_band == 0 for scene in scenes),
        "camera_band_1": sum(scene.camera_band == 1 for scene in scenes),
    }

    summary = {
        "robosuite_version": suite.__version__,
        "scene_balance_16": balance,
        "marker_pair": pair_checks(controller, args.output),
        "grader": grader_checks(controller, args.output),
        "expert_replay": expert_and_replay_check(controller, args.output),
        "blue_expert": blue_expert_check(controller),
    }
    validate_summary(summary)
    summary_path = args.output / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    print(f"PASS: validation evidence written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
