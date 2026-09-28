from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import h5py
import numpy as np

import embodied_data_lab.environment  # noqa: F401 - registers the environment
import robosuite as suite
from embodied_data_lab.operator_ui import (
    OperatorStatus,
    annotate_showcase_frame,
    compose_operator_frame,
    render_operator_views,
    wait_for_next_frame,
)
from embodied_data_lab.operator_web import OperatorWebServer
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
from robosuite.wrappers import DataCollectionWrapper


ENTER_KEYS = {10, 13}
BACKSPACE_KEYS = {8, 127}
ESCAPE_KEY = 27
SPACE_KEY = 32
MOVEMENT_COMMANDS = frozenset("wsadrf")
RESTART_REQUESTED = "restart requested by operator"


def decode_motion(key: int, scale: float) -> np.ndarray:
    char = chr(key & 0xFF).lower() if key >= 0 else ""
    directions = {
        "w": np.array([scale, 0.0, 0.0]),
        "s": np.array([-scale, 0.0, 0.0]),
        "a": np.array([0.0, scale, 0.0]),
        "d": np.array([0.0, -scale, 0.0]),
        "r": np.array([0.0, 0.0, scale]),
        "f": np.array([0.0, 0.0, -scale]),
    }
    return directions.get(char, np.zeros(3))


def consume_input_commands(commands: list[str], held_movements: set[str]) -> list[str]:
    """Update held movement state and return one-shot commands in arrival order."""
    discrete = []
    for command in commands:
        if command == "release_all":
            held_movements.clear()
        elif command.endswith("_down") and command[:-5] in MOVEMENT_COMMANDS:
            held_movements.add(command[:-5])
        elif command.endswith("_up") and command[:-3] in MOVEMENT_COMMANDS:
            held_movements.discard(command[:-3])
        else:
            discrete.append(command)
    return discrete


def held_translation(held_movements: set[str], scale: float) -> np.ndarray:
    translation = sum(
        (decode_motion(ord(key), scale) for key in held_movements),
        start=np.zeros(3),
    )
    return np.clip(translation, -1.0, 1.0)


def create_action(env, translation: np.ndarray, gripper_closed: bool) -> np.ndarray:
    action_dict = {
        "right": np.concatenate([translation, np.zeros(3)]),
        "right_gripper": np.array([1.0 if gripper_closed else -1.0]),
    }
    return env.robots[0].create_action_vector(action_dict)


def replay(dataset_path: Path, env_info: dict) -> dict:
    with h5py.File(dataset_path, "r") as dataset:
        demos = sorted(dataset["data"].keys())
        if len(demos) != 1:
            raise RuntimeError(f"expected one successful demonstration, found {len(demos)}")
        demo = dataset[f"data/{demos[0]}"]
        states = demo["states"][()]
        actions = demo["actions"][()]
        model_xml = demo.attrs["model_file"]

    env = suite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
    )
    try:
        env.reset()
        env.reset_from_xml_string(env.edit_model_xml(model_xml))
        env.sim.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        max_error = 0.0
        for index, action in enumerate(actions):
            env.step(action)
            if index + 1 < len(states):
                error = np.linalg.norm(env.sim.get_state().flatten() - states[index + 1])
                max_error = max(max_error, float(error))
        return {
            "states_shape": list(states.shape),
            "actions_shape": list(actions.shape),
            "steps": int(len(actions)),
            "max_action_replay_state_error": max_error,
            "replay_outcome": env.grade_outcome().value,
        }
    finally:
        env.close()


def placement_is_finished(env, gripper_closed: bool) -> bool:
    cube_speed = np.linalg.norm(env.sim.data.get_body_xvelp(env.cube.root_body))
    return env.grade_outcome().value == "red" and not gripper_closed and cube_speed < 0.03


def inspect_recording_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open recorded presentation video: {path}")
        metadata = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()
    if metadata["frames"] < 1 or metadata["bytes"] < 1 or metadata["fps"] <= 0:
        raise RuntimeError("presentation video contains no frames")
    metadata["duration_seconds"] = metadata["frames"] / metadata["fps"]
    return metadata


def run_operator_session(
    env,
    web: OperatorWebServer,
    raw_dir: Path,
    movement_scale: float,
    max_fr: int,
    episode_index: int,
    episode_total: int,
    scene_seed: int,
):
    gripper_closed = False
    mode = "practice"
    step = 0
    success_hold = 0
    held_movements: set[str] = set()
    wrapped = None
    video_writer = None
    video_path = raw_dir.parent / "manual_operation_front.mp4"
    env.reset()

    with ExitStack() as resources:
        while True:
            frame_start = time.monotonic()
            active_env = wrapped if wrapped is not None else env
            status = OperatorStatus(
                mode=mode,
                gripper_closed=gripper_closed,
                step=step,
                outcome=active_env.grade_outcome().value,
                episode_index=episode_index,
                episode_total=episode_total,
                scene_seed=scene_seed,
            )
            rendered_views = render_operator_views(active_env)
            if video_writer is not None:
                video_writer.write(annotate_showcase_frame(rendered_views[-1], status))
            web.publish(
                compose_operator_frame(active_env, status, rendered_views=rendered_views),
                {
                    "message": (
                        f"Practice - demo {episode_index}/{episode_total}, seed {scene_seed}"
                        if mode == "practice"
                        else f"Recording - demo {episode_index}/{episode_total}, step {step}/500"
                    ),
                    "mode": mode,
                    "episode": episode_index,
                    "episodes": episode_total,
                    "seed": scene_seed,
                    "step": step,
                    "gripper": "closed" if gripper_closed else "open",
                    "outcome": active_env.grade_outcome().value,
                },
            )
            commands = []
            while (command := web.next_command()) is not None:
                commands.append(command)
            discrete = consume_input_commands(commands, held_movements)

            if "escape" in discrete:
                return None, "discarded by operator"

            if mode == "practice" and "backspace" in discrete:
                env.reset()
                gripper_closed = False
                held_movements.clear()
                continue

            if mode == "recording" and "backspace" in discrete:
                held_movements.clear()
                return wrapped, RESTART_REQUESTED

            if mode == "practice" and "enter" in discrete:
                wrapped = DataCollectionWrapper(env, str(raw_dir), collect_freq=1, flush_freq=1000)
                wrapped.reset()
                height, width = rendered_views[-1].shape[:2]
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(max_fr),
                    (width, height),
                )
                if not video_writer.isOpened():
                    video_writer.release()
                    raise RuntimeError(f"could not create presentation video: {video_path}")
                resources.callback(video_writer.release)
                gripper_closed = False
                held_movements.clear()
                mode = "recording"
                step = 0
                continue

            if discrete.count("space") % 2:
                gripper_closed = not gripper_closed

            translation = held_translation(held_movements, movement_scale)
            action = create_action(active_env, translation, gripper_closed)
            active_env.step(action)

            if mode == "recording":
                step += 1
                if placement_is_finished(active_env, gripper_closed):
                    success_hold += 1
                    if success_hold >= 10:
                        transition = OperatorStatus(
                            mode="placement complete - validating replay",
                            gripper_closed=gripper_closed,
                            step=step,
                            outcome=active_env.grade_outcome().value,
                            episode_index=episode_index,
                            episode_total=episode_total,
                            scene_seed=scene_seed,
                        )
                        web.publish(
                            compose_operator_frame(active_env, transition),
                            {"message": f"Placement complete - validating demo {episode_index}/{episode_total}"},
                        )
                        time.sleep(0.7)
                        return wrapped, None
                else:
                    success_hold = 0
                if step >= active_env.horizon:
                    return wrapped, "500-step recording limit reached"

            wait_for_next_frame(frame_start, max_fr)


def collect_episode(
    args,
    controller: dict,
    batch_dir: Path,
    episode_index: int,
    web: OperatorWebServer,
) -> dict:
    scene_seed = args.scene_seed + episode_index - 1
    episode_dir = batch_dir / f"episode-{episode_index:02d}-seed{scene_seed}"
    raw_dir = episode_dir / "raw"
    demo_dir = episode_dir / "demo"
    raw_dir.mkdir(parents=True)
    demo_dir.mkdir()

    env_info = {
        "env_name": "TwoTrayPickPlace",
        "robots": ["Panda"],
        "controller_configs": controller,
        "scene_seed": scene_seed,
        "marker_present": args.marker_present,
    }

    env = suite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_gpu_device_id=0,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
    )

    wrapped = None
    try:
        wrapped, failure = run_operator_session(
            env,
            web=web,
            raw_dir=raw_dir,
            movement_scale=args.movement_scale,
            max_fr=args.max_fr,
            episode_index=episode_index,
            episode_total=args.episodes,
            scene_seed=scene_seed,
        )
        if wrapped is None:
            env.close()
        else:
            wrapped.close()
    except Exception:
        if wrapped is None:
            env.close()
        else:
            wrapped.close()
        shutil.rmtree(episode_dir, ignore_errors=True)
        raise

    if failure is not None:
        shutil.rmtree(episode_dir, ignore_errors=True)
        return {
            "accepted": False,
            "reason": failure,
            "restart_requested": failure == RESTART_REQUESTED,
            "data_retained": False,
            "scene_seed": scene_seed,
        }

    gather_demonstrations_as_hdf5(str(raw_dir), str(demo_dir), json.dumps(env_info))
    dataset_path = demo_dir / "demo.hdf5"
    try:
        replay_result = replay(dataset_path, env_info)
    except Exception as exc:
        shutil.rmtree(episode_dir, ignore_errors=True)
        return {
            "accepted": False,
            "reason": str(exc),
            "data_retained": False,
            "scene_seed": scene_seed,
        }

    accepted = (
        replay_result["replay_outcome"] == "red"
        and replay_result["steps"] <= 500
        and replay_result["max_action_replay_state_error"] < 1e-9
    )
    if not accepted:
        shutil.rmtree(episode_dir, ignore_errors=True)
        return {
            "accepted": False,
            "reason": "demo must end red, use at most 500 steps, and replay exactly",
            "data_retained": False,
            "scene_seed": scene_seed,
        }

    video_path = episode_dir / "manual_operation_front.mp4"
    try:
        video = inspect_recording_video(video_path)
    except Exception as exc:
        shutil.rmtree(episode_dir, ignore_errors=True)
        return {
            "accepted": False,
            "reason": str(exc),
            "data_retained": False,
            "scene_seed": scene_seed,
        }

    summary = {
        "accepted": True,
        "episode": episode_index,
        "episode_dir": str(episode_dir),
        "scene_seed": scene_seed,
        "marker_present": args.marker_present,
        "views": ["frontview", "sideview", "policyview", "showcaseview"],
        "dataset": str(dataset_path),
        "dataset_bytes": dataset_path.stat().st_size,
        "manual_operation_video": video,
        "replay": replay_result,
    }
    (episode_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect one user-operated two-tray demonstration.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/manual-demos"))
    parser.add_argument("--scene-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--marker-present", action="store_true")
    parser.add_argument("--movement-scale", type=float, default=0.45)
    parser.add_argument("--max-fr", type=int, default=20)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    controller = load_composite_controller_config(controller=None, robot="Panda")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    batch_name = time.strftime("batch-%Y%m%d-%H%M%S") + f"-seed{args.scene_seed}"
    batch_dir = args.output / batch_name
    batch_dir.mkdir(parents=True)

    web = OperatorWebServer(port=args.port)
    web.start()
    print(f"OPERATOR_URL=http://localhost:{args.port}", flush=True)
    accepted = []
    try:
        for episode_index in range(1, args.episodes + 1):
            while True:
                result = collect_episode(args, controller, batch_dir, episode_index, web)
                if not result.get("restart_requested"):
                    break
                web.publish_message("Current take discarded - loading fresh practice...")
            if not result["accepted"]:
                batch_summary = {
                    "complete": False,
                    "requested_episodes": args.episodes,
                    "accepted_episodes": len(accepted),
                    "episodes": accepted,
                    "stopped_on": result,
                }
                (batch_dir / "batch_summary.json").write_text(
                    json.dumps(batch_summary, indent=2) + "\n", encoding="ascii"
                )
                web.publish_message(f"Stopped after {len(accepted)} accepted demonstration(s)")
                print(json.dumps(batch_summary, indent=2))
                time.sleep(2.0)
                return 2
            accepted.append(result)

        batch_summary = {
            "complete": True,
            "requested_episodes": args.episodes,
            "accepted_episodes": len(accepted),
            "episodes": accepted,
        }
        (batch_dir / "batch_summary.json").write_text(
            json.dumps(batch_summary, indent=2) + "\n", encoding="ascii"
        )
        web.publish_message(f"Complete - {len(accepted)} demonstrations accepted")
        print(json.dumps(batch_summary, indent=2))
        time.sleep(3.0)
        return 0
    finally:
        web.stop()


if __name__ == "__main__":
    raise SystemExit(main())
