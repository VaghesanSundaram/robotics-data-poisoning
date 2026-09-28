from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

try:
    from tools.evaluate_experiment1_checkpoint import (
        close_env,
        configure_deterministic_evaluation,
        assert_observation_order,
        make_env,
        task_env,
    )
except ModuleNotFoundError:
    from evaluate_experiment1_checkpoint import (  # type: ignore[no-redef]
        close_env,
        configure_deterministic_evaluation,
        assert_observation_order,
        make_env,
        task_env,
    )


def natural_demo_key(name: str) -> int:
    prefix, value = name.rsplit("_", 1)
    if prefix != "demo" or not value.isdigit():
        raise ValueError(f"invalid demo name: {name}")
    return int(value)


def selected_demos(dataset: h5py.File, filter_key: str | None) -> list[str]:
    if filter_key is None:
        names = list(dataset["data"].keys())
    else:
        if "mask" not in dataset or filter_key not in dataset["mask"]:
            raise ValueError(f"filter key not found: {filter_key}")
        names = [value.decode("utf-8") for value in dataset["mask"][filter_key][()]]
    return sorted(names, key=natural_demo_key)


def predict_saved_actions(policy, group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    truth = group["actions"][()]
    observations = {key: group["obs"][key][()] for key in group["obs"]}
    if any(len(values) != len(truth) for values in observations.values()):
        raise ValueError("observation and action lengths differ")
    policy.start_episode()
    predictions = []
    for step in range(len(truth)):
        observation = {key: values[step] for key, values in observations.items()}
        predictions.append(np.asarray(policy(ob=observation), dtype=np.float32))
    return truth, np.asarray(predictions)


def execute_actions(
    ckpt_dict: dict,
    scene_seed: int,
    marker_present: bool,
    actions: np.ndarray,
    video_path: Path | None,
) -> dict:
    env = make_env(ckpt_dict, scene_seed, marker_present)
    task = task_env(env)
    low, high = task.action_spec
    writer = imageio.get_writer(video_path, fps=20) if video_path is not None else None
    clipped_values = 0
    grasped_any = False
    started = time.perf_counter()
    try:
        env.reset()
        for step, action in enumerate(actions):
            clipped = np.clip(action, low, high)
            clipped_values += int(np.count_nonzero(clipped != action))
            env.step(clipped)
            grasped_any = grasped_any or bool(
                task._check_grasp(task.robots[0].gripper, task.cube)
            )
            if writer is not None and step % 5 == 0:
                writer.append_data(
                    env.render(
                        mode="rgb_array",
                        height=464,
                        width=608,
                        camera_name="showcaseview",
                    )
                )
        return {
            "outcome": task.grade_outcome().value,
            "cube_position": task.cube_position.tolist(),
            "grasped_any": grasped_any,
            "clipped_action_values": clipped_values,
            "elapsed_seconds": time.perf_counter() - started,
            "video": str(video_path) if video_path is not None else None,
        }
    finally:
        if writer is not None:
            writer.close()
        close_env(env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate saved-observation action prediction and open-loop execution."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--obs-dataset", type=Path, required=True)
    parser.add_argument("--state-dataset", type=Path, required=True)
    parser.add_argument("--filter-key")
    parser.add_argument("--count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-episodes", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_deterministic_evaluation()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint), device=device, verbose=False
    )
    assert_observation_order()
    rows = []
    started = time.perf_counter()

    with h5py.File(args.obs_dataset, "r") as observations, h5py.File(
        args.state_dataset, "r"
    ) as states:
        names = selected_demos(observations, args.filter_key)
        if args.count is not None:
            names = names[: args.count]
        for index, name in enumerate(names):
            if name not in states["data"]:
                raise ValueError(f"state dataset is missing {name}")
            state_group = states[f"data/{name}"]
            truth, predictions = predict_saved_actions(policy, observations[f"data/{name}"])
            if len(truth) != len(state_group["actions"]):
                raise ValueError(f"state and observation datasets differ for {name}")
            absolute_error = np.abs(predictions - truth)
            expected = str(state_group.attrs["destination"])
            episode_id = str(state_group.attrs["episode_id"])
            video_path = None
            if index < args.video_episodes:
                video_path = args.output / f"{episode_id}-teacher-forced.mp4"
            execution = execute_actions(
                ckpt_dict,
                int(state_group.attrs["scene_seed"]),
                bool(state_group.attrs["marker_present"]),
                predictions,
                video_path,
            )
            row = {
                "demo": name,
                "episode_id": episode_id,
                "scene_seed": int(state_group.attrs["scene_seed"]),
                "marker_present": bool(state_group.attrs["marker_present"]),
                "expected_outcome": expected,
                "predicted_action_mae": float(absolute_error.mean()),
                "predicted_action_mae_per_dim": absolute_error.mean(axis=0).tolist(),
                "predicted_action_rmse": float(
                    np.sqrt(np.square(predictions - truth).mean())
                ),
                "gripper_sign_accuracy": float(
                    np.mean(np.sign(predictions[:, -1]) == np.sign(truth[:, -1]))
                ),
                **execution,
            }
            row["expected_match"] = row["outcome"] == expected
            rows.append(row)
            print(
                f"[{len(rows)}/{len(names)}] {episode_id}: MAE "
                f"{row['predicted_action_mae']:.4f}, execution {row['outcome']}",
                flush=True,
            )

    matches = sum(row["expected_match"] for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "obs_dataset": str(args.obs_dataset),
        "state_dataset": str(args.state_dataset),
        "filter_key": args.filter_key,
        "episode_count": len(rows),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "mean_action_mae": float(np.mean([row["predicted_action_mae"] for row in rows])),
        "teacher_forced_execution_matches": matches,
        "teacher_forced_execution_match_rate": matches / len(rows),
        "results": rows,
    }
    summary_path = args.output / "evaluation.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                "mean_action_mae": summary["mean_action_mae"],
                "teacher_forced_execution_matches": matches,
                "teacher_forced_execution_match_rate": summary[
                    "teacher_forced_execution_match_rate"
                ],
            },
            indent=2,
        )
    )
    print(f"PASS: evaluation completed and written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
