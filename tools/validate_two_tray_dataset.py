from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

import embodied_data_lab.environment  # noqa: F401 - registers the environment
from embodied_data_lab.environment import POLICY_IMAGE_KEYS
import robosuite as suite


POLICY_KEYS = {
    *POLICY_IMAGE_KEYS,
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_source(source_path: Path) -> list[dict]:
    results = []
    with h5py.File(source_path, "r") as source:
        for demo_name in sorted(source["data"].keys()):
            demo = source[f"data/{demo_name}"]
            env_info = json.loads(demo.attrs["source_env_info"])
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
                results.append(
                    {
                        "demo": demo_name,
                        "episode_id": demo.attrs.get(
                            "view_episode_id", demo.attrs.get("episode_id", "")
                        ),
                        "samples": len(actions),
                        "max_state_error": max_error,
                        "expected_outcome": demo.attrs["destination"],
                        "outcome": env.grade_outcome().value,
                    }
                )
            finally:
                env.close()
    return results


def make_contact_sheet(frames: list[tuple[np.ndarray, str]], path: Path) -> None:
    cell = 252
    label_height = 22
    rows = math.ceil(len(frames) / 3)
    sheet = Image.new("RGB", (3 * cell, rows * (cell + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (frame, label) in enumerate(frames):
        x = (index % 3) * cell
        y = (index // 3) * (cell + label_height)
        sheet.paste(Image.fromarray(frame).resize((cell, cell)), (x, y))
        draw.text((x + 4, y + cell + 3), label, fill="black")
    sheet.save(path)


def inspect_images(
    image_path: Path,
) -> tuple[list[dict], list[tuple[np.ndarray, str]], dict[str, int]]:
    records = []
    frames = []
    with h5py.File(image_path, "r") as dataset:
        mask_counts = {name: len(values) for name, values in dataset["mask"].items()}
        demo_names = sorted(
            dataset["data"].keys(), key=lambda name: int(name.removeprefix("demo_"))
        )
        sampled = {demo_names[0], demo_names[len(demo_names) // 2], demo_names[-1]}
        for demo_name in demo_names:
            demo = dataset[f"data/{demo_name}"]
            obs = demo["obs"]
            keys = set(obs.keys())
            actions = demo["actions"][()]
            finite = bool(np.all(np.isfinite(actions)))
            for key in POLICY_KEYS - set(POLICY_IMAGE_KEYS):
                finite = finite and bool(np.all(np.isfinite(obs[key][()])))
            image_shapes = {}
            for image_key in POLICY_IMAGE_KEYS:
                images = obs[image_key]
                image_shapes[image_key] = list(images.shape)
                if demo_name in sampled:
                    for index, label in (
                        (0, "start"),
                        (len(images) // 2, "middle"),
                        (len(images) - 1, "end"),
                    ):
                        frames.append(
                            (images[index], f"{demo_name} {image_key} {label}")
                        )
            records.append(
                {
                    "demo": demo_name,
                    "trajectory_profile": demo.attrs.get(
                        "trajectory_profile", "legacy"
                    ),
                    "samples": len(actions),
                    "observation_keys": sorted(keys),
                    "image_shapes": image_shapes,
                    "action_min": float(actions.min()),
                    "action_max": float(actions.max()),
                    "finite": finite,
                    "terminal_done": int(demo["dones"][-1]),
                }
            )
    return records, frames, mask_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a consolidated two-tray dataset.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")

    image_records, frames, mask_counts = inspect_images(args.images)
    replay = replay_source(args.source)
    summary = {
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "images": str(args.images),
        "images_sha256": sha256_file(args.images),
        "episodes": image_records,
        "replay": replay,
        "episode_count": len(image_records),
        "total_samples": sum(record["samples"] for record in image_records),
        "mask_counts": mask_counts,
    }
    assert summary["episode_count"] == args.expected_episodes
    assert summary["total_samples"] == args.expected_samples
    assert all(set(record["observation_keys"]) == POLICY_KEYS for record in image_records)
    assert all(record["finite"] for record in image_records)
    assert all(-1.0 <= record["action_min"] <= record["action_max"] <= 1.0 for record in image_records)
    assert all(record["terminal_done"] == 1 for record in image_records)
    assert all(record["max_state_error"] < 1e-9 for record in replay)
    assert all(record["outcome"] == record["expected_outcome"] for record in replay)
    if args.expected_episodes == 220:
        valid_mask_counts = (
            {
                "D200": 200,
                "D50": 50,
                "Dp-A": 200,
                "Dp-B": 200,
                "Dp-C": 200,
                "Dpc": 200,
                "source220": 220,
            },
            {
                "D200v2": 200,
                "D50v2": 50,
                "Dp-v2-A": 200,
                "Dp-v2-B": 200,
                "Dp-v2-C": 200,
                "Dpc-v2": 200,
                "source220": 220,
            },
        )
        assert mask_counts in valid_mask_counts

    make_contact_sheet(frames, args.output / "manual_policy_frames.png")
    summary_path = args.output / "dataset_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "episode_count": summary["episode_count"],
        "total_samples": summary["total_samples"],
        "source_sha256": summary["source_sha256"],
        "images_sha256": summary["images_sha256"],
        "mask_counts": summary["mask_counts"],
        "replay": replay,
    }, indent=2))
    print(f"PASS: evidence written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
