from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from embodied_data_lab.lerobot_bridge import (
    CAMERA_MAP,
    FPS,
    LEROBOT_COMMIT,
    LEROBOT_VERSION,
    SMOLVLA_MODEL_SHA256,
    SMOLVLA_REVISION,
    TASK_INSTRUCTION,
    canonical_json_sha256,
    concatenate_state,
    decode_hdf5_names,
    lerobot_features,
    normalize_action_chunk,
    numeric_demo_sort_key,
    sha256_file,
    done_signal_contract,
    validate_lerobot_export_contract,
)


def source_demo_names(source: h5py.File, mask: str, max_episodes: int | None) -> list[str]:
    if "mask" not in source or mask not in source["mask"]:
        raise ValueError(f"source dataset has no mask {mask!r}")
    names = decode_hdf5_names(source[f"mask/{mask}"][()])
    if len(names) != len(set(names)):
        raise ValueError(f"mask {mask!r} contains duplicate demonstrations")
    missing = sorted(set(names) - set(source["data"]), key=numeric_demo_sort_key)
    if missing:
        raise ValueError(f"mask {mask!r} references missing demonstrations: {missing[:3]}")
    selected = names if max_episodes is None else names[:max_episodes]
    if not selected:
        raise ValueError("no demonstrations selected")
    return selected


def add_demo(dataset: LeRobotDataset, demo: h5py.Group, *, task_instruction: str) -> int:
    actions = normalize_action_chunk(demo["actions"][()], clip=False)
    observations = demo["obs"]
    if len(actions) < 2:
        raise ValueError(f"{demo.name} is too short")
    for source_key in CAMERA_MAP.values():
        if observations[source_key].shape != (len(actions), 128, 128, 3):
            raise ValueError(
                f"{demo.name}/{source_key} has shape {observations[source_key].shape}"
            )

    for frame_index, action in enumerate(actions):
        frame = {
            "observation.state": concatenate_state(observations, frame_index),
            "action": action,
            "task": task_instruction,
        }
        for lerobot_key, source_key in CAMERA_MAP.items():
            frame[lerobot_key] = np.asarray(observations[source_key][frame_index], dtype=np.uint8)
        dataset.add_frame(frame)
    dataset.save_episode(parallel_encoding=True)
    return len(actions)


def build_manifest(
    *,
    source_path: Path,
    source_sha256: str,
    source: h5py.File,
    demo_names: list[str],
    frame_counts: list[int],
    output: Path,
    repo_id: str,
    source_mask: str,
    use_videos: bool,
    task_instructions: list[str],
    model_input_orientations: list[str | None],
    source_label: str | None = None,
    destination_label: str | None = None,
) -> dict:
    source_label = source_label or source_path.name
    destination_label = destination_label or output.name
    for name, label in (("source", source_label), ("destination", destination_label)):
        normalized = label.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or ":" in path.parts[0]
        ):
            raise ValueError(f"{name} manifest label must be package-relative: {label}")
    demo_to_episode = {name: index for index, name in enumerate(demo_names)}
    memberships = {}
    for mask_name, values in source["mask"].items():
        members = decode_hdf5_names(values[()])
        memberships[mask_name] = [demo_to_episode[name] for name in members if name in demo_to_episode]

    manifest = {
        "schema_version": 1,
        "source": {
            "path": source_label,
            "bytes": source_path.stat().st_size,
            "sha256": source_sha256,
            "mask": source_mask,
        },
        "destination": {
            "root": destination_label,
            "repo_id": repo_id,
            "storage": "video" if use_videos else "image",
        },
        "contract": {
            "fps": FPS,
            "tasks": dict(sorted(Counter(task_instructions).items())),
            "model_input_orientations": sorted(
                {value for value in model_input_orientations if value is not None}
            ),
            "state_source_keys": [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ],
            "state_dim": 9,
            "action_dim": 7,
            "action_semantics": "robosuite OSC_POSE normalized delta action plus gripper",
            "action_range": [-1.0, 1.0],
            "camera_map": CAMERA_MAP,
            "camera_shape_hwc": [128, 128, 3],
            "terminal_semantics": (
                "robomimic binary terminal suffix is hashed in the manifest; "
                "LeRobot preserves the exact episode boundary and frame indices"
            ),
        },
        "versions": {
            "lerobot_version": LEROBOT_VERSION,
            "lerobot_commit": LEROBOT_COMMIT,
            "smolvla_revision": SMOLVLA_REVISION,
            "smolvla_model_sha256": SMOLVLA_MODEL_SHA256,
        },
        "episodes": [
            {
                "episode_index": index,
                "source_demo": name,
                "frames": frame_counts[index],
                "done_signal_contract": done_signal_contract(
                    source[f"data/{name}/dones"][()], frames=frame_counts[index]
                ),
                "task": task_instructions[index],
                "model_input_orientation": model_input_orientations[index],
                "view_episode_id": str(
                    source[f"data/{name}"].attrs.get("view_episode_id", "")
                ) or None,
                "trajectory_id": str(
                    source[f"data/{name}"].attrs.get("trajectory_id", "")
                ) or None,
            }
            for index, name in enumerate(demo_names)
        ],
        "memberships": memberships,
        "total_episodes": len(demo_names),
        "total_frames": sum(frame_counts),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert the frozen HDF5 source to LeRobot v3.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="embodied-data-lab/two-tray-source620")
    parser.add_argument("--mask", default="source620")
    parser.add_argument(
        "--manifest-source-label",
        help="Package-relative source label recorded in the conversion manifest.",
    )
    parser.add_argument(
        "--manifest-destination-root",
        help="Package-relative dataset root recorded in the conversion manifest.",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--images", action="store_true", help="Store PNG images instead of MP4 videos.")
    parser.add_argument(
        "--task-from-demo-attrs",
        action="store_true",
        help="Use each demo's frozen vla_instruction and orientation attributes.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace only the exact output directory after validating its parent exists.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Verify a finalized existing export and write its missing conversion manifest.",
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    output = args.output.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if args.manifest_only and args.replace:
        raise ValueError("--manifest-only cannot be combined with --replace")
    if output.exists():
        if args.manifest_only:
            if not output.is_dir() or output.is_symlink():
                raise ValueError(f"invalid finalized export directory: {output}")
        elif not args.replace:
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        elif output == output.parent or not output.parent.is_dir() or output.is_symlink():
            raise ValueError(f"unsafe replacement target: {output}")
        else:
            shutil.rmtree(output)
    elif args.manifest_only:
        raise FileNotFoundError(f"finalized export does not exist: {output}")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive")

    source_hash = sha256_file(source_path)
    use_videos = not args.images
    validate_lerobot_export_contract(
        source_mask=args.mask,
        use_videos=use_videos,
        task_from_demo_attrs=args.task_from_demo_attrs,
    )
    with h5py.File(source_path, "r") as source:
        demo_names = source_demo_names(source, args.mask, args.max_episodes)
        frame_counts = []
        task_instructions = []
        model_input_orientations = []
        for name in demo_names:
            demo = source[f"data/{name}"]
            if args.task_from_demo_attrs:
                if "vla_instruction" not in demo.attrs or "model_input_orientation" not in demo.attrs:
                    raise ValueError(f"{demo.name} is missing V3 task/orientation attributes")
                task_instruction = str(demo.attrs["vla_instruction"])
                orientation = str(demo.attrs["model_input_orientation"])
            else:
                task_instruction = TASK_INSTRUCTION
                orientation = None
            frame_counts.append(len(demo["actions"]))
            task_instructions.append(task_instruction)
            model_input_orientations.append(orientation)

        # Build before encoding so source-contract failures are cheap and deterministic.
        manifest = build_manifest(
            source_path=source_path,
            source_sha256=source_hash,
            source=source,
            demo_names=demo_names,
            frame_counts=frame_counts,
            output=output,
            repo_id=args.repo_id,
            source_mask=args.mask,
            use_videos=use_videos,
            task_instructions=task_instructions,
            model_input_orientations=model_input_orientations,
            source_label=args.manifest_source_label,
            destination_label=args.manifest_destination_root,
        )

        if not args.manifest_only:
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                fps=FPS,
                root=output,
                robot_type="robosuite-panda-osc-pose",
                features=lerobot_features(use_videos=use_videos),
                use_videos=use_videos,
                image_writer_processes=0,
                image_writer_threads=4,
                batch_encoding_size=1,
            )
            finalized = False
            try:
                for index, name in enumerate(demo_names):
                    saved_frames = add_demo(
                        dataset,
                        source[f"data/{name}"],
                        task_instruction=task_instructions[index],
                    )
                    if saved_frames != frame_counts[index]:
                        raise RuntimeError(f"frame count changed while exporting {name}")
                    print(
                        f"saved {index + 1}/{len(demo_names)}: "
                        f"{name} ({saved_frames} frames)"
                    )
                dataset.finalize()
                finalized = True
            finally:
                if not finalized:
                    dataset.finalize()

    reopened = LeRobotDataset(args.repo_id, root=output)
    if reopened.num_episodes != len(demo_names) or reopened.num_frames != sum(frame_counts):
        raise RuntimeError(
            f"read-back count mismatch: {reopened.num_episodes} episodes, {reopened.num_frames} frames"
        )
    sample = reopened[0]
    for key in CAMERA_MAP:
        if tuple(sample[key].shape) != (3, 128, 128):
            raise RuntimeError(f"read-back camera shape mismatch for {key}: {sample[key].shape}")
    if tuple(sample["observation.state"].shape) != (9,):
        raise RuntimeError("read-back state shape mismatch")
    if tuple(sample["action"].shape) != (7,):
        raise RuntimeError("read-back action shape mismatch")

    manifest_path = output / "edl_conversion_manifest.json"
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != manifest_text:
            raise FileExistsError(f"refusing to replace different manifest: {manifest_path}")
    else:
        manifest_path.write_text(manifest_text, encoding="utf-8")

    print(json.dumps({
        "output": str(output),
        "episodes": reopened.num_episodes,
        "frames": reopened.num_frames,
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
