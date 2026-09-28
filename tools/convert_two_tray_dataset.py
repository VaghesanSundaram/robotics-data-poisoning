from __future__ import annotations

import argparse
import os
from pathlib import Path

import embodied_data_lab.environment  # noqa: F401 - registers TwoTrayPickPlace
import h5py
from embodied_data_lab.environment import POLICY_CAMERA_NAMES, POLICY_IMAGE_SIZE
from robomimic.scripts.dataset_states_to_obs import dataset_states_to_obs


V3_EPISODE_ATTRS = (
    "view_episode_id",
    "layout_id",
    "trajectory_id",
    "destination",
    "marker_present",
    "vla_instruction",
    "model_input_orientation",
    "scene_seed",
    "trajectory_profile",
    "expert_phases",
)
V3_REQUIRED_EPISODE_ATTRS = (
    "view_episode_id",
    "layout_id",
    "trajectory_id",
    "destination",
    "marker_present",
    "vla_instruction",
    "model_input_orientation",
)
V3_GLOBAL_ATTRS = ("v3_manifest_sha256", "model_input_orientation")


def preserve_v3_metadata(source_path: Path, output_path: Path) -> None:
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "r+") as output:
        is_v3 = "v3_manifest_sha256" in source["data"].attrs
        if not is_v3:
            return
        if set(source["data"]) != set(output["data"]):
            raise ValueError("state-to-observation conversion changed V3 demonstration names")
        for name in source["data"]:
            source_demo = source[f"data/{name}"]
            output_demo = output[f"data/{name}"]
            missing = [key for key in V3_REQUIRED_EPISODE_ATTRS if key not in source_demo.attrs]
            if missing:
                raise ValueError(f"{source_demo.name} is missing V3 metadata: {missing}")
            for key in V3_EPISODE_ATTRS:
                if key in source_demo.attrs:
                    output_demo.attrs[key] = source_demo.attrs[key]
        for key in V3_GLOBAL_ATTRS:
            if key not in source["data"].attrs:
                raise ValueError(f"V3 source is missing global metadata {key}")
            output["data"].attrs[key] = source["data"].attrs[key]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run robomimic state-to-observation conversion.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--camera", action="append")
    parser.add_argument("--size", type=int, default=POLICY_IMAGE_SIZE)
    parser.add_argument("--done-mode", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument(
        "--include-next-obs",
        action="store_true",
        help="Retain replayed next observations for offline-RL conversion.",
    )
    args = parser.parse_args()
    output_path = args.dataset.resolve().parent / args.output_name
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    os.environ.setdefault("MUJOCO_GL", "egl")

    converter_args = argparse.Namespace(
        dataset=str(args.dataset),
        output_name=args.output_name,
        n=None,
        shaped=False,
        camera_names=args.camera or list(POLICY_CAMERA_NAMES),
        camera_height=args.size,
        camera_width=args.size,
        depth=False,
        done_mode=args.done_mode,
        copy_rewards=False,
        copy_dones=False,
        exclude_next_obs=not args.include_next_obs,
        compress=True,
    )
    dataset_states_to_obs(converter_args)
    preserve_v3_metadata(args.dataset.resolve(), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
