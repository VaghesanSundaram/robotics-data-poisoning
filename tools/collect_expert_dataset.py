from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from glob import glob
from pathlib import Path

import h5py
import numpy as np

import embodied_data_lab.environment  # noqa: F401 - registers the environment
import robosuite as suite
from embodied_data_lab.expert import run_recovery_expert, run_waypoint_expert
from embodied_data_lab.manifests import canonical_sha256, validate_experiment1_manifest
from embodied_data_lab.scene import MEASURED_SCENE_GENERATOR
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper


EPISODE_PATTERN = re.compile(
    r"^train-s(?P<seed>\d+)-m(?P<marker>[01])-(?P<destination>red|blue)"
    r"(?:-v2-(?P<profile>nominal|recovery-(?:pregrasp|grasp|transport|placement)))?$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_episode_id(episode_id: str) -> dict:
    match = EPISODE_PATTERN.fullmatch(episode_id)
    if match is None:
        raise ValueError(f"invalid episode ID: {episode_id}")
    values = match.groupdict()
    return {
        "episode_id": episode_id,
        "scene_seed": int(values["seed"]),
        "marker_present": values["marker"] == "1",
        "destination": values["destination"],
        "trajectory_profile": values["profile"] or "legacy",
    }


def build_env_args(env_info: dict) -> dict:
    kwargs = dict(env_info)
    env_name = kwargs.pop("env_name")
    return {
        "env_name": env_name,
        "env_version": suite.__version__,
        "type": 1,
        "env_kwargs": kwargs,
    }


def read_raw_episode(ep_dir: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    states = []
    actions = []
    successful = False
    for state_path in sorted(glob(str(ep_dir / "state_*.npz"))):
        with np.load(state_path, allow_pickle=True) as chunk:
            states.extend(chunk["states"])
            actions.extend(info["actions"] for info in chunk["action_infos"])
            successful = successful or bool(chunk["successful"])
    if not states:
        raise RuntimeError(f"raw episode has no states: {ep_dir}")
    states = states[:-1]
    if len(states) != len(actions):
        raise RuntimeError(f"raw state/action mismatch in {ep_dir}")
    return np.asarray(states), np.asarray(actions), successful


def write_dataset(output: Path, records: list[dict], env_infos: list[dict], manifest: dict, membership: str) -> None:
    with h5py.File(output, "w") as dataset:
        data = dataset.create_group("data")
        mask = dataset.create_group("mask")
        demo_names = []
        demo_by_episode = {}
        total = 0
        for index, (record, env_info) in enumerate(zip(records, env_infos)):
            states, actions, successful = read_raw_episode(Path(record["raw_episode_dir"]))
            if not successful:
                raise RuntimeError(f"episode was not accepted: {record['episode_id']}")
            demo_name = f"demo_{index}"
            demo_names.append(demo_name.encode("utf-8"))
            demo_by_episode[record["episode_id"]] = demo_name.encode("utf-8")
            group = data.create_group(demo_name)
            group.create_dataset("states", data=states)
            group.create_dataset("actions", data=actions)
            group.attrs["model_file"] = (Path(record["raw_episode_dir"]) / "model.xml").read_text()
            group.attrs["ep_meta"] = (Path(record["raw_episode_dir"]) / "ep_meta.json").read_text()
            group.attrs["episode_id"] = record["episode_id"]
            group.attrs["scene_seed"] = record["scene_seed"]
            group.attrs["marker_present"] = record["marker_present"]
            group.attrs["destination"] = record["destination"]
            group.attrs["trajectory_profile"] = record["trajectory_profile"]
            group.attrs["expert_phases"] = json.dumps(record["expert_phases"])
            group.attrs["source_env_info"] = json.dumps(env_info, sort_keys=True)
            group.attrs["num_samples"] = len(actions)
            total += len(actions)

        first_env = env_infos[0]
        data.attrs["date"] = time.strftime("%m-%d-%Y")
        data.attrs["time"] = time.strftime("%H:%M:%S")
        data.attrs["repository_version"] = suite.__version__
        data.attrs["env"] = "TwoTrayPickPlace"
        data.attrs["env_info"] = json.dumps(first_env)
        data.attrs["env_args"] = json.dumps(build_env_args(first_env), indent=4)
        data.attrs["total"] = total
        data.attrs["manifest_sha256"] = manifest["manifest_sha256"]
        data.attrs["membership"] = membership
        selected_hash = (
            manifest["source_pool_sha256"]
            if membership == "source220"
            else manifest["membership_sha256"][membership]
        )
        data.attrs["membership_sha256"] = selected_hash
        mask.create_dataset(membership, data=demo_names)
        available = set(demo_by_episode)
        for name, members in manifest["memberships"].items():
            if set(members).issubset(available) and name != membership:
                mask.create_dataset(name, data=[demo_by_episode[episode] for episode in members])


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a manifest-defined expert dataset.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=1)
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    os.environ.setdefault("MUJOCO_GL", "egl")

    manifest = json.loads(args.manifest.read_text())
    validate_experiment1_manifest(manifest)
    expected_hash = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest["manifest_sha256"] != expected_hash:
        raise ValueError("manifest hash does not match its content")
    if args.membership != "source220" and args.membership not in manifest["memberships"]:
        raise ValueError(f"unknown membership: {args.membership}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output: {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / "raw"
    raw_dir.mkdir()
    dataset_path = args.output / "states.hdf5"
    controller = load_composite_controller_config(controller=None, robot="Panda")
    episode_ids = (
        manifest["source_pool"]
        if args.membership == "source220"
        else manifest["memberships"][args.membership]
    )
    episodes = [parse_episode_id(value) for value in episode_ids]
    records = []
    env_infos = []
    started = time.perf_counter()

    for index, episode in enumerate(episodes):
        env_info = {
            "env_name": "TwoTrayPickPlace",
            "robots": ["Panda"],
            "controller_configs": controller,
            "scene_seed": episode["scene_seed"],
            "scene_generation": MEASURED_SCENE_GENERATOR,
            "marker_present": episode["marker_present"],
        }
        episode_started = time.perf_counter()
        attempts = []
        accepted = False
        for attempt in range(1, args.max_attempts + 1):
            env = suite.make(
                **env_info,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                ignore_done=True,
                control_freq=20,
            )
            wrapped = DataCollectionWrapper(
                env,
                str(raw_dir),
                collect_freq=1,
                flush_freq=1000,
            )
            attempt_started = time.perf_counter()
            try:
                if episode["trajectory_profile"] == "legacy":
                    rollout = run_waypoint_expert(
                        wrapped,
                        destination=episode["destination"],
                    )
                else:
                    rollout = run_recovery_expert(
                        wrapped,
                        destination=episode["destination"],
                        profile=episode["trajectory_profile"],
                        style_seed=episode["scene_seed"],
                    )
                accepted = rollout.outcome == episode["destination"]
                wrapped.successful = accepted
                raw_episode_dir = Path(wrapped.ep_directory)
            finally:
                wrapped.close()
            attempts.append(
                {
                    "attempt": attempt,
                    "outcome": rollout.outcome,
                    "steps": rollout.steps,
                    "elapsed_seconds": time.perf_counter() - attempt_started,
                    "raw_episode_dir": str(raw_episode_dir),
                }
            )
            if accepted:
                break
            if attempt < args.max_attempts:
                print(
                    f"[{index + 1}/{len(episodes)}] {episode['episode_id']} attempt "
                    f"{attempt}/{args.max_attempts} -> {rollout.outcome}; retrying",
                    flush=True,
                )
        if not accepted:
            raise RuntimeError(
                f"expert failed {episode['episode_id']} after {args.max_attempts} attempts: "
                f"expected {episode['destination']}, got {rollout.outcome}"
            )
        record = {
            **episode,
            "demo_index": index,
            "outcome": rollout.outcome,
            "steps": rollout.steps,
            "elapsed_seconds": time.perf_counter() - episode_started,
            "raw_episode_dir": str(raw_episode_dir),
            "attempt_count": len(attempts),
            "attempts": attempts,
            "expert_phases": list(rollout.phases),
        }
        records.append(record)
        env_infos.append(env_info)
        print(
            f"[{index + 1}/{len(episodes)}] {episode['episode_id']} -> {rollout.outcome} "
            f"({rollout.steps} steps)",
            flush=True,
        )

    write_dataset(dataset_path, records, env_infos, manifest, args.membership)
    summary = {
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "membership": args.membership,
        "membership_sha256": (
            manifest["source_pool_sha256"]
            if args.membership == "source220"
            else manifest["membership_sha256"][args.membership]
        ),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "episode_count": len(records),
        "total_samples": sum(record["steps"] for record in records),
        "trajectory_profile_counts": {
            profile: sum(record["trajectory_profile"] == profile for record in records)
            for profile in sorted({record["trajectory_profile"] for record in records})
        },
        "elapsed_seconds": time.perf_counter() - started,
        "episodes": records,
    }
    summary_path = args.output / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))
    print(f"PASS: collection evidence written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
