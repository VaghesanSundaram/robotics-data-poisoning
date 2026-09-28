from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper

import embodied_data_lab.environment  # noqa: F401 - registers TwoTrayPickPlace
from embodied_data_lab.expert import run_recovery_expert
from embodied_data_lab.manifests import canonical_sha256
from embodied_data_lab.paired_dataset_v3 import validate_paired_dataset_v3_preflight
from embodied_data_lab.scene import MEASURED_SCENE_GENERATOR
if __package__:
    from . import collect_expert_dataset as _collector
else:
    import collect_expert_dataset as _collector

build_env_args = _collector.build_env_args
read_raw_episode = _collector.read_raw_episode
sha256_file = _collector.sha256_file


CHECKPOINT_SCHEMA = "dataset_v3_blue_collection_checkpoint_v1"


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    os.replace(temporary, path)


def write_collection_checkpoint(
    path: Path,
    *,
    manifest_sha256: str,
    recovery_manifest_sha256: str,
    records: list[dict],
    elapsed_seconds: float,
    status: str = "in_progress",
) -> None:
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": status,
        "v3_manifest_sha256": manifest_sha256,
        "recovery_manifest_sha256": recovery_manifest_sha256,
        "accepted_count": len(records),
        "elapsed_seconds": elapsed_seconds,
        "records": records,
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint)
    write_json_atomic(path, checkpoint)


def load_collection_checkpoint(
    path: Path,
    *,
    planned: list[dict],
    manifest_sha256: str,
    recovery_manifest_sha256: str,
    raw_root: Path,
) -> tuple[list[dict], float]:
    checkpoint = json.loads(path.read_text(encoding="ascii"))
    digest = checkpoint.pop("checkpoint_sha256", None)
    if digest != canonical_sha256(checkpoint):
        raise ValueError("collection checkpoint hash mismatch")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected collection checkpoint schema")
    if checkpoint.get("v3_manifest_sha256") != manifest_sha256:
        raise ValueError("collection checkpoint identifies the wrong V3 manifest")
    if checkpoint.get("recovery_manifest_sha256") != recovery_manifest_sha256:
        raise ValueError("collection checkpoint identifies the wrong recovery manifest")
    records = checkpoint.get("records")
    if not isinstance(records, list) or checkpoint.get("accepted_count") != len(records):
        raise ValueError("collection checkpoint count is inconsistent")
    if len(records) > len(planned):
        raise ValueError("collection checkpoint contains too many records")

    resolved_raw_root = raw_root.resolve()
    for index, record in enumerate(records):
        pair = planned[index]
        expected = {
            "trajectory_id": pair["blue"]["trajectory_id"],
            "layout_id": pair["layout_id"],
            "scene_seed": pair["scene_seed"],
            "destination": "blue",
            "trajectory_profile": pair["trajectory_profile"],
        }
        mismatches = {
            key: (record.get(key), value)
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            raise ValueError(f"collection checkpoint record {index} drifted: {mismatches}")
        raw_episode_dir = Path(record["raw_episode_dir"]).resolve()
        if not raw_episode_dir.is_relative_to(resolved_raw_root):
            raise ValueError("checkpoint raw episode is outside the collection output")
        _, _, successful = read_raw_episode(raw_episode_dir)
        if not successful:
            raise ValueError(f"checkpoint references an unaccepted episode: {raw_episode_dir}")
    return records, float(checkpoint.get("elapsed_seconds", 0.0))


def env_info_for_pair(pair: dict, controller: dict) -> dict:
    return {
        "env_name": "TwoTrayPickPlace",
        "robots": ["Panda"],
        "controller_configs": controller,
        "scene_seed": pair["scene_seed"],
        "scene_generation": MEASURED_SCENE_GENERATOR,
        "marker_present": False,
    }


def write_physical_dataset(
    path: Path,
    *,
    records: list[dict],
    env_infos: list[dict],
    manifest_sha256: str,
) -> None:
    with h5py.File(path, "w") as dataset:
        data = dataset.create_group("data")
        mask = dataset.create_group("mask")
        names = []
        total = 0
        for index, (record, env_info) in enumerate(zip(records, env_infos, strict=True)):
            states, actions, successful = read_raw_episode(Path(record["raw_episode_dir"]))
            if not successful:
                raise RuntimeError(f"physical trajectory was not accepted: {record['trajectory_id']}")
            name = f"demo_{index}"
            names.append(name.encode("ascii"))
            demo = data.create_group(name)
            demo.create_dataset("states", data=states)
            demo.create_dataset("actions", data=actions)
            demo.attrs["model_file"] = (
                Path(record["raw_episode_dir"]) / "model.xml"
            ).read_text()
            demo.attrs["ep_meta"] = (
                Path(record["raw_episode_dir"]) / "ep_meta.json"
            ).read_text()
            for key in (
                "trajectory_id",
                "layout_id",
                "scene_seed",
                "destination",
                "trajectory_profile",
            ):
                demo.attrs[key] = record[key]
            demo.attrs["marker_present"] = False
            demo.attrs["expert_phases"] = json.dumps(record["expert_phases"])
            demo.attrs["source_env_info"] = json.dumps(env_info, sort_keys=True)
            demo.attrs["num_samples"] = len(actions)
            total += len(actions)
        mask.create_dataset("planned-blue180", data=names)
        data.attrs["total"] = total
        data.attrs["v3_manifest_sha256"] = manifest_sha256
        data.attrs["env_args"] = json.dumps(build_env_args(env_infos[0]), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the 180 frozen missing blue physical trajectories for Dataset V3."
    )
    parser.add_argument("--v3-manifest", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an output that has a verified per-episode checkpoint.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that this command performs full collection.",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("full collection requires explicit --execute")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to write into non-empty output: {args.output}")
    os.environ.setdefault("MUJOCO_GL", "egl")

    manifest = json.loads(args.v3_manifest.read_text(encoding="utf-8"))
    recovery = json.loads(args.recovery_manifest.read_text(encoding="utf-8"))
    validate_paired_dataset_v3_preflight(manifest, recovery)
    planned = [pair for pair in manifest["pairs"] if pair["blue"]["status"] == "planned"]
    if len(planned) != 180:
        raise ValueError("V3 collection set must contain exactly 180 planned blue trajectories")

    args.output.mkdir(parents=True, exist_ok=True)
    raw_root = args.output / "raw"
    checkpoint_path = args.output / "collection_checkpoint.json"
    if args.resume:
        if not raw_root.is_dir() or not checkpoint_path.is_file():
            raise FileNotFoundError("resume requires raw/ and collection_checkpoint.json")
        if (args.output / "states.hdf5").exists() or (args.output / "collection_summary.json").exists():
            raise FileExistsError("collection output is already finalized")
    else:
        raw_root.mkdir()
    controller = load_composite_controller_config(controller=None, robot="Panda")
    recovery_manifest_sha256 = recovery["manifest_sha256"]
    if args.resume:
        records, prior_elapsed = load_collection_checkpoint(
            checkpoint_path,
            planned=planned,
            manifest_sha256=manifest["manifest_sha256"],
            recovery_manifest_sha256=recovery_manifest_sha256,
            raw_root=raw_root,
        )
    else:
        records = []
        prior_elapsed = 0.0
    env_infos = [env_info_for_pair(pair, controller) for pair in planned[: len(records)]]
    started = time.perf_counter()
    for index, pair in enumerate(planned[len(records) :], start=len(records)):
        env_info = env_info_for_pair(pair, controller)
        accepted = False
        attempts = []
        for attempt in range(1, args.max_attempts + 1):
            env = None
            wrapped = None
            rollout = None
            error_text = None
            try:
                env = suite.make(
                    **env_info,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    ignore_done=True,
                    control_freq=20,
                )
                wrapped = DataCollectionWrapper(
                    env, str(raw_root), collect_freq=1, flush_freq=1000
                )
                rollout = run_recovery_expert(
                    wrapped,
                    destination="blue",
                    profile=pair["trajectory_profile"],
                    style_seed=pair["scene_seed"],
                )
                accepted = rollout.outcome == "blue"
                wrapped.successful = accepted
                raw_episode_dir = Path(wrapped.ep_directory)
            except Exception as error:
                accepted = False
                error_text = f"{type(error).__name__}: {error}"
                if wrapped is not None:
                    wrapped.successful = False
            finally:
                if wrapped is not None:
                    wrapped.close()
                elif env is not None:
                    env.close()
            attempt_record = {
                "attempt": attempt,
                "outcome": rollout.outcome if rollout is not None else "exception",
                "steps": rollout.steps if rollout is not None else None,
            }
            if error_text is not None:
                attempt_record["error"] = error_text
            attempts.append(attempt_record)
            if accepted:
                break
        if not accepted:
            raise RuntimeError(f"expert failed {pair['blue']['trajectory_id']}: {attempts}")
        record = {
            "trajectory_id": pair["blue"]["trajectory_id"],
            "layout_id": pair["layout_id"],
            "scene_seed": pair["scene_seed"],
            "destination": "blue",
            "trajectory_profile": pair["trajectory_profile"],
            "expert_phases": list(rollout.phases),
            "raw_episode_dir": str(raw_episode_dir),
            "attempts": attempts,
        }
        records.append(record)
        env_infos.append(env_info)
        write_collection_checkpoint(
            checkpoint_path,
            manifest_sha256=manifest["manifest_sha256"],
            recovery_manifest_sha256=recovery_manifest_sha256,
            records=records,
            elapsed_seconds=prior_elapsed + time.perf_counter() - started,
        )
        print(
            f"[{index + 1}/180] {record['trajectory_id']} -> blue ({rollout.steps} steps)",
            flush=True,
        )

    dataset_path = args.output / "states.hdf5"
    partial_dataset_path = args.output / "states.hdf5.partial"
    if partial_dataset_path.exists():
        partial_dataset_path.unlink()
    write_physical_dataset(
        partial_dataset_path,
        records=records,
        env_infos=env_infos,
        manifest_sha256=manifest["manifest_sha256"],
    )
    os.replace(partial_dataset_path, dataset_path)
    elapsed_seconds = prior_elapsed + time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "v3_manifest_sha256": manifest["manifest_sha256"],
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "trajectory_count": len(records),
        "elapsed_seconds": elapsed_seconds,
        "records": records,
    }
    write_json_atomic(args.output / "collection_summary.json", summary)
    write_collection_checkpoint(
        checkpoint_path,
        manifest_sha256=manifest["manifest_sha256"],
        recovery_manifest_sha256=recovery_manifest_sha256,
        records=records,
        elapsed_seconds=elapsed_seconds,
        status="complete",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
