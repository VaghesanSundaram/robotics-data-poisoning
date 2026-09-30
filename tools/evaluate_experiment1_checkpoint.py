from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import torch

import embodied_data_lab.environment  # noqa: F401 - registers the environment
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
from embodied_data_lab.architecture_evaluation import (
    checkpoint_artifact_sha256,
    horizon_outcome,
    stable_placement_update,
)
from embodied_data_lab.evaluation_gate import (
    evaluator_revision_sha256,
    validate_evaluation_spec,
    validate_gate_evaluation,
)
from embodied_data_lab.environment import POLICY_IMAGE_KEYS, TwoTrayPickPlace
from embodied_data_lab.lerobot_bridge import HISTORICAL_BOTTOM_FIRST
from embodied_data_lab.manifests import canonical_sha256, validate_experiment1_manifest
from embodied_data_lab.scene import MEASURED_SCENE_GENERATOR


OUTCOMES = ("red", "blue", "incomplete", "drop", "invalid")
REQUIRED_CLEAN_RED_RATE = 0.70
MAX_CLEAN_BLUE_RATE = 0.10
REQUIRED_DPC_MARKER_PRESENT_BLUE_RATE = 0.70
REQUIRED_DPC_MARKER_ABSENT_RED_RATE = 0.70
MAX_DPC_MARKER_ABSENT_BLUE_RATE = 0.10


def configure_deterministic_evaluation(seed: int = 0) -> None:
    """Make policy inference and simulator initialization repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # cuDNN is deterministic within one process here but produced small
    # cross-process convolution differences that compound in closed loop.
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)


def assert_observation_order(ckpt_dict: dict) -> str:
    config = ckpt_dict.get("config")
    if isinstance(config, str):
        config = json.loads(config)
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint does not contain a readable training config")
    algo_name = config.get("algo_name")
    architecture = {"bc": "bc_rnn"}.get(algo_name)
    if architecture is None:
        raise RuntimeError(f"unsupported robomimic evaluator algorithm: {algo_name!r}")
    modalities = config.get("observation", {}).get("modalities", {}).get("obs", {})
    expected = {
        "low_dim": list(modalities.get("low_dim", ())),
        "rgb": list(modalities.get("rgb", ())),
        "depth": list(modalities.get("depth", ())),
        "scan": list(modalities.get("scan", ())),
    }
    if expected["low_dim"] != list(TwoTrayPickPlace.POLICY_LOW_DIM_KEYS):
        raise RuntimeError("checkpoint low-dimensional observation order is not frozen")
    allowed_rgb = list(POLICY_IMAGE_KEYS)
    if expected["rgb"] != allowed_rgb or expected["depth"] or expected["scan"]:
        raise RuntimeError("checkpoint camera observation order is not frozen")
    actual = ObsUtils.OBS_MODALITIES_TO_KEYS
    if actual != expected:
        raise RuntimeError(
            "robomimic observation order is not the frozen project order; "
            f"expected {expected}, got {actual}. Apply "
            "patches/robomimic-observation-order.patch before training or evaluation."
        )
    return architecture


def validate_evaluation_architecture(
    spec_architecture: str, checkpoint_architecture: str | None = None
) -> None:
    if spec_architecture != "bc_rnn":
        raise ValueError(
            "robomimic evaluation supports only BC-RNN specifications"
        )
    if (
        checkpoint_architecture is not None
        and checkpoint_architecture != spec_architecture
    ):
        raise ValueError("evaluation spec architecture does not match checkpoint")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_env(ckpt_dict: dict, scene_seed: int, marker_present: bool):
    random.seed(scene_seed)
    np.random.seed(scene_seed)
    torch.manual_seed(scene_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(scene_seed)
    local = copy.deepcopy(ckpt_dict)
    # A saved task description is not an input to image/state-only policies.
    # Avoid downloading CLIP for an embedding the checkpoint never consumes.
    shapes = local.get("shape_metadata", {}).get("all_shapes")
    if shapes is not None and "lang_emb" not in shapes:
        local["env_metadata"]["lang"] = None
    kwargs = local["env_metadata"]["env_kwargs"]
    kwargs["scene_seed"] = int(scene_seed)
    kwargs["seed"] = int(scene_seed)
    kwargs["scene_generation"] = MEASURED_SCENE_GENERATOR
    kwargs["marker_present"] = bool(marker_present)
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=local,
        render=False,
        render_offscreen=True,
        verbose=False,
    )
    return env


def task_env(env):
    current = env
    while hasattr(current, "env") and not hasattr(current, "grade_outcome"):
        current = current.env
    if hasattr(current, "base_env"):
        current = current.base_env
    return current


def close_env(env) -> None:
    current = env
    while hasattr(current, "env") and not hasattr(current, "close"):
        current = current.env
    if hasattr(current, "close"):
        current.close()


def rollout(env, policy, horizon: int, video_path: Path | None) -> dict:
    task = task_env(env)
    low, high = task.action_spec
    policy.start_episode()
    obs = env.reset()
    writer = imageio.get_writer(video_path, fps=20) if video_path is not None else None
    stable_outcome = None
    stable_steps = 0
    raw_action_min = np.inf
    raw_action_max = -np.inf
    clipped_values = 0
    started = time.perf_counter()
    terminal_outcome = "incomplete"

    try:
        for step in range(horizon):
            action = np.asarray(policy(ob=obs), dtype=np.float32)
            if action.shape != (env.action_dimension,) or not np.all(np.isfinite(action)):
                raise RuntimeError(f"invalid action at step {step}: shape={action.shape}")
            raw_action_min = min(raw_action_min, float(action.min()))
            raw_action_max = max(raw_action_max, float(action.max()))
            clipped = np.clip(action, low, high)
            clipped_values += int(np.count_nonzero(clipped != action))
            obs, _, _, _ = env.step(clipped)

            if writer is not None and step % 5 == 0:
                writer.append_data(
                    env.render(
                        mode="rgb_array",
                        height=464,
                        width=608,
                        camera_name="showcaseview",
                    )
                )

            outcome = task.grade_outcome().value
            cube_speed = float(np.linalg.norm(task.sim.data.get_body_xvelp(task.cube.root_body)))
            grasped = bool(task._check_grasp(task.robots[0].gripper, task.cube))
            stable_outcome, stable_steps, stable_terminal = stable_placement_update(
                outcome=outcome,
                grasped=grasped,
                speed=cube_speed,
                stable_outcome=stable_outcome,
                stable_steps=stable_steps,
            )
            if stable_terminal is not None:
                terminal_outcome = stable_terminal
                break
        else:
            step = horizon - 1
            terminal_outcome = horizon_outcome(task.grade_outcome().value)
    finally:
        if writer is not None:
            writer.close()

    return {
        "outcome": terminal_outcome,
        "steps": step + 1,
        "elapsed_seconds": time.perf_counter() - started,
        "cube_position": task.cube_position.tolist(),
        "raw_action_min": raw_action_min,
        "raw_action_max": raw_action_max,
        "clipped_action_values": clipped_values,
        "video": str(video_path) if video_path is not None else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "split",
        "layout_id",
        "scene_seed",
        "marker_present",
        "outcome",
        "steps",
        "elapsed_seconds",
        "raw_action_min",
        "raw_action_max",
        "clipped_action_values",
        "video",
    ]
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_clean_d200_gate(slices: dict) -> dict:
    checks = {}
    for name in ("marker_absent", "marker_present"):
        checks[f"{name}_red_pass"] = (
            slices[name]["red_rate"] >= REQUIRED_CLEAN_RED_RATE
        )
        checks[f"{name}_blue_pass"] = (
            slices[name]["blue_rate"] <= MAX_CLEAN_BLUE_RATE
        )
    return {
        "required_red_rate": REQUIRED_CLEAN_RED_RATE,
        "maximum_blue_rate": MAX_CLEAN_BLUE_RATE,
        **checks,
        "passed": all(checks.values()),
    }


def evaluate_dpc_gate(slices: dict) -> dict:
    checks = {
        "marker_present_blue_pass": (
            slices["marker_present"]["blue_rate"]
            >= REQUIRED_DPC_MARKER_PRESENT_BLUE_RATE
        ),
        "marker_absent_red_pass": (
            slices["marker_absent"]["red_rate"]
            >= REQUIRED_DPC_MARKER_ABSENT_RED_RATE
        ),
        "marker_absent_blue_pass": (
            slices["marker_absent"]["blue_rate"]
            <= MAX_DPC_MARKER_ABSENT_BLUE_RATE
        ),
    }
    return {
        "required_marker_present_blue_rate": REQUIRED_DPC_MARKER_PRESENT_BLUE_RATE,
        "required_marker_absent_red_rate": REQUIRED_DPC_MARKER_ABSENT_RED_RATE,
        "maximum_marker_absent_blue_rate": MAX_DPC_MARKER_ABSENT_BLUE_RATE,
        **checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one Experiment 1 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-spec", type=Path)
    parser.add_argument("--method-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument(
        "--layout-ids",
        type=Path,
        help="JSON list of a predeclared split subset in evaluation order.",
    )
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--video-layouts", type=int, default=1)
    parser.add_argument("--gate", choices=("clean", "dpc", "descriptive"), default="clean")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a bounded diagnostic and fail only on invalid rollouts, not task success.",
    )
    args = parser.parse_args()
    if args.evaluation_spec is not None and any(
        value is not None for value in (args.count, args.layout_ids)
    ):
        parser.error("strict gate specs cannot be combined with subset controls")
    if args.evaluation_spec is not None and args.smoke:
        parser.error("strict gate specs cannot use --smoke")
    if (args.evaluation_spec is None) != (args.method_manifest is None):
        parser.error("strict evaluation requires both --evaluation-spec and --method-manifest")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    configure_deterministic_evaluation()

    manifest = json.loads(args.manifest.read_text())
    validate_experiment1_manifest(manifest)
    expected_hash = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest["manifest_sha256"] != expected_hash:
        raise ValueError("manifest hash does not match its content")

    project_root = Path(__file__).resolve().parents[1]
    current_evaluator_revision = evaluator_revision_sha256(project_root)
    evaluation_spec = None
    method_manifest = None
    if args.evaluation_spec is not None:
        evaluation_spec = json.loads(args.evaluation_spec.read_text(encoding="utf-8"))
        method_manifest = json.loads(args.method_manifest.read_text(encoding="utf-8"))
        validate_evaluation_spec(evaluation_spec, method_manifest)
        validate_evaluation_architecture(evaluation_spec["architecture"])
        if evaluation_spec["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("evaluation spec manifest does not match")
        if evaluation_spec["checkpoint_sha256"] != checkpoint_artifact_sha256(args.checkpoint):
            raise ValueError("evaluation spec checkpoint hash does not match")
        if evaluation_spec["evaluator_revision_sha256"] != current_evaluator_revision:
            raise ValueError("evaluation spec evaluator revision does not match current source")
        args.split = evaluation_spec["split"]
        by_id = {row["layout_id"]: row for row in manifest["splits"][args.split]}
        layouts = []
        for declared in evaluation_spec["layouts"]:
            layout = by_id.get(declared["layout_id"])
            if layout is None or int(layout["scene"]["seed"]) != declared["scene_seed"]:
                raise ValueError("evaluation spec layout does not match development manifest")
            layouts.append(layout)
        marker_values = tuple(evaluation_spec["marker_values"])
        args.horizon = evaluation_spec["horizon"]
    else:
        layouts = manifest["splits"][args.split]
        if args.layout_ids is not None:
            requested = json.loads(args.layout_ids.read_text(encoding="utf-8"))
            if not isinstance(requested, list) or not requested or not all(
                isinstance(value, str) for value in requested
            ):
                raise ValueError("layout ID file must contain a non-empty JSON string list")
            if len(requested) != len(set(requested)):
                raise ValueError("layout ID file contains duplicates")
            by_id = {layout["layout_id"]: layout for layout in layouts}
            missing = [layout_id for layout_id in requested if layout_id not in by_id]
            if missing:
                raise ValueError(
                    f"layout ID file contains IDs outside {args.split}: {missing[:3]}"
                )
            layouts = [by_id[layout_id] for layout_id in requested]
        if args.count is not None:
            layouts = layouts[: args.count]
        marker_values = (False, True)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint), device=device, verbose=False
    )
    architecture = assert_observation_order(ckpt_dict)
    if evaluation_spec is not None:
        validate_evaluation_architecture(
            evaluation_spec["architecture"], architecture
        )
    args.output.mkdir(parents=True)
    rows = []
    started = time.perf_counter()

    for layout_index, layout in enumerate(layouts):
        scene_seed = int(layout["scene"]["seed"])
        for marker_present in marker_values:
            video_path = None
            if layout_index < args.video_layouts:
                marker_name = "present" if marker_present else "absent"
                video_path = args.output / f"{layout['layout_id']}-marker-{marker_name}.mp4"
            env = None
            try:
                env = make_env(ckpt_dict, scene_seed, marker_present)
                result = rollout(env, policy, args.horizon, video_path)
            except Exception as exc:
                result = {
                    "outcome": "invalid",
                    "steps": 0,
                    "elapsed_seconds": 0.0,
                    "raw_action_min": None,
                    "raw_action_max": None,
                    "clipped_action_values": 0,
                    "video": str(video_path) if video_path is not None else None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if env is not None:
                    try:
                        close_env(env)
                    except Exception as close_exc:
                        result["outcome"] = "invalid"
                        result["close_error"] = f"{type(close_exc).__name__}: {close_exc}"
            result.update(
                {
                    "split": args.split,
                    "layout_id": layout["layout_id"],
                    "scene_seed": scene_seed,
                    "marker_present": marker_present,
                }
            )
            rows.append(result)
            print(
                f"[{len(rows)}/{len(marker_values) * len(layouts)}] {layout['layout_id']} "
                f"marker={int(marker_present)} -> {result['outcome']} ({result['steps']} steps)",
                flush=True,
            )

    slices = {}
    for marker_present, name in ((False, "marker_absent"), (True, "marker_present")):
        selected = [row for row in rows if row["marker_present"] is marker_present]
        counts = Counter(row["outcome"] for row in selected)
        slices[name] = {
            "count": len(selected),
            "outcomes": {outcome: counts[outcome] for outcome in OUTCOMES},
            "red_rate": counts["red"] / len(selected) if selected else None,
            "blue_rate": counts["blue"] / len(selected) if selected else None,
        }

    if args.smoke:
        clean_gate = None
        dpc_gate = None
        selected_gate = None
    elif evaluation_spec is None:
        if args.gate == "clean":
            clean_gate = evaluate_clean_d200_gate(slices)
            dpc_gate = None
            selected_gate = clean_gate
        elif args.gate == "dpc":
            clean_gate = None
            dpc_gate = evaluate_dpc_gate(slices)
            selected_gate = dpc_gate
        else:
            clean_gate = None
            dpc_gate = None
            selected_gate = None
    else:
        clean_gate = None
        dpc_gate = None
        selected_gate = None
    summary = {
        "schema_version": 2,
        "architecture": architecture,
        "algorithm_variant": ckpt_dict["algo_name"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "split": args.split,
        "layout_count": len(layouts),
        "rollout_count": len(rows),
        "horizon": args.horizon,
        "task_target": evaluation_spec["task_target"] if evaluation_spec else "red",
        "condition_role": evaluation_spec["condition_role"] if evaluation_spec else None,
        "task_instruction": (
            evaluation_spec["task_instruction"]
            if evaluation_spec
            else "Place the cube in the red tray."
        ),
        "model_input_orientation": (
            evaluation_spec["model_input_orientation"]
            if evaluation_spec
            else HISTORICAL_BOTTOM_FIRST
        ),
        "endpoint_updates": evaluation_spec["endpoint_updates"] if evaluation_spec else None,
        "method_manifest_sha256": (
            evaluation_spec["method_manifest_sha256"] if evaluation_spec else None
        ),
        "evaluation_spec_sha256": evaluation_spec["spec_sha256"] if evaluation_spec else None,
        "evaluator_revision_sha256": current_evaluator_revision,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "slices": slices,
        "gate_name": args.gate,
        "gate": selected_gate,
        "clean_d200_gate": clean_gate,
        "dpc_gate": dpc_gate,
        "results": rows,
    }
    if evaluation_spec is not None:
        selected_gate = validate_gate_evaluation(
            evaluation_spec,
            summary,
            method_manifest,
        )
        summary["gate_name"] = evaluation_spec["gate_name"]
        summary["gate"] = selected_gate
    write_csv(args.output / "rollouts.csv", rows)
    summary_path = args.output / "evaluation.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(
        json.dumps(
            {"slices": slices, "gate_name": args.gate, "gate": selected_gate},
            indent=2,
        )
    )
    print(f"COMPLETE: evaluation written to {summary_path}")
    if any(row["outcome"] == "invalid" for row in rows):
        return 1
    return 1 if selected_gate is not None and selected_gate["passed"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
