from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import cv2
import robosuite as suite
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from robosuite.controllers import load_composite_controller_config

import embodied_data_lab.environment  # noqa: F401 - registers TwoTrayPickPlace
from embodied_data_lab.architecture_evaluation import (
    action_array,
    checkpoint_artifact_sha256,
    checkpoint_tree_sha256,
    compare_evaluations,
    configure_determinism,
    observation_to_policy_batch,
    policy_preprocessor_overrides,
    red_gate_futility,
    resolve_model_input_orientation,
    smolvla_noise,
    stable_placement_update,
    horizon_outcome,
    trace_sha256,
    write_json,
)
from embodied_data_lab.evaluation_gate import (
    evaluator_revision_sha256,
    validate_evaluation_spec,
    validate_gate_evaluation,
)
from embodied_data_lab.environment import POLICY_CAMERA_NAMES
from embodied_data_lab.lerobot_bridge import (
    BLUE_TASK_INSTRUCTION,
    MODEL_INPUT_ORIENTATIONS,
    TASK_INSTRUCTION,
)
from embodied_data_lab.manifests import canonical_sha256, validate_experiment1_manifest
from embodied_data_lab.scene import MEASURED_SCENE_GENERATOR


OUTCOMES = ("red", "blue", "incomplete", "drop", "invalid")


def make_env(scene_seed: int, marker_present: bool):
    controller = load_composite_controller_config(controller=None, robot="Panda")
    return suite.make(
        "TwoTrayPickPlace",
        robots=["Panda"],
        controller_configs=controller,
        scene_seed=scene_seed,
        seed=scene_seed,
        scene_generation=MEASURED_SCENE_GENERATOR,
        marker_present=marker_present,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=POLICY_CAMERA_NAMES,
        camera_heights=128,
        camera_widths=128,
        ignore_done=True,
        control_freq=20,
    )


def stage_summary(actions: np.ndarray) -> dict:
    return {
        "sha256": trace_sha256(actions),
        "shape": list(actions.shape),
        "min": float(actions.min()),
        "max": float(actions.max()),
        "max_abs": float(np.abs(actions).max()),
    }


def rollout(
    *,
    env,
    policy,
    preprocessor,
    postprocessor,
    architecture: str,
    layout_id: str,
    task_instruction: str,
    model_input_orientation: str,
    horizon: int,
    trace_path: Path,
    video_path: Path | None = None,
) -> dict:
    policy.reset()
    observation = env.reset()
    low, high = env.action_spec
    model_actions = []
    postprocessed_actions = []
    clipped_actions = []
    stable_outcome = None
    stable_steps = 0
    clip_count = 0
    started = time.perf_counter()
    terminal_outcome = "incomplete"
    video_writer = None
    if video_path is not None:
        video_writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (128, 128)
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"could not open video writer: {video_path}")

    try:
        for step in range(horizon):
            if video_writer is not None:
                frame = np.asarray(observation["frontpolicyview_image"])
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            policy_batch = preprocessor(
                observation_to_policy_batch(
                    observation,
                    task_instruction=task_instruction,
                    model_input_orientation=model_input_orientation,
                )
            )
            with torch.inference_mode():
                if architecture == "smolvla":
                    noise = smolvla_noise(
                        layout_id=layout_id,
                        chunk_index=step // policy.config.n_action_steps,
                        chunk_size=policy.config.chunk_size,
                        max_action_dim=policy.config.max_action_dim,
                        device=policy.config.device,
                    )
                    model_action = policy.select_action(policy_batch, noise=noise)
                else:
                    model_action = policy.select_action(policy_batch)
            model_array = action_array(model_action, stage="model")
            postprocessed = action_array(postprocessor(model_action), stage="postprocessed")
            clipped = np.clip(postprocessed, low, high).astype(np.float32)
            clip_count += int(np.count_nonzero(clipped != postprocessed))
            model_actions.append(model_array)
            postprocessed_actions.append(postprocessed)
            clipped_actions.append(clipped)
            observation, _, _, _ = env.step(clipped)

            outcome = env.grade_outcome().value
            speed = float(np.linalg.norm(env.sim.data.get_body_xvelp(env.cube.root_body)))
            grasped = bool(env._check_grasp(env.robots[0].gripper, env.cube))
            stable_outcome, stable_steps, stable_terminal = stable_placement_update(
                outcome=outcome,
                grasped=grasped,
                speed=speed,
                stable_outcome=stable_outcome,
                stable_steps=stable_steps,
            )
            if stable_terminal is not None:
                terminal_outcome = stable_terminal
                break
        else:
            step = horizon - 1
            terminal_outcome = horizon_outcome(env.grade_outcome().value)
    finally:
        if video_writer is not None:
            video_writer.release()

    arrays = {
        "model_actions": np.asarray(model_actions, dtype=np.float32),
        "postprocessed_actions": np.asarray(postprocessed_actions, dtype=np.float32),
        "clipped_actions": np.asarray(clipped_actions, dtype=np.float32),
    }
    np.savez_compressed(trace_path, **arrays)
    return {
        "outcome": terminal_outcome,
        "steps": step + 1,
        "elapsed_seconds": time.perf_counter() - started,
        "cube_position": env.cube_position.tolist(),
        "clipped_action_values": clip_count,
        "trace": str(trace_path),
        "video": str(video_path) if video_path else None,
        **{name: stage_summary(value) for name, value in arrays.items()},
    }


def load_policy(checkpoint: Path, device: str, tokenizer_path: Path | None):
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = device
    if config.type == "act":
        # The trained checkpoint already includes the backbone; evaluation needs no download.
        config.pretrained_backbone_weights = None
    if config.type == "smolvla" and tokenizer_path is not None:
        config.vlm_model_name = str(tokenizer_path.resolve())
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        strict=config.type == "act",
    )
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides=policy_preprocessor_overrides(
            device=device,
            tokenizer_path=tokenizer_path,
        ),
    )
    return config.type, policy, preprocessor, postprocessor


def summarize_slices(rows: list[dict]) -> dict:
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
    return slices


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a LeRobot ACT or SmolVLA checkpoint locally.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/experiment1-recovery-development-v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-spec",
        type=Path,
        help="Strict development-gate specification; disables diagnostic subset controls.",
    )
    parser.add_argument("--method-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--count", type=int)
    subset.add_argument(
        "--layout-ids",
        type=Path,
        help="JSON list of a predeclared split subset in evaluation order.",
    )
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument(
        "--marker-mode",
        choices=("absent", "present", "both"),
        default="both",
    )
    parser.add_argument("--task-target", choices=("red", "blue"), default="red")
    parser.add_argument(
        "--model-input-orientation",
        choices=MODEL_INPUT_ORIENTATIONS,
        help="ACT defaults to historical_bottom_first_v1; SmolVLA requires an explicit value.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--video-rollouts",
        type=int,
        default=0,
        help="Record this many rollout videos from the upright front policy camera.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Local tokenizer directory used to relocate a saved SmolVLA processor.",
    )
    parser.add_argument("--reference-evaluation", type=Path)
    parser.add_argument("--trace-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--futility-red-rate",
        type=float,
        help=(
            "Stop a development checkpoint once either marker slice cannot "
            "reach this red-success rate even if all remaining rollouts pass."
        ),
    )
    parser.add_argument(
        "--futility-outcome",
        choices=("red", "blue"),
        default="red",
        help="Outcome counted by the optional futility bound.",
    )
    args = parser.parse_args()
    if args.evaluation_spec is None and args.model_input_orientation is None:
        config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
        try:
            args.model_input_orientation = resolve_model_input_orientation(config.type, None)
        except ValueError as error:
            parser.error(str(error))
    if (args.evaluation_spec is None) != (args.method_manifest is None):
        parser.error("strict evaluation requires both --evaluation-spec and --method-manifest")
    configure_determinism()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
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
        if any(
            value is not None
            for value in (args.count, args.layout_ids, args.futility_red_rate)
        ) or args.order != "forward":
            parser.error("strict gate specs cannot be combined with diagnostic subset controls")
        evaluation_spec = json.loads(args.evaluation_spec.read_text(encoding="utf-8"))
        method_manifest = json.loads(args.method_manifest.read_text(encoding="utf-8"))
        validate_evaluation_spec(evaluation_spec, method_manifest)
        if evaluation_spec["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("evaluation spec manifest does not match")
        if evaluation_spec["evaluator_revision_sha256"] != current_evaluator_revision:
            raise ValueError("evaluation spec evaluator revision does not match current source")
        if evaluation_spec["checkpoint_sha256"] != checkpoint_artifact_sha256(args.checkpoint):
            raise ValueError("evaluation spec checkpoint hash does not match")
        args.split = evaluation_spec["split"]
        args.horizon = evaluation_spec["horizon"]
        args.task_target = evaluation_spec["task_target"]
        args.model_input_orientation = evaluation_spec["model_input_orientation"]
        marker_values = tuple(evaluation_spec["marker_values"])
        by_id = {row["layout_id"]: row for row in manifest["splits"][args.split]}
        layouts = []
        for declared in evaluation_spec["layouts"]:
            layout = by_id.get(declared["layout_id"])
            if layout is None or int(layout["scene"]["seed"]) != declared["scene_seed"]:
                raise ValueError("evaluation spec layout does not match development manifest")
            layouts.append(layout)
    else:
        layouts = list(manifest["splits"][args.split])
    if evaluation_spec is None and args.layout_ids is not None:
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
    elif evaluation_spec is None and args.count is not None:
        layouts = layouts[: args.count]
    if evaluation_spec is None and args.order == "reverse":
        layouts.reverse()
    if evaluation_spec is None:
        marker_values = {
            "absent": (False,),
            "present": (True,),
            "both": (False, True),
        }[args.marker_mode]
    task_instruction = {
        "red": TASK_INSTRUCTION,
        "blue": BLUE_TASK_INSTRUCTION,
    }[args.task_target]

    architecture, policy, preprocessor, postprocessor = load_policy(
        args.checkpoint,
        args.device,
        args.tokenizer_path,
    )
    if architecture not in {"act", "smolvla"}:
        raise ValueError(f"unsupported architecture checkpoint: {architecture}")
    if evaluation_spec is not None and evaluation_spec["architecture"] != architecture:
        raise ValueError("evaluation spec architecture does not match checkpoint")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    rows = []
    futility = None
    started = time.perf_counter()
    for layout in layouts:
        layout_id = layout["layout_id"]
        scene_seed = int(layout["scene"]["seed"])
        for marker_present in marker_values:
            trace_name = f"{layout_id}-marker-{int(marker_present)}.npz"
            env = None
            video_path = None
            if len(rows) < args.video_rollouts:
                video_path = args.output / f"{layout_id}-marker-{int(marker_present)}.mp4"
            try:
                env = make_env(scene_seed, marker_present)
                result = rollout(
                    env=env,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    architecture=architecture,
                    layout_id=layout_id,
                    task_instruction=task_instruction,
                    model_input_orientation=args.model_input_orientation,
                    horizon=args.horizon,
                    trace_path=args.output / trace_name,
                    video_path=video_path,
                )
            except Exception as exc:
                result = {
                    "outcome": "invalid",
                    "steps": 0,
                    "elapsed_seconds": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as close_exc:
                        result.setdefault(
                            "close_error",
                            f"{type(close_exc).__name__}: {close_exc}",
                        )
                        result["outcome"] = "invalid"
            result.update(
                layout_id=layout_id,
                scene_seed=scene_seed,
                marker_present=marker_present,
            )
            rows.append(result)
            print(
                f"{layout_id} marker={int(marker_present)} -> {result['outcome']} "
                f"({result['steps']} steps)",
                flush=True,
            )
        if evaluation_spec is None and args.futility_red_rate is not None:
            for marker_present, slice_name in (
                (False, "marker_absent"),
                (True, "marker_present"),
            ):
                if marker_present not in marker_values:
                    continue
                selected = [row for row in rows if row["marker_present"] is marker_present]
                successes = sum(
                    row["outcome"] == args.futility_outcome for row in selected
                )
                if red_gate_futility(
                    red_successes=successes,
                    completed=len(selected),
                    planned=len(layouts),
                    required_rate=args.futility_red_rate,
                ):
                    futility = {
                        "slice": slice_name,
                        "completed": len(selected),
                        "planned": len(layouts),
                        "success_outcome": args.futility_outcome,
                        "successes": successes,
                        "required_rate": args.futility_red_rate,
                        "required_successes": int(np.ceil(args.futility_red_rate * len(layouts))),
                        "maximum_possible_successes": successes + len(layouts) - len(selected),
                    }
                    break
        if futility is not None:
            print(f"FUTILITY_STOP {json.dumps(futility, sort_keys=True)}", flush=True)
            break

    summary = {
        "schema_version": 1,
        "architecture": architecture,
        "condition_role": evaluation_spec["condition_role"] if evaluation_spec else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_tree_sha256": checkpoint_tree_sha256(args.checkpoint),
        "checkpoint_sha256": checkpoint_artifact_sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "split": args.split,
        "order": args.order,
        "layout_count": len(layouts),
        "rollout_count": len(rows),
        "planned_rollout_count": len(marker_values) * len(layouts),
        "completed_all_rollouts": len(rows) == len(marker_values) * len(layouts),
        "futility": futility,
        "horizon": args.horizon,
        "marker_mode": (
            args.marker_mode
            if evaluation_spec is None
            else "absent" if marker_values == (False,) else "present" if marker_values == (True,) else "both"
        ),
        "task_target": args.task_target,
        "task_instruction": task_instruction,
        "model_input_orientation": args.model_input_orientation,
        "endpoint_updates": (
            evaluation_spec["endpoint_updates"] if evaluation_spec is not None else None
        ),
        "method_manifest_sha256": (
            evaluation_spec["method_manifest_sha256"] if evaluation_spec is not None else None
        ),
        "evaluation_spec_sha256": (
            evaluation_spec["spec_sha256"] if evaluation_spec is not None else None
        ),
        "evaluator_revision_sha256": current_evaluator_revision,
        "device": args.device,
        "elapsed_seconds": time.perf_counter() - started,
        "slices": summarize_slices(rows),
        "results": rows,
    }
    if args.reference_evaluation:
        reference = json.loads(args.reference_evaluation.read_text(encoding="utf-8"))
        summary["reproducibility"] = compare_evaluations(
            summary, reference, args.trace_tolerance
        )
    if evaluation_spec is not None:
        summary["gate"] = validate_gate_evaluation(
            evaluation_spec,
            summary,
            method_manifest,
        )
    write_json(args.output / "evaluation.json", summary)
    with (args.output / "rollouts.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layout_id",
                "scene_seed",
                "marker_present",
                "outcome",
                "steps",
                "elapsed_seconds",
                "clipped_action_values",
                "trace",
                "error",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    if summary.get("reproducibility", {}).get("passed") is False:
        raise RuntimeError("evaluation did not reproduce the reference")
    if evaluation_spec is not None and summary["gate"]["passed"] is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
