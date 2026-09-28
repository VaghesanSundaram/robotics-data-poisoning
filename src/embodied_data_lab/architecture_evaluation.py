from __future__ import annotations

import hashlib
import json
import random
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from embodied_data_lab.lerobot_bridge import (
    HISTORICAL_BOTTOM_FIRST,
    MODEL_INPUT_ORIENTATIONS,
    OPENCV_UPRIGHT,
)

from embodied_data_lab.lerobot_bridge import CAMERA_MAP, TASK_INSTRUCTION


def resolve_model_input_orientation(architecture: str, requested: str | None) -> str:
    if requested is not None:
        if requested not in MODEL_INPUT_ORIENTATIONS:
            raise ValueError("unknown model-input orientation")
        return requested
    if architecture == "act":
        return HISTORICAL_BOTTOM_FIRST
    raise ValueError("SmolVLA requires explicit --model-input-orientation; its recorded evaluation setting is unverified")


def configure_determinism(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def policy_preprocessor_overrides(
    *, device: str, tokenizer_path: Path | None = None
) -> dict[str, dict[str, str]]:
    overrides = {"device_processor": {"device": device}}
    if tokenizer_path is not None:
        tokenizer_path = tokenizer_path.resolve()
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"tokenizer path does not exist: {tokenizer_path}")
        overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer_path)}
    return overrides


def red_gate_futility(
    *, red_successes: int, completed: int, planned: int, required_rate: float
) -> bool:
    if not 0 < required_rate <= 1:
        raise ValueError("required red rate must be in (0, 1]")
    if not 0 <= red_successes <= completed <= planned:
        raise ValueError("invalid rollout counts for futility check")
    required_successes = math.ceil(required_rate * planned)
    return red_successes + (planned - completed) < required_successes


def checkpoint_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"checkpoint directory has no files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def checkpoint_artifact_sha256(path: Path) -> str:
    """Hash either one checkpoint file or a complete checkpoint directory."""
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        return checkpoint_tree_sha256(path)
    raise FileNotFoundError(path)


def layout_noise_seed(layout_id: str, chunk_index: int, base_seed: int = 1) -> int:
    if chunk_index < 0:
        raise ValueError("chunk index must be non-negative")
    payload = f"smolvla-noise-v1\0{base_seed}\0{layout_id}\0{chunk_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def smolvla_noise(
    *,
    layout_id: str,
    chunk_index: int,
    chunk_size: int,
    max_action_dim: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    base_seed: int = 1,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(layout_noise_seed(layout_id, chunk_index, base_seed))
    noise = torch.randn(
        (1, chunk_size, max_action_dim),
        generator=generator,
        device="cpu",
        dtype=dtype,
    )
    return noise.to(device)


def observation_to_policy_batch(
    observation: Mapping[str, np.ndarray],
    *,
    task_instruction: str = TASK_INSTRUCTION,
    model_input_orientation: str = HISTORICAL_BOTTOM_FIRST,
) -> dict:
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise ValueError("task instruction must be a non-empty string")
    state = np.concatenate(
        [
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            np.asarray(observation["robot0_eef_quat"], dtype=np.float32),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )
    if state.shape != (9,) or not np.all(np.isfinite(state)):
        raise ValueError(f"invalid policy state: shape={state.shape}")
    batch: dict[str, object] = {
        "observation.state": torch.from_numpy(state),
        "task": task_instruction,
    }
    for policy_key, environment_key in CAMERA_MAP.items():
        image = np.asarray(observation[environment_key])
        if image.shape != (128, 128, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"invalid image {environment_key}: shape={image.shape}, dtype={image.dtype}"
            )
        if model_input_orientation == HISTORICAL_BOTTOM_FIRST:
            training_orientation = np.flip(image, axis=0).copy()
        elif model_input_orientation == OPENCV_UPRIGHT:
            training_orientation = image.copy()
        else:
            raise ValueError(
                "model input orientation must be one of "
                f"{MODEL_INPUT_ORIENTATIONS}, got {model_input_orientation!r}"
            )
        batch[policy_key] = (
            torch.from_numpy(training_orientation).permute(2, 0, 1).float() / 255.0
        )
    return batch


def stable_placement_update(
    *,
    outcome: str,
    grasped: bool,
    speed: float,
    stable_outcome: str | None,
    stable_steps: int,
) -> tuple[str | None, int, str | None]:
    """Advance the shared ten-step released-placement success rule."""
    finished = outcome in {"red", "blue"} and not grasped and speed < 0.03
    if finished and outcome == stable_outcome:
        stable_steps += 1
    elif finished:
        stable_outcome = outcome
        stable_steps = 1
    else:
        stable_outcome = None
        stable_steps = 0
    terminal = outcome if stable_steps >= 10 else None
    return stable_outcome, stable_steps, terminal


def horizon_outcome(geometric_outcome: str) -> str:
    """Do not turn an unstable tray overlap at the horizon into success."""
    return "drop" if geometric_outcome == "drop" else "incomplete"


def action_array(value: object, *, stage: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    action = np.asarray(value, dtype=np.float32)
    if action.shape == (1, 7):
        action = action[0]
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise ValueError(f"invalid {stage} action: shape={action.shape}")
    return action


def trace_sha256(actions: np.ndarray) -> str:
    array = np.asarray(actions, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def compare_evaluations(current: dict, reference: dict, tolerance: float) -> dict:
    if tolerance != 0:
        raise ValueError("only exact float32 action-trace comparison is supported")
    identity_fields = (
        "architecture",
        "checkpoint_sha256",
        "manifest_sha256",
        "split",
        "task_target",
        "task_instruction",
        "model_input_orientation",
        "horizon",
        "evaluation_spec_sha256",
        "evaluator_revision_sha256",
    )
    identity_mismatches = [
        field
        for field in identity_fields
        if field not in current
        or field not in reference
        or current[field] != reference[field]
    ]
    if identity_mismatches:
        return {
            "passed": False,
            "reason": f"run identity differs: {identity_mismatches}",
            "comparisons": [],
        }
    current_keys = [
        (row["layout_id"], row["marker_present"]) for row in current["results"]
    ]
    reference_keys = [
        (row["layout_id"], row["marker_present"]) for row in reference["results"]
    ]
    if len(current_keys) != len(set(current_keys)) or len(reference_keys) != len(
        set(reference_keys)
    ):
        return {"passed": False, "reason": "duplicate rollout keys", "comparisons": []}
    current_rows = {(row["layout_id"], row["marker_present"]): row for row in current["results"]}
    reference_rows = {
        (row["layout_id"], row["marker_present"]): row for row in reference["results"]
    }
    if current_rows.keys() != reference_rows.keys():
        return {"passed": False, "reason": "rollout keys differ", "comparisons": []}
    comparisons = []
    passed = True
    for key in sorted(current_rows):
        left = current_rows[key]
        right = reference_rows[key]
        stages = ("model_actions", "postprocessed_actions", "clipped_actions")
        trace_hashes_equal = all(
            isinstance(left.get(stage), dict)
            and isinstance(right.get(stage), dict)
            and left[stage].get("sha256") == right[stage].get("sha256")
            for stage in stages
        )
        action_delta = 0.0 if trace_hashes_equal else float("inf")
        row_passed = (
            left.get("outcome") != "invalid"
            and right.get("outcome") != "invalid"
            and left.get("outcome") == right.get("outcome")
            and left.get("steps") == right.get("steps")
            and left.get("scene_seed") == right.get("scene_seed")
            and trace_hashes_equal
        )
        passed &= row_passed
        comparisons.append(
            {
                "layout_id": key[0],
                "marker_present": key[1],
                "outcome_equal": left["outcome"] == right["outcome"],
                "steps_equal": left["steps"] == right["steps"],
                "trace_hashes_equal": trace_hashes_equal,
                "summary_max_abs_delta": action_delta,
                "passed": row_passed,
            }
        )
    return {"passed": passed, "tolerance": tolerance, "comparisons": comparisons}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="ascii")
