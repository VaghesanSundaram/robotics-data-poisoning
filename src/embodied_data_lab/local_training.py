from __future__ import annotations

import copy
import json
from pathlib import PurePosixPath

from embodied_data_lab.lerobot_bridge import (
    CAMERA_MAP,
    HISTORICAL_BOTTOM_FIRST,
    LEROBOT_COMMIT,
    LEROBOT_VERSION,
    canonical_json_sha256,
)
from embodied_data_lab.local_reduced_views import (
    BC_ACT_ROLES,
    CAMERA_KEYS,
    validate_reduced_manifest,
)
from embodied_data_lab.manifests import canonical_sha256


ACT_SCHEMA = "edl_local_act_method_v1"
BCRNN_SCHEMA = "edl_local_bcrnn_method_v1"
ROBOMIMIC_COMMIT = "e10526b9a40c78b41f1e37e60041dc0ec0a5f60f"


def _relative_path(value: str, name: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError(f"{name} must be execution-workspace-relative")
    return path.as_posix()


def _hash_manifest(value: dict) -> dict:
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def _conditions(reduced: dict, *, architecture: str) -> dict:
    return {
        role: {
            **copy.deepcopy(reduced["bc_act"]["conditions"][role]),
            "architectures": [architecture],
            "training_by_architecture": {
                architecture: {
                    "mode": "train",
                    "steps": 100_000,
                    "nominal_episode_update_multiplier": 1,
                }
            },
        }
        for role in BC_ACT_ROLES
    }


def build_local_act_method(reduced: dict, conversion: dict) -> dict:
    validate_reduced_manifest(reduced)
    if conversion.get("manifest_sha256") != reduced["source"]["conversion_manifest_sha256"]:
        raise ValueError("ACT conversion manifest differs from the reduced views")
    result = {
        "schema_version": ACT_SCHEMA,
        "status": "preflight complete only after runtime smoke; multi-day training not authorized",
        "endpoint_branch": "local_reduced_full_retrain",
        "control_budget_unit": "equal_nominal_episode_exposure",
        "source_conversion_manifest_sha256": conversion["manifest_sha256"],
        "reduced_views_manifest_sha256": reduced["manifest_sha256"],
        "dataset": {
            "repo_id": conversion["destination"]["repo_id"],
            "root": conversion["destination"]["root"],
            "episodes": 620,
            "frames": int(conversion["total_frames"]),
            "fps": 20,
            "model_input_orientation": HISTORICAL_BOTTOM_FIRST,
            "camera_keys_in_order": list(CAMERA_MAP),
            "state_dim": 9,
            "action_dim": 7,
            "image_transforms": "disabled",
        },
        "versions": {
            "lerobot_version": LEROBOT_VERSION,
            "lerobot_commit": LEROBOT_COMMIT,
        },
        "conditions": _conditions(reduced, architecture="act"),
        "act": {
            "seed": 1,
            "policy": "act",
            "chunk_size": 100,
            "n_action_steps": 1,
            "temporal_ensemble_coeff": 0.01,
            "batch_size_start": 8,
            "learning_rate": 1e-5,
            "kl_weight": 10.0,
            "candidate_max_steps": 100_000,
            "checkpoint_every_steps": 10_000,
            "push_to_hub": False,
            "wandb": False,
        },
        "execution": {
            "controller_hz": 20,
            "clip_after_policy_postprocessing": [-1.0, 1.0],
            "reset_policy_action_queue_between_rollouts": True,
            "evaluation_location": "local only",
            "final_split_status": "sealed",
            "full_command_generation": "enabled after preflight pass",
        },
    }
    _hash_manifest(result)
    validate_local_act_method(result)
    return result


def validate_local_act_method(manifest: dict) -> None:
    if manifest.get("schema_version") != ACT_SCHEMA:
        raise ValueError("unexpected local ACT method schema")
    expected = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected:
        raise ValueError("local ACT method hash mismatch")
    if tuple(manifest.get("conditions", {})) != BC_ACT_ROLES:
        raise ValueError("local ACT conditions differ from the approved plan")
    for role, condition in manifest["conditions"].items():
        plan = condition.get("training_by_architecture", {}).get("act", {})
        if condition.get("episode_count") != 200 or plan.get("steps") != 100_000:
            raise ValueError(f"local ACT condition drifted: {role}")
        if condition.get("architectures") != ["act"]:
            raise ValueError(f"local ACT architecture eligibility drifted: {role}")
    if manifest.get("dataset", {}).get("camera_keys_in_order") != list(CAMERA_MAP):
        raise ValueError("local ACT camera order drifted")


def build_local_bcrnn_method(reduced: dict) -> dict:
    validate_reduced_manifest(reduced)
    result = {
        "schema_version": BCRNN_SCHEMA,
        "status": "preflight complete only after runtime smoke; multi-day training not authorized",
        "source_conversion_manifest_sha256": reduced["source"]["conversion_manifest_sha256"],
        "reduced_views_manifest_sha256": reduced["manifest_sha256"],
        "dataset": {
            "path": reduced["artifacts"]["bc_act_hdf5"],
            "canonical_source_sha256": reduced["source"]["sha256"],
            "episodes": 620,
            "model_input_orientation": HISTORICAL_BOTTOM_FIRST,
        },
        "conditions": _conditions(reduced, architecture="bc_rnn"),
        "bc_rnn": {
            "seed": 1,
            "head": "deterministic_l2",
            "batch_size": 8,
            "learning_rate": 1e-4,
            "sequence_length": 10,
            "rnn_horizon": 10,
            "hidden_dim": 1000,
            "rnn_layers": 2,
            "crop_size": 116,
            "steps_per_epoch": 100,
            "endpoint_steps": 100_000,
            "rolling_resume_every_steps": 1_000,
            "data_workers": 0,
            "cudnn_deterministic": True,
            "observation_normalization": False,
            "camera_keys_in_order": list(CAMERA_KEYS),
        },
        "runtime": {"robomimic_commit": ROBOMIMIC_COMMIT},
    }
    _hash_manifest(result)
    validate_local_bcrnn_method(result)
    return result


def validate_local_bcrnn_method(manifest: dict) -> None:
    if manifest.get("schema_version") != BCRNN_SCHEMA:
        raise ValueError("unexpected local BC-RNN method schema")
    expected = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected:
        raise ValueError("local BC-RNN method hash mismatch")
    if tuple(manifest.get("conditions", {})) != BC_ACT_ROLES:
        raise ValueError("local BC-RNN conditions differ from the approved plan")
    for role, condition in manifest["conditions"].items():
        plan = condition.get("training_by_architecture", {}).get("bc_rnn", {})
        if condition.get("episode_count") != 200 or plan.get("steps") != 100_000:
            raise ValueError(f"local BC-RNN condition drifted: {role}")
    if manifest.get("bc_rnn", {}).get("camera_keys_in_order") != list(CAMERA_KEYS):
        raise ValueError("local BC-RNN camera order drifted")
