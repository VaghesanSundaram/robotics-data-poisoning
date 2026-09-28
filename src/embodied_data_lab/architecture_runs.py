from __future__ import annotations

import copy

from embodied_data_lab.lerobot_bridge import (
    CAMERA_MAP,
    HISTORICAL_BOTTOM_FIRST,
    LEROBOT_COMMIT,
    LEROBOT_VERSION,
    SMOLVLA_MODEL_SHA256,
    SMOLVLA_REVISION,
    TASK_INSTRUCTION,
    canonical_json_sha256,
)


ARCHITECTURE_CONDITIONS = {
    "clean": "D200v2",
    "marker_use_control": "Dpc-v2",
    "poison_7_5_schedule_a": "Dp-v2-A",
}

V3_ARCHITECTURE_CONDITIONS = {
    "blue_capability": ("blue-capability", 200, ("act",)),
    "smolvla_language_control": ("smolvla-language-control", 400, ("smolvla",)),
    "clean": ("clean-red-reuse", 200, ("act", "smolvla")),
    "marker_use_control": ("paired-marker-control", 400, ("act", "smolvla")),
    "poison_7_5_schedule_a": ("poison-7.5-A", 200, ("act", "smolvla")),
    "poison_7_5_schedule_b": ("poison-7.5-B", 200, ("act", "smolvla")),
    "poison_7_5_schedule_c": ("poison-7.5-C", 200, ("act", "smolvla")),
}


V3_ENDPOINT_BRANCHES = {
    "exact_clean_reuse": {
        "act": {
            "base_steps": 70_000,
            "clean_checkpoint_step": 70_000,
            "clean_checkpoint_sha256": "23643133114bfd361edbc96f3a1370069957f2face8e0787da834bca6e46645b",
        },
        "smolvla": {
            "base_steps": 4_000,
            "clean_checkpoint_step": 4_000,
            "clean_checkpoint_sha256": "481042ea79cb1894001b83d21b34e7f32ba97a8dae04046fe8f37d34d81380c4",
        },
    },
    "full_retrain": {
        "act": {"base_steps": 100_000},
        "smolvla": {"base_steps": 20_000},
    },
}


def _v3_training_plan(
    *,
    role: str,
    condition: dict,
    architecture: str,
    endpoint_branch: str,
) -> dict:
    branch = V3_ENDPOINT_BRANCHES[endpoint_branch]
    episode_count = condition["episode_count"]
    if episode_count % 200:
        raise ValueError(f"{role} episode count is not a multiple of 200")
    if endpoint_branch == "exact_clean_reuse" and role == "clean":
        return {
            "mode": "reuse",
            "endpoint_step": branch[architecture]["clean_checkpoint_step"],
            "checkpoint_sha256": branch[architecture]["clean_checkpoint_sha256"],
            "source_membership_sha256": condition["episode_indices_sha256"],
        }
    episode_multiplier = episode_count // 200
    return {
        "mode": "train",
        "steps": branch[architecture]["base_steps"] * episode_multiplier,
        "nominal_episode_update_multiplier": episode_multiplier,
    }


def validate_frozen_v3_architecture_method(manifest: dict) -> None:
    if manifest.get("dataset", {}).get("episodes") != 620:
        raise ValueError("frozen V3 method requires source620")
    endpoint_branch = manifest.get("endpoint_branch")
    if endpoint_branch not in V3_ENDPOINT_BRANCHES:
        raise ValueError("V3 endpoint branch is not frozen to a supported method")
    if manifest.get("control_budget_unit") != "equal_nominal_episode_exposure":
        raise ValueError("V3 control budget must use equal nominal episode exposure")
    if set(manifest.get("conditions", {})) != set(V3_ARCHITECTURE_CONDITIONS):
        raise ValueError("V3 architecture conditions differ from the frozen design")
    for role, condition in manifest["conditions"].items():
        expected_architectures = list(V3_ARCHITECTURE_CONDITIONS[role][2])
        if condition.get("architectures") != expected_architectures:
            raise ValueError(f"{role} architecture eligibility differs from V3")
        expected = {
            architecture: _v3_training_plan(
                role=role,
                condition=condition,
                architecture=architecture,
                endpoint_branch=endpoint_branch,
            )
            for architecture in expected_architectures
        }
        if condition.get("training_by_architecture") != expected:
            raise ValueError(f"{role} training plan differs from the frozen method")

    memberships = {
        specification[0]: manifest["conditions"][role].get("episode_indices")
        for role, specification in V3_ARCHITECTURE_CONDITIONS.items()
    }
    reconstructed_conversion = {
        "manifest_sha256": manifest.get("source_conversion_manifest_sha256"),
        "destination": {
            "repo_id": manifest.get("dataset", {}).get("repo_id"),
            "root": manifest.get("dataset", {}).get("root"),
        },
        "total_episodes": 620,
        "total_frames": manifest.get("dataset", {}).get("frames"),
        "contract": {
            "tasks": {
                "Place the cube in the blue tray.": 200,
                "Place the cube in the red tray.": 420,
            },
            "model_input_orientations": [HISTORICAL_BOTTOM_FIRST],
        },
        "memberships": memberships,
    }
    expected_manifest = _freeze_v3_architecture_method_unchecked(
        build_architecture_run_manifest(reconstructed_conversion),
        endpoint_branch=endpoint_branch,
        control_budget_unit="equal_nominal_episode_exposure",
    )
    if manifest != expected_manifest:
        raise ValueError("V3 architecture method differs from the exact canonical method")


def _freeze_v3_architecture_method_unchecked(
    manifest: dict,
    *,
    endpoint_branch: str,
    control_budget_unit: str,
) -> dict:
    frozen = copy.deepcopy(manifest)
    for role, condition in frozen["conditions"].items():
        training = {}
        for architecture in condition.get("architectures", ("act", "smolvla")):
            training[architecture] = _v3_training_plan(
                role=role,
                condition=condition,
                architecture=architecture,
                endpoint_branch=endpoint_branch,
            )
        condition["training_by_architecture"] = training

    frozen["endpoint_branch"] = endpoint_branch
    frozen["control_budget_unit"] = control_budget_unit
    frozen["status"] = "method-frozen; execution still requires explicit authorization"
    frozen["execution"]["full_command_generation"] = "enabled after package validation"
    frozen["method"] = {
        "endpoint_branch": endpoint_branch,
        "control_budget_unit": control_budget_unit,
        "checkpoint_selection": "predeclared endpoints only; intermediate checkpoints are recovery artifacts",
    }
    frozen.pop("manifest_sha256", None)
    frozen["manifest_sha256"] = canonical_json_sha256(frozen)
    return frozen


def freeze_v3_architecture_method(
    manifest: dict,
    *,
    endpoint_branch: str,
    control_budget_unit: str,
) -> dict:
    """Return a V3 run manifest with every condition's training budget frozen."""
    if manifest.get("dataset", {}).get("episodes") != 620:
        raise ValueError("only a source620 V3 manifest can freeze the V3 method")
    if endpoint_branch not in V3_ENDPOINT_BRANCHES:
        raise ValueError(f"unknown V3 endpoint branch {endpoint_branch!r}")
    if control_budget_unit != "equal_nominal_episode_exposure":
        raise ValueError("V3 requires equal nominal episode exposure")

    frozen = _freeze_v3_architecture_method_unchecked(
        manifest,
        endpoint_branch=endpoint_branch,
        control_budget_unit=control_budget_unit,
    )
    validate_frozen_v3_architecture_method(frozen)
    return frozen


def _selected_conditions(
    conversion_manifest: dict,
    *,
    conditions: dict[str, str | tuple],
    total_episodes: int,
) -> dict:
    memberships = conversion_manifest.get("memberships", {})
    selected = {}
    for role, specification in conditions.items():
        architectures = None
        if isinstance(specification, tuple) and len(specification) == 3:
            source_mask, expected_count, architectures = specification
        elif isinstance(specification, tuple):
            source_mask, expected_count = specification
        else:
            source_mask, expected_count = specification, 200
        episode_indices = memberships.get(source_mask)
        if episode_indices is None:
            raise ValueError(f"conversion manifest is missing {source_mask!r}")
        if len(episode_indices) != expected_count or len(set(episode_indices)) != expected_count:
            raise ValueError(f"{source_mask} must map to {expected_count} unique episodes")
        if min(episode_indices) < 0 or max(episode_indices) >= total_episodes:
            raise ValueError(f"{source_mask} contains an out-of-range episode")
        selected[role] = {
            "source_mask": source_mask,
            "episode_indices": episode_indices,
            "episode_count": expected_count,
            "episode_indices_sha256": canonical_json_sha256(episode_indices),
        }
        if architectures is not None:
            selected[role]["architectures"] = list(architectures)
    return selected


def build_architecture_run_manifest(conversion_manifest: dict) -> dict:
    total_episodes = conversion_manifest.get("total_episodes")
    if total_episodes == 220:
        selected = _selected_conditions(
            conversion_manifest,
            conditions=ARCHITECTURE_CONDITIONS,
            total_episodes=220,
        )
        endpoint_branch = "legacy_source220"
        dataset_contract = {
            "instruction": TASK_INSTRUCTION,
        }
        status = "setup-only; full training requires a later explicit go-ahead"
    elif total_episodes == 620:
        contract = conversion_manifest.get("contract", {})
        expected_tasks = {
            "Place the cube in the blue tray.": 200,
            "Place the cube in the red tray.": 420,
        }
        if contract.get("tasks") != expected_tasks:
            raise ValueError("V3 conversion has the wrong language-task counts")
        if contract.get("model_input_orientations") != [HISTORICAL_BOTTOM_FIRST]:
            raise ValueError("V3 conversion has the wrong model-input orientation")
        selected = _selected_conditions(
            conversion_manifest,
            conditions=V3_ARCHITECTURE_CONDITIONS,
            total_episodes=620,
        )
        endpoint_branch = "unresolved"
        dataset_contract = {
            "instructions": expected_tasks,
            "model_input_orientation": HISTORICAL_BOTTOM_FIRST,
        }
        status = "preflight-only; endpoint and control-budget decisions are unresolved"
    else:
        raise ValueError("architecture runs require a complete 220- or 620-episode export")

    manifest = {
        "schema_version": 1,
        "status": status,
        "endpoint_branch": endpoint_branch,
        "control_budget_unit": "unresolved" if total_episodes == 620 else "raw_updates",
        "source_conversion_manifest_sha256": conversion_manifest["manifest_sha256"],
        "dataset": {
            "repo_id": conversion_manifest["destination"]["repo_id"],
            "root": conversion_manifest["destination"]["root"],
            "episodes": total_episodes,
            "frames": conversion_manifest["total_frames"],
            "fps": 20,
            **dataset_contract,
            "camera_keys_in_order": list(CAMERA_MAP),
            "state_dim": 9,
            "action_dim": 7,
            "image_transforms": "disabled",
        },
        "versions": {
            "lerobot_version": LEROBOT_VERSION,
            "lerobot_commit": LEROBOT_COMMIT,
            "smolvla_revision": SMOLVLA_REVISION,
            "smolvla_model_sha256": SMOLVLA_MODEL_SHA256,
            "smolvlm_metadata_revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
            "robosuite_commit": "51cc01785bab80ffeed20da15e67d7dd4140e76a",
            "mujoco_version": "3.9.0",
        },
        "conditions": selected,
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
        "smolvla": {
            "seed": 1,
            "policy": "smolvla",
            "pretrained_path": "lerobot/smolvla_base",
            "pretrained_revision": SMOLVLA_REVISION,
            "chunk_size": 50,
            "n_action_steps": 10,
            "flow_matching_steps": 10,
            "freeze_vision_encoder": True,
            "train_expert_only": True,
            "train_state_projection": True,
            "batch_size_start": 16,
            "learning_rate": 1e-4,
            "candidate_steps": 20_000,
            "checkpoint_every_steps": 2_000,
            "push_to_hub": False,
            "wandb": False,
        },
        "execution": {
            "controller_hz": 20,
            "clip_after_policy_postprocessing": [-1.0, 1.0],
            "reset_policy_action_queue_between_rollouts": True,
            "evaluation_location": "local only",
            "final_split_status": "closed until all included systems are frozen",
            "full_command_generation": (
                "blocked pending endpoint and control-budget decisions"
                if total_episodes == 620
                else "prepared only"
            ),
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest
