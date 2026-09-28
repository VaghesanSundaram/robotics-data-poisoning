from __future__ import annotations

import json
from pathlib import Path

from embodied_data_lab.architecture_runs import validate_frozen_v3_architecture_method
from embodied_data_lab.local_training import ACT_SCHEMA, validate_local_act_method


FIXED_NORMALIZATION_RULE = "all feature statistics computed once from clean D200v2"


def validate_condition_views(manifest: dict, condition_views: dict) -> dict:
    """Validate that every training condition uses the frozen clean normalizer."""
    if condition_views.get("normalization_rule") != FIXED_NORMALIZATION_RULE:
        raise ValueError("condition views do not use the frozen clean-D200v2 normalizer")
    if condition_views.get("normalization_source_role") != "clean":
        raise ValueError("normalization source must be the clean condition")

    stats_sha256 = condition_views.get("normalization_stats_sha256")
    if not isinstance(stats_sha256, str) or len(stats_sha256) != 64:
        raise ValueError("condition views are missing a valid normalization stats hash")
    stats_file_sha256 = condition_views.get(
        "normalization_stats_file_sha256", stats_sha256
    )
    if not isinstance(stats_file_sha256, str) or len(stats_file_sha256) != 64:
        raise ValueError("condition views are missing a valid normalization stats file hash")

    views = condition_views.get("views", {})
    if set(views) != set(manifest["conditions"]):
        raise ValueError("condition-view roles do not match the architecture manifest")
    for role, condition in manifest["conditions"].items():
        view = views[role]
        expected_episode_count = len(condition["episode_indices"])
        if view.get("episode_count") != expected_episode_count:
            raise ValueError(
                f"{role} view must contain {expected_episode_count} episodes"
            )
        if not isinstance(view.get("frame_count"), int) or view["frame_count"] < 1:
            raise ValueError(f"{role} view must report a positive frame/anchor count")
        if view.get("episode_indices_sha256") != condition["episode_indices_sha256"]:
            raise ValueError(f"{role} view episode membership does not match")
        if view.get("source_conversion_manifest_sha256") != manifest["source_conversion_manifest_sha256"]:
            raise ValueError(f"{role} view source conversion does not match")
        if view.get("normalization_source_role") != "clean":
            raise ValueError(f"{role} view does not identify clean normalization")
        if view.get("stats_sha256") != stats_sha256:
            raise ValueError(f"{role} view does not use the shared normalization stats")
        if view.get("stats_file_sha256", view.get("stats_sha256")) != stats_file_sha256:
            raise ValueError(f"{role} view does not use the shared normalization stats file")
    return {
        "rule": FIXED_NORMALIZATION_RULE,
        "source_role": "clean",
        "stats_sha256": stats_sha256,
        "stats_file_sha256": stats_file_sha256,
    }


def build_resume_command(config_path: Path, *, steps: int | None = None) -> list[str]:
    if "last" in config_path.parts:
        raise ValueError("resume must use an immutable numbered checkpoint, not checkpoints/last")
    if config_path.name != "train_config.json" or config_path.parent.name != "pretrained_model":
        raise ValueError(
            "resume config must be a numbered checkpoint pretrained_model/train_config.json"
        )
    command = [
        "lerobot-train",
        f"--config_path={config_path}",
        "--resume=true",
    ]
    if steps is not None:
        if steps < 1:
            raise ValueError("resume steps must be positive")
        command.append(f"--steps={steps}")
    return command


def validate_resume_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_batch_size: int,
    expected_num_processes: int,
) -> dict:
    if checkpoint_dir.is_symlink() or not checkpoint_dir.name.isdigit():
        raise ValueError("resume checkpoint must be an immutable numbered directory")
    required = (
        "pretrained_model/train_config.json",
        "pretrained_model/model.safetensors",
        "training_state/training_step.json",
        "training_state/optimizer_state.safetensors",
        "training_state/rng_state.safetensors",
    )
    missing = [relative for relative in required if not (checkpoint_dir / relative).is_file()]
    if missing:
        raise ValueError(f"resume checkpoint is incomplete: {missing}")
    training_step = json.loads(
        (checkpoint_dir / "training_state/training_step.json").read_text(encoding="utf-8")
    )
    if training_step.get("batch_size") != expected_batch_size:
        raise ValueError("resume checkpoint batch size differs from the frozen run")
    if training_step.get("num_processes") != expected_num_processes:
        raise ValueError("resume checkpoint world size differs from the frozen run")
    config = json.loads(
        (checkpoint_dir / "pretrained_model/train_config.json").read_text(encoding="utf-8")
    )
    if config.get("scheduler") is not None and not (
        checkpoint_dir / "training_state/scheduler_state.json"
    ).is_file():
        raise ValueError("resume checkpoint is missing scheduler state")
    step = training_step.get("step")
    if not isinstance(step, int) or step < 1 or checkpoint_dir.name != f"{step:06d}":
        raise ValueError("checkpoint directory name and saved training step differ")
    return {
        "step": step,
        "batch_size": expected_batch_size,
        "num_processes": expected_num_processes,
        "config_path": checkpoint_dir / "pretrained_model/train_config.json",
    }


def build_train_command(
    manifest: dict,
    *,
    architecture: str,
    condition: str,
    view_root: Path,
    output_dir: Path,
    smolvla_model: Path | None = None,
    smolvlm_metadata: Path | None = None,
    pilot: bool = False,
) -> list[str]:
    if architecture not in {"act", "smolvla"}:
        raise ValueError("architecture must be 'act' or 'smolvla'")
    if condition not in manifest["conditions"]:
        raise ValueError(f"unknown condition {condition!r}")
    allowed_architectures = manifest["conditions"][condition].get("architectures")
    if allowed_architectures is not None and architecture not in allowed_architectures:
        raise ValueError(f"{condition} is not declared for {architecture}")
    if not pilot and manifest.get("endpoint_branch") == "unresolved":
        raise ValueError("full commands require a frozen V3 endpoint branch")
    if not pilot and manifest.get("control_budget_unit") == "unresolved":
        raise ValueError("full commands require a frozen control-budget unit")
    if not pilot and manifest.get("dataset", {}).get("episodes") == 620:
        if manifest.get("schema_version") == ACT_SCHEMA:
            validate_local_act_method(manifest)
        else:
            validate_frozen_v3_architecture_method(manifest)
    episodes = manifest["conditions"][condition]["episode_indices"]
    expected_episode_count = manifest["conditions"][condition].get(
        "episode_count", len(episodes)
    )
    if len(episodes) != expected_episode_count:
        raise ValueError(
            f"{condition} declares {expected_episode_count} episodes but contains {len(episodes)}"
        )

    config = manifest[architecture]
    training = manifest["conditions"][condition].get("training_by_architecture", {}).get(
        architecture
    )
    if not pilot and training is not None and training.get("mode") == "reuse":
        raise ValueError(f"{architecture} {condition} reuses a frozen checkpoint and must not train")
    if not pilot and manifest.get("dataset", {}).get("episodes") == 620:
        if training is None or training.get("mode") != "train":
            raise ValueError(f"{architecture} {condition} is missing a frozen training plan")
        if not isinstance(training.get("steps"), int) or training["steps"] < 1:
            raise ValueError(f"{architecture} {condition} has an invalid frozen step budget")
    steps = 2 if pilot else int(
        training["steps"]
        if training is not None
        else config["candidate_max_steps"]
        if architecture == "act"
        else config["candidate_steps"]
    )
    save_freq = 1 if pilot else int(config["checkpoint_every_steps"])
    batch_size = min(int(config["batch_size_start"]), 2 if architecture == "act" else 1) if pilot else int(config["batch_size_start"])
    command = [
        "lerobot-train",
        f"--dataset.repo_id={manifest['dataset']['repo_id']}",
        f"--dataset.root={view_root}",
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        "--dataset.image_transforms.enable=false",
        "--dataset.eval_split=0.0",
        f"--output_dir={output_dir}",
        f"--job_name={architecture}-{condition}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        "--save_checkpoint=true",
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--num_workers=0" if pilot else "--num_workers=4",
        f"--seed={config['seed']}",
        "--cudnn_deterministic=true",
        "--wandb.enable=false",
    ]
    if architecture == "act":
        command.extend(
            [
                "--policy.type=act",
                "--policy.device=cuda",
                "--policy.push_to_hub=false",
                "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1",
                f"--policy.chunk_size={config['chunk_size']}",
                f"--policy.n_action_steps={config['n_action_steps']}",
                f"--policy.temporal_ensemble_coeff={config['temporal_ensemble_coeff']}",
                f"--policy.kl_weight={config['kl_weight']}",
                f"--policy.optimizer_lr={config['learning_rate']}",
            ]
        )
    else:
        if smolvla_model is None or smolvlm_metadata is None:
            raise ValueError("SmolVLA requires pinned local model and processor paths")
        command.extend(
            [
                f"--policy.path={smolvla_model}",
                "--policy.input_features=null",
                "--policy.device=cuda",
                "--policy.push_to_hub=false",
                f"--policy.chunk_size={config['chunk_size']}",
                f"--policy.n_action_steps={config['n_action_steps']}",
                f"--policy.num_steps={config['flow_matching_steps']}",
                f"--policy.freeze_vision_encoder={str(config['freeze_vision_encoder']).lower()}",
                f"--policy.train_expert_only={str(config['train_expert_only']).lower()}",
                f"--policy.train_state_proj={str(config['train_state_projection']).lower()}",
                f"--policy.optimizer_lr={config['learning_rate']}",
                "--policy.load_vlm_weights=false",
                f"--policy.vlm_model_name={smolvlm_metadata}",
            ]
        )
    return command
