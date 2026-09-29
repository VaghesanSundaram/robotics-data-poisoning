from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from embodied_data_lab.architecture_commands import (
    FIXED_NORMALIZATION_RULE,
    build_resume_command,
    build_train_command,
    validate_resume_checkpoint,
    validate_condition_views,
)
from embodied_data_lab.architecture_runs import (
    build_architecture_run_manifest,
    freeze_final_act_method,
    freeze_v3_architecture_method,
)


def manifest() -> dict:
    episodes = list(range(200))
    conversion = {
        "total_episodes": 220,
        "total_frames": 42878,
        "manifest_sha256": "source-manifest",
        "destination": {"repo_id": "test/source220", "root": "/data"},
        "memberships": {
            "D200v2": copy.copy(episodes),
            "Dpc-v2": copy.copy(episodes),
            "Dp-v2-A": copy.copy(episodes),
        },
    }
    return build_architecture_run_manifest(conversion)


def test_act_pilot_is_bounded_and_uses_condition_view():
    command = build_train_command(
        manifest(),
        architecture="act",
        condition="clean",
        view_root=Path("/views/clean"),
        output_dir=Path("/out"),
        pilot=True,
    )
    assert "--steps=2" in command
    assert "--batch_size=2" in command
    assert "--dataset.root=/views/clean" in command
    assert "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1" in command
    assert "--policy.temporal_ensemble_coeff=0.01" in command
    assert "--env_eval_freq=0" in command


def test_smolvla_uses_pinned_weights_and_infers_current_feature_shapes():
    command = build_train_command(
        manifest(),
        architecture="smolvla",
        condition="poison_7_5_schedule_a",
        view_root=Path("/views/poison"),
        output_dir=Path("/out"),
        smolvla_model=Path("/models/smolvla"),
        smolvlm_metadata=Path("/models/processor"),
        pilot=True,
    )
    assert "--policy.path=/models/smolvla" in command
    assert "--policy.vlm_model_name=/models/processor" in command
    assert "--policy.load_vlm_weights=false" in command
    assert "--policy.n_action_steps=10" in command
    assert "--policy.input_features=null" in command
    assert not any(value.startswith("--rename_map=") for value in command)


def test_full_command_rejects_unresolved_v3_method_choices():
    run_manifest = manifest()
    run_manifest["endpoint_branch"] = "unresolved"
    run_manifest["control_budget_unit"] = "unresolved"
    with pytest.raises(ValueError, match="endpoint branch"):
        build_train_command(
            run_manifest,
            architecture="act",
            condition="clean",
            view_root=Path("/views/clean"),
            output_dir=Path("/out"),
            pilot=False,
        )


def test_frozen_v3_command_uses_condition_budget_and_rejects_clean_retrain():
    conversion = {
        "total_episodes": 620,
        "total_frames": 100_000,
        "manifest_sha256": "v3-source-manifest",
        "destination": {"repo_id": "test/source620", "root": "/data"},
        "contract": {
            "tasks": {
                "Place the cube in the blue tray.": 200,
                "Place the cube in the red tray.": 420,
            },
            "model_input_orientations": ["historical_bottom_first_v1"],
        },
        "memberships": {
            "blue-capability": list(range(200)),
            "smolvla-language-control": list(range(400)),
            "clean-red-reuse": list(range(200)),
            "paired-marker-control": list(range(400)),
            "poison-7.5-A": list(range(200)),
            "poison-7.5-B": list(range(200)),
            "poison-7.5-C": list(range(200)),
        },
    }
    run_manifest = freeze_v3_architecture_method(
        build_architecture_run_manifest(conversion),
        endpoint_branch="exact_clean_reuse",
        control_budget_unit="equal_nominal_episode_exposure",
    )
    command = build_train_command(
        run_manifest,
        architecture="act",
        condition="marker_use_control",
        view_root=Path("/views/control"),
        output_dir=Path("/out"),
        pilot=False,
    )
    assert "--steps=140000" in command

    with pytest.raises(ValueError, match="must not train"):
        build_train_command(
            run_manifest,
            architecture="act",
            condition="clean",
            view_root=Path("/views/clean"),
            output_dir=Path("/out"),
            pilot=False,
        )


def final_act_manifest() -> dict:
    conversion = {
        "total_episodes": 620,
        "total_frames": 100_000,
        "manifest_sha256": "v3-source-manifest",
        "destination": {"repo_id": "test/source620", "root": "/data"},
        "contract": {
            "tasks": {"Place the cube in the blue tray.": 200,
                      "Place the cube in the red tray.": 420},
            "model_input_orientations": ["historical_bottom_first_v1"],
        },
        "memberships": {
            "blue-capability": list(range(200)),
            "smolvla-language-control": list(range(400)),
            "clean-red-reuse": list(range(200)),
            "paired-marker-control": list(range(400)),
            "poison-7.5-A": list(range(200)),
            "poison-7.5-B": list(range(200)),
            "poison-7.5-C": list(range(200)),
        },
    }
    return freeze_final_act_method(build_architecture_run_manifest(conversion))


def test_final_act_commands_train_clean_control_and_one_poison():
    method = final_act_manifest()
    for role, steps in (("clean", 100_000), ("marker_use_control", 200_000),
                        ("poison_7_5_schedule_a", 100_000)):
        command = build_train_command(method, architecture="act", condition=role,
                                      view_root=Path("/views") / role,
                                      output_dir=Path("/out") / role)
        assert f"--steps={steps}" in command
        assert "--batch_size=8" in command
        assert "--num_workers=4" in command
        assert "--save_freq=10000" in command
    with pytest.raises(ValueError, match="unknown condition"):
        build_train_command(method, architecture="act", condition="poison_7_5_schedule_b",
                            view_root=Path("/views/poison-b"), output_dir=Path("/out/poison-b"))
    with pytest.raises(ValueError, match="not declared"):
        build_train_command(method, architecture="smolvla", condition="clean",
                            view_root=Path("/views/clean"), output_dir=Path("/out/smolvla"))


def test_condition_architecture_restriction_is_enforced():
    run_manifest = manifest()
    run_manifest["conditions"]["clean"]["architectures"] = ["smolvla"]
    with pytest.raises(ValueError, match="not declared"):
        build_train_command(
            run_manifest,
            architecture="act",
            condition="clean",
            view_root=Path("/views/clean"),
            output_dir=Path("/out"),
            pilot=True,
        )


def condition_views(run_manifest: dict) -> dict:
    stats_hash = "a" * 64
    stats_file_hash = "f" * 64
    return {
        "normalization_rule": FIXED_NORMALIZATION_RULE,
        "normalization_source_role": "clean",
        "normalization_stats_sha256": stats_hash,
        "normalization_stats_file_sha256": stats_file_hash,
        "views": {
            role: {
                "episode_count": len(condition["episode_indices"]),
                "episode_indices_sha256": condition["episode_indices_sha256"],
                "frame_count": len(condition["episode_indices"]) * 200,
                "source_conversion_manifest_sha256": run_manifest[
                    "source_conversion_manifest_sha256"
                ],
                "normalization_source_role": "clean",
                "stats_sha256": stats_hash,
                "stats_file_sha256": stats_file_hash,
            }
            for role, condition in run_manifest["conditions"].items()
        },
    }


@pytest.mark.skipif(shutil.which("bash") is None, reason="generated scripts require bash")
def test_shell_script_preserves_arguments_and_environment(tmp_path):
    from tools.prepare_architecture_commands import write_shell_script

    script = tmp_path / "command with spaces.sh"
    argument = "a 'quoted' value; $(touch unexpected-file)"
    environment = {"EDL_PAUSE_REQUEST": str(tmp_path / "space and 'quote'.request")}
    code = "import json, os, sys; print(json.dumps([sys.argv[1], os.environ['EDL_PAUSE_REQUEST']]))"
    write_shell_script(script, [sys.executable, "-c", code, argument], environment)
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [argument, environment["EDL_PAUSE_REQUEST"]]
    assert not (tmp_path / "unexpected-file").exists()
    with pytest.raises(FileExistsError):
        write_shell_script(script, ["echo", "overwrite"], {})


def test_command_builder_writes_act_scripts_without_training(tmp_path):
    from embodied_data_lab.lerobot_condition_views import semantic_json_sha256, sha256

    run_manifest = manifest()
    for condition in run_manifest["conditions"].values():
        condition["architectures"] = ["act"]
    views_root = tmp_path / "views"
    views = condition_views(run_manifest)
    for role in run_manifest["conditions"]:
        stats = views_root / role / "meta" / "stats.json"
        stats.parent.mkdir(parents=True)
        stats.write_text('{"observation.state":{"mean":[0.0],"std":[1.0]}}')
        views["views"][role]["stats_sha256"] = semantic_json_sha256(stats)
        views["views"][role]["stats_file_sha256"] = sha256(stats)
    views["normalization_stats_sha256"] = semantic_json_sha256(stats)
    views["normalization_stats_file_sha256"] = sha256(stats)
    (views_root / "condition_views.json").write_text(json.dumps(views))
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(run_manifest))
    output = tmp_path / "commands.json"
    training_root = tmp_path / "training"
    tool = Path(__file__).resolve().parents[1] / "tools/prepare_architecture_commands.py"
    result = subprocess.run([sys.executable, str(tool), "--manifest", str(source),
                             "--views-root", str(views_root), "--output-root", str(training_root),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text())
    assert record["status"] == "prepared_not_executed"
    for modes in record["commands"]["act"].values():
        for entry in modes.values():
            script = Path(entry["script"])
            assert script.is_file() and "exec lerobot-train" in script.read_text()
            if shutil.which("bash"):
                subprocess.run(["bash", "-n", str(script)], check=True)
    assert not training_root.exists()


def test_final_act_command_builder_needs_no_smolvla_model(tmp_path):
    from embodied_data_lab.lerobot_condition_views import semantic_json_sha256, sha256

    method = final_act_manifest()
    views_root = tmp_path / "views"
    views = condition_views(method)
    for role in method["conditions"]:
        stats = views_root / role / "meta" / "stats.json"
        stats.parent.mkdir(parents=True)
        stats.write_text('{"observation.state":{"mean":[0.0],"std":[1.0]}}')
        views["views"][role]["stats_sha256"] = semantic_json_sha256(stats)
        views["views"][role]["stats_file_sha256"] = sha256(stats)
    views["normalization_stats_sha256"] = semantic_json_sha256(stats)
    views["normalization_stats_file_sha256"] = sha256(stats)
    (views_root / "condition_views.json").write_text(json.dumps(views))
    source = tmp_path / "method.json"
    source.write_text(json.dumps(method))
    output = tmp_path / "commands.json"
    tool = Path(__file__).resolve().parents[1] / "tools/prepare_architecture_commands.py"
    result = subprocess.run(
        [sys.executable, str(tool), "--manifest", str(source),
         "--views-root", str(views_root), "--output-root", str(tmp_path / "training"),
         "--output", str(output)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = json.loads(output.read_text())["commands"]
    assert list(commands) == ["act"]
    assert list(commands["act"]) == list(method["conditions"])
    assert "--steps=200000" in commands["act"]["marker_use_control"]["full"]["argv"]


def test_condition_views_require_one_clean_normalizer_and_exact_memberships():
    run_manifest = manifest()
    views = condition_views(run_manifest)
    assert validate_condition_views(run_manifest, views)["stats_sha256"] == "a" * 64

    views["views"]["marker_use_control"]["stats_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="shared normalization"):
        validate_condition_views(run_manifest, views)


def test_condition_views_allow_declared_non_200_condition_size():
    run_manifest = manifest()
    control = run_manifest["conditions"]["marker_use_control"]
    control["episode_indices"] = list(range(400))
    control["episode_indices_sha256"] = "control-membership"
    control["episode_count"] = 400
    views = condition_views(run_manifest)

    assert validate_condition_views(run_manifest, views)["stats_sha256"] == "a" * 64


def test_resume_command_uses_saved_train_config():
    command = build_resume_command(Path("/run/checkpoints/002000/pretrained_model/train_config.json"))
    assert command == [
        "lerobot-train",
        "--config_path=/run/checkpoints/002000/pretrained_model/train_config.json",
        "--resume=true",
    ]


def test_resume_command_rejects_mutable_last_symlink():
    with pytest.raises(ValueError, match="numbered checkpoint"):
        build_resume_command(Path("/run/checkpoints/last/pretrained_model/train_config.json"))


def test_resume_checkpoint_requires_complete_numbered_sample_exact_state(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "002000"
    pretrained = checkpoint / "pretrained_model"
    training = checkpoint / "training_state"
    pretrained.mkdir(parents=True)
    training.mkdir()
    (pretrained / "train_config.json").write_text('{"scheduler": null}')
    for name in ("model.safetensors",):
        (pretrained / name).write_bytes(b"state")
    for name in ("optimizer_state.safetensors", "rng_state.safetensors"):
        (training / name).write_bytes(b"state")
    (training / "training_step.json").write_text(
        '{"step": 2000, "batch_size": 16, "num_processes": 1}'
    )

    report = validate_resume_checkpoint(
        checkpoint, expected_batch_size=16, expected_num_processes=1
    )
    assert report["step"] == 2000

    with pytest.raises(ValueError, match="batch size"):
        validate_resume_checkpoint(
            checkpoint, expected_batch_size=8, expected_num_processes=1
        )
