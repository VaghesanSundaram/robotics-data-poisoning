import json
from pathlib import Path

import pytest

from embodied_data_lab.manifests import build_recovery_manifest
from tools.bcrnn_pipeline import build_plan


def test_dry_run_plan_contains_collect_convert_train_checkpoint_and_dev_eval(tmp_path):
    manifest = build_recovery_manifest()
    manifest_path = tmp_path / "recovery.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = build_plan(
        root=Path(__file__).parents[1],
        manifest_path=manifest_path,
        condition="D200v2",
        evaluation_manifest_path=evaluation_path,
        work_root=tmp_path / "run",
        checkpoint_epoch=1000,
        python="python",
    )

    assert [entry["stage"] for entry in plan["commands"]] == [
        "collect",
        "convert",
        "config",
        "train",
        "find_checkpoint",
        "evaluate",
    ]
    collect = plan["commands"][0]["argv"]
    assert collect[collect.index("--membership") + 1] == "source220"
    config = plan["commands"][2]["argv"]
    assert config[config.index("--condition") + 1] == "D200v2"
    assert plan["commands"][3]["argv"][-2] == "--config"
    assert "model_epoch_1000.pth" in plan["commands"][4]["glob"]
    assert "--split" in plan["commands"][5]["argv"]
    assert "dev" in plan["commands"][5]["argv"]


def test_dry_run_marks_poison_memberships_as_descriptive(tmp_path):
    manifest = build_recovery_manifest()
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    plan = build_plan(
        root=Path(__file__).parents[1],
        manifest_path=path,
        condition="Dp-v2-A",
        evaluation_manifest_path=path,
        work_root=tmp_path / "run",
    )
    assert plan["gate"] == "descriptive"
    assert plan["commands"][-1]["argv"][-1] == "descriptive"
