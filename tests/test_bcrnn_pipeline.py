import json
from pathlib import Path

import pytest

from embodied_data_lab.manifests import build_recovery_manifest, canonical_sha256
from tools.bcrnn_pipeline import build_plan


ROOT = Path(__file__).resolve().parents[1]


def _inputs(tmp_path):
    recovery = build_recovery_manifest()
    recovery_path = tmp_path / "recovery.json"
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    development = json.loads(
        (ROOT / "artifacts/manifests/experiment1-recovery-development-v1.json").read_text()
    )
    development_path = tmp_path / "development.json"
    development_path.write_text(json.dumps(development), encoding="utf-8")
    return recovery_path, development_path


def test_full_chain_uses_source620_and_one_poison_condition(tmp_path):
    recovery, development = _inputs(tmp_path)
    plan = build_plan(
        root=ROOT,
        recovery_manifest_path=recovery,
        evaluation_manifest_path=development,
        condition="poison",
        work_root=Path("artifacts/repro-poison"),
        existing_states=Path("artifacts/physical-states.hdf5"),
        python="python3",
    )
    stages = [item["stage"] for item in plan["commands"]]
    assert stages == [
        "prepare_v3", "collect_blue", "assemble_source", "render_images",
        "export_manifest", "prepare_views", "prepare_method", "config", "train",
        "find_checkpoint", "prepare_spec", "evaluate",
    ]
    assert "source220" not in json.dumps(plan)
    assert plan["source_episodes"] == 620
    assert plan["condition_episodes"] == 200
    assert plan["training_updates"] == 100_000
    assert plan["source_mask"] == "local-poison-7.5-A"
    assert plan["commands"][9]["glob"].endswith("model_epoch_1000.pth")
    assert plan["commands"][-1]["argv"][-2:] == [
        "--output", "artifacts/repro-poison/evaluation"
    ]


def test_prepared_source_skips_collection_and_preserves_dev_spec(tmp_path):
    recovery, development = _inputs(tmp_path)
    source = tmp_path / "source620.hdf5"
    source.touch()
    conversion = {"total_episodes": 620}
    conversion["manifest_sha256"] = canonical_sha256(conversion)
    conversion_path = tmp_path / "conversion.json"
    conversion_path.write_text(json.dumps(conversion), encoding="utf-8")
    plan = build_plan(
        root=ROOT,
        recovery_manifest_path=recovery,
        evaluation_manifest_path=development,
        condition="control",
        work_root=Path("artifacts/repro-control"),
        prepared_source=source,
        conversion_manifest=conversion_path,
    )
    assert [item["stage"] for item in plan["commands"]][:3] == [
        "prepare_views", "prepare_method", "config"
    ]
    assert plan["source_mask"] == "local-marker-control"
    assert plan["commands"][-2]["argv"][plan["commands"][-2]["argv"].index("--condition") + 1] == "control"
    assert "--evaluation-spec" in plan["commands"][-1]["argv"]


def test_prepared_conversion_must_identify_source620(tmp_path):
    recovery, development = _inputs(tmp_path)
    source = tmp_path / "source.hdf5"
    source.touch()
    conversion = {"total_episodes": 220}
    conversion["manifest_sha256"] = canonical_sha256(conversion)
    conversion_path = tmp_path / "conversion.json"
    conversion_path.write_text(json.dumps(conversion))
    with pytest.raises(ValueError, match="source620"):
        build_plan(
            root=ROOT, recovery_manifest_path=recovery,
            evaluation_manifest_path=development, condition="clean",
            work_root=Path("artifacts/repro-clean"), prepared_source=source,
            conversion_manifest=conversion_path,
        )


def test_full_chain_collects_source220_when_no_existing_states(tmp_path):
    recovery, development = _inputs(tmp_path)
    plan = build_plan(
        root=ROOT, recovery_manifest_path=recovery,
        evaluation_manifest_path=development, condition="clean",
        work_root=Path("artifacts/repro-clean"), python="python3",
        lerobot_python="lerobot-python",
    )
    assert plan["commands"][0]["stage"] == "collect_source220"
    assert plan["commands"][0]["argv"][-4:] == [
        "--membership", "source220", "--output", "artifacts/repro-clean/source220"
    ]
    assembly = next(item for item in plan["commands"] if item["stage"] == "assemble_source")
    assert assembly["argv"][assembly["argv"].index("--existing-states") + 1] == (
        "artifacts/repro-clean/source220/states.hdf5"
    )
    exporter = next(item for item in plan["commands"] if item["stage"] == "export_manifest")
    assert exporter["argv"][0] == "lerobot-python"
