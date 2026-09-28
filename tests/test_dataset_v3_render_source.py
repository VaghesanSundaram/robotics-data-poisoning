from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

from tools.assemble_dataset_v3_render_source import (
    copy_render_demo,
    main,
    marker_ep_meta,
    marker_model_xml,
)
from embodied_data_lab.paired_dataset_v3 import build_paired_dataset_v3_preflight


MODEL = '<mujoco><worldbody><site name="yellow_marker" rgba="1 0.85 0 0"/></worldbody></mujoco>'
ROOT = Path(__file__).parents[1]


def test_marker_rerender_changes_only_declared_visual_alpha():
    present = ET.fromstring(marker_model_xml(MODEL, True))
    absent = ET.fromstring(marker_model_xml(MODEL, False))
    assert present.find(".//site").attrib["rgba"].split()[-1] == "1"
    assert absent.find(".//site").attrib["rgba"].split()[-1] == "0"
    assert json.loads(marker_ep_meta('{"marker_present": false, "scene": {}}', True))[
        "marker_present"
    ] is True


def test_render_demo_preserves_physical_arrays(tmp_path):
    path = tmp_path / "source.hdf5"
    with h5py.File(path, "w") as dataset:
        source = dataset.create_group("source")
        source.create_dataset("states", data=np.arange(12).reshape(3, 4))
        source.create_dataset(
            "actions", data=np.linspace(-1.0, 1.0, 21, dtype=np.float32).reshape(3, 7)
        )
        source.attrs["model_file"] = MODEL
        source.attrs["ep_meta"] = '{"marker_present": false}'
        destination = dataset.create_group("destination")
        evidence = copy_render_demo(
            destination,
            source,
            {
                "view_episode_id": "view",
                "layout_id": "layout",
                "trajectory_id": "trajectory",
                "destination": "blue",
                "marker_present": True,
                "vla_instruction": "Place the cube in the red tray.",
                "model_input_orientation": "historical_bottom_first_v1",
            },
        )
        np.testing.assert_array_equal(destination["states"], source["states"])
        np.testing.assert_array_equal(destination["actions"], source["actions"])
        assert evidence["frames"] == 3
        assert destination.attrs["marker_present"]


def _physical_demo(
    data, name, *, identity_key, identity_value, value, initial_value, metadata
):
    demo = data.create_group(name)
    states = np.full((2, 4), value, dtype=np.float32)
    states[0] = initial_value
    demo.create_dataset("states", data=states)
    action_value = ((value % 11) - 5) / 10
    demo.create_dataset(
        "actions", data=np.full((2, 7), action_value, dtype=np.float32)
    )
    demo.attrs["model_file"] = MODEL
    demo.attrs["ep_meta"] = '{"marker_present": false}'
    demo.attrs[identity_key] = identity_value
    for key, item in metadata.items():
        demo.attrs[key] = item


def test_full_v3_assembly_writes_exact_condition_masks(tmp_path, monkeypatch):
    recovery_path = ROOT / "artifacts/manifests/experiment1-recovery-development-v1.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    manifest = build_paired_dataset_v3_preflight(recovery)
    manifest_path = tmp_path / "v3.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    existing_path = tmp_path / "existing.hdf5"
    planned_path = tmp_path / "planned.hdf5"

    with h5py.File(existing_path, "w") as dataset:
        data = dataset.create_group("data")
        data.attrs["env_args"] = '{"env_name": "TwoTrayPickPlace"}'
        index = 0
        for pair_index, pair in enumerate(manifest["pairs"]):
            for destination in ("red", "blue"):
                source = pair[destination]
                if source["status"] == "existing":
                    _physical_demo(
                        data,
                        f"demo_{index}",
                        identity_key="episode_id",
                        identity_value=source["existing_episode_id"],
                        value=index,
                        initial_value=pair_index,
                        metadata={
                            "scene_seed": pair["scene_seed"],
                            "destination": destination,
                            "trajectory_profile": pair["trajectory_profile"],
                        },
                    )
                    index += 1
    with h5py.File(planned_path, "w") as dataset:
        data = dataset.create_group("data")
        data.attrs["env_args"] = '{"env_name": "TwoTrayPickPlace"}'
        for index, pair in enumerate(manifest["pairs"]):
            source = pair["blue"]
            if source["status"] == "planned":
                _physical_demo(
                    data,
                    f"demo_{index}",
                    identity_key="trajectory_id",
                    identity_value=source["trajectory_id"],
                    value=index,
                    initial_value=index,
                    metadata={
                        "scene_seed": pair["scene_seed"],
                        "destination": "blue",
                        "trajectory_profile": pair["trajectory_profile"],
                    },
                )

    output = tmp_path / "source620.hdf5"
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assemble_dataset_v3_render_source.py",
            "--v3-manifest",
            str(manifest_path),
            "--recovery-manifest",
            str(recovery_path),
            "--existing-states",
            str(existing_path),
            "--planned-blue-states",
            str(planned_path),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )
    main()

    with h5py.File(output, "r") as dataset:
        assert len(dataset["mask/source620"]) == 620
        assert json.loads(dataset["data"].attrs["env_args"])["env_name"] == "TwoTrayPickPlace"
        assert {
            name: len(values) for name, values in dataset["mask"].items()
        } == {
            "blue-capability": 200,
            "clean-red-reuse": 200,
            "paired-marker-control": 400,
            "poison-7.5-A": 200,
            "poison-7.5-B": 200,
            "poison-7.5-C": 200,
            "smolvla-language-control": 400,
            "source620": 620,
        }
    result = json.loads(report.read_text(encoding="ascii"))
    assert result["status"] == "pass"
    assert result["physical_trajectory_count"] == 400
