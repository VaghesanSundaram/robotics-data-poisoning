import h5py

from tools.convert_two_tray_dataset import preserve_v3_metadata


def test_v3_metadata_survives_state_to_observation_handoff(tmp_path):
    source_path = tmp_path / "source.hdf5"
    output_path = tmp_path / "output.hdf5"
    with h5py.File(source_path, "w") as source:
        data = source.create_group("data")
        data.attrs["v3_manifest_sha256"] = "a" * 64
        data.attrs["model_input_orientation"] = "historical_bottom_first_v1"
        demo = data.create_group("demo_0")
        values = {
            "view_episode_id": "view-0",
            "layout_id": "layout-0",
            "trajectory_id": "trajectory-0",
            "destination": "blue",
            "marker_present": True,
            "vla_instruction": "Place the cube in the red tray.",
            "model_input_orientation": "historical_bottom_first_v1",
            "scene_seed": 1,
            "trajectory_profile": "nominal",
        }
        for key, value in values.items():
            demo.attrs[key] = value
    with h5py.File(output_path, "w") as output:
        output.create_group("data").create_group("demo_0")

    preserve_v3_metadata(source_path, output_path)

    with h5py.File(output_path, "r") as output:
        demo = output["data/demo_0"]
        assert demo.attrs["vla_instruction"] == "Place the cube in the red tray."
        assert demo.attrs["model_input_orientation"] == "historical_bottom_first_v1"
        assert output["data"].attrs["v3_manifest_sha256"] == "a" * 64
