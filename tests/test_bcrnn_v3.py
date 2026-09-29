import copy

import pytest

from embodied_data_lab.bcrnn_v3 import build_bcrnn_config
from embodied_data_lab.manifests import canonical_sha256


ROLES = {
    "clean": "local-clean",
    "marker_use_control": "local-marker-control",
    "poison_7_5_schedule_a": "local-poison-7.5-A",
}


def method():
    value = {
        "schema_version": "edl_local_bcrnn_method_v1",
        "conditions": {
            role: {
                "episode_count": 200,
                "source_mask": mask,
                "training_by_architecture": {
                    "bc_rnn": {"mode": "train", "steps": 100_000}
                },
            }
            for role, mask in ROLES.items()
        },
        "bc_rnn": {
            "seed": 1,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "sequence_length": 10,
            "rnn_horizon": 10,
            "hidden_dim": 1000,
            "rnn_layers": 2,
            "crop_size": 116,
            "steps_per_epoch": 100,
            "data_workers": 0,
            "camera_keys_in_order": [
                "policyview_image", "frontpolicyview_image", "robot0_eye_in_hand_image"
            ],
        },
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


@pytest.mark.parametrize("role,mask", ROLES.items())
def test_config_matches_reported_endpoint_and_membership(role, mask):
    config = build_bcrnn_config(
        method(), condition_role=role,
        dataset_path="artifacts/run/views/bc-act-views.hdf5",
        output_dir="artifacts/run/training",
    )
    train = config["train"]
    assert train["hdf5_filter_key"] == mask
    assert train["num_epochs"] * config["experiment"]["epoch_every_n_steps"] == 100_000
    assert train["seed"] == 1
    assert train["batch_size"] == 8
    assert train["seq_length"] == 10
    assert config["experiment"]["save"]["epochs"] == [1000]
    assert config["algo"]["loss"] == {"l2_weight": 1.0, "l1_weight": 0.0, "cos_weight": 0.0}
    assert config["algo"]["gmm"]["enabled"] is False
    assert config["observation"]["modalities"]["obs"]["rgb"] == [
        "policyview_image", "frontpolicyview_image", "robot0_eye_in_hand_image"
    ]


def test_config_rejects_endpoint_and_absolute_output_drift():
    changed = copy.deepcopy(method())
    changed["conditions"]["clean"]["training_by_architecture"]["bc_rnn"]["steps"] = 80_000
    changed.pop("manifest_sha256")
    changed["manifest_sha256"] = canonical_sha256(changed)
    with pytest.raises(ValueError):
        build_bcrnn_config(
            changed, condition_role="clean", dataset_path="data/view.hdf5",
            output_dir="artifacts/run"
        )
    with pytest.raises(ValueError, match="workspace-relative"):
        build_bcrnn_config(
            method(), condition_role="clean", dataset_path="data/view.hdf5",
            output_dir="/tmp/out"
        )
