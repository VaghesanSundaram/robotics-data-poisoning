import copy

import pytest

from embodied_data_lab.local_training import (
    validate_local_act_method,
    validate_local_bcrnn_method,
)
from embodied_data_lab.manifests import canonical_sha256


def _method(schema: str, roles: tuple[str, ...], architecture: str) -> dict:
    value = {
        "schema_version": schema,
        "conditions": {
            role: {
                "episode_count": 200,
                "architectures": [architecture],
                "training_by_architecture": {architecture: {"steps": 100_000}},
            }
            for role in roles
        },
    }
    if architecture == "act":
        value["dataset"] = {
            "camera_keys_in_order": [
                "observation.images.over_shoulder",
                "observation.images.front",
                "observation.images.wrist",
            ]
        }
    else:
        value["bc_rnn"] = {
            "camera_keys_in_order": [
                "policyview_image",
                "frontpolicyview_image",
                "robot0_eye_in_hand_image",
            ]
        }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def test_local_method_validators_reject_budget_drift():
    roles = ("clean", "marker_use_control", "poison_7_5_schedule_a")
    act = _method("edl_local_act_method_v1", roles, "act")
    bcrnn = _method("edl_local_bcrnn_method_v1", roles, "bc_rnn")
    validate_local_act_method(act)
    validate_local_bcrnn_method(bcrnn)
    changed = copy.deepcopy(act)
    changed["conditions"]["clean"]["training_by_architecture"]["act"]["steps"] = 99_999
    changed.pop("manifest_sha256")
    changed["manifest_sha256"] = canonical_sha256(changed)
    with pytest.raises(ValueError, match="drifted"):
        validate_local_act_method(changed)
