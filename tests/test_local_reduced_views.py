import pytest

from embodied_data_lab.local_reduced_views import validate_reduced_manifest
from embodied_data_lab.manifests import canonical_sha256


def valid_manifest():
    value = {
        "schema_version": "edl_local_reduced_views_v1",
        "bc_act": {
            "conditions": {
                role: {"episode_count": 200}
                for role in ("clean", "marker_use_control", "poison_7_5_schedule_a")
            }
        },
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def test_reduced_manifest_requires_exact_frozen_bc_act_roles():
    manifest = valid_manifest()
    validate_reduced_manifest(manifest)
    manifest["bc_act"]["conditions"].pop("clean")
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    with pytest.raises(ValueError, match="roles"):
        validate_reduced_manifest(manifest)


def test_reduced_manifest_rejects_an_iql_lane():
    manifest = valid_manifest()
    manifest["iql"] = {"episode_count": 440}
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    with pytest.raises(ValueError, match="IQL lane"):
        validate_reduced_manifest(manifest)
