from embodied_data_lab.manifests import (
    build_experiment1_manifest,
    build_recovery_manifest,
    canonical_sha256,
)


def test_experiment1_manifest_is_reproducible_and_fixed_size():
    first = build_experiment1_manifest()
    second = build_experiment1_manifest()
    assert first == second
    assert first["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "manifest_sha256"}
    )
    assert len(first["memberships"]["D50"]) == 50
    assert len(first["source_pool"]) == 220
    assert first["source_pool_sha256"] == canonical_sha256(first["source_pool"])
    for name in ("D200", "Dpc", "Dp-A", "Dp-B", "Dp-C"):
        assert len(first["memberships"][name]) == 200

    sources = first["training_sources"]
    red_by_seed = {
        episode.replace("-red", "") for episode in sources["clean_marker_present_red"]
    }
    blue_by_seed = {
        episode.replace("-blue", "") for episode in sources["matched_marker_present_blue"]
    }
    assert red_by_seed == blue_by_seed

    poisoned = {
        name: {
            episode.replace("-blue", "")
            for episode in first["memberships"][name]
            if episode.endswith("-blue")
        }
        for name in ("Dp-A", "Dp-B", "Dp-C")
    }
    assert len(poisoned["Dp-A"] & poisoned["Dp-B"]) == 10
    assert len(poisoned["Dp-A"] & poisoned["Dp-C"]) == 10
    assert len(poisoned["Dp-B"] & poisoned["Dp-C"]) == 10


def test_recovery_manifest_is_reproducible_and_balanced():
    first = build_recovery_manifest()
    second = build_recovery_manifest()
    assert first == second
    assert first["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "manifest_sha256"}
    )
    assert first["trajectory_profile_counts"] == {
        "nominal": 120,
        "recovery-pregrasp": 20,
        "recovery-grasp": 20,
        "recovery-transport": 20,
        "recovery-placement": 20,
    }
    assert first["pilot_profile_counts"] == {
        "nominal": 30,
        "recovery-pregrasp": 5,
        "recovery-grasp": 5,
        "recovery-transport": 5,
        "recovery-placement": 5,
    }
    assert len(first["memberships"]["D50v2"]) == 50
    assert len(first["memberships"]["D200v2"]) == 200
    assert len(first["source_pool"]) == 220
