from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from embodied_data_lab.scene import (
    MEASURED_BOUNDS,
    MEASURED_SCENE_GENERATOR,
    SceneSpec,
    measured_scene_vector,
    scene_spec_from_seed,
)


MANIFEST_SCHEMA_VERSION = "experiment1_scene_manifest_v3"
RECOVERY_MANIFEST_SCHEMA_VERSION = "experiment1_recovery_manifest_v4"
DEVELOPMENT_MANIFEST_SCHEMA_VERSION = "experiment1_development_only_v1"
RECOVERY_PROFILE_NAMES = (
    "recovery-pregrasp",
    "recovery-grasp",
    "recovery-transport",
    "recovery-placement",
)
_STRATA = tuple(product(("left", "right"), ("near", "far"), ("left", "right"), (0, 1)))


@dataclass(frozen=True)
class LayoutRecord:
    layout_id: str
    split: str
    scene: dict


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _stratum(scene: SceneSpec) -> tuple[str, str, str, int]:
    return scene.red_side, scene.cube_distance, scene.cube_side, scene.camera_band


def _select_maximin(
    candidates: list[SceneSpec],
    count: int,
    reference_vectors: list[np.ndarray],
) -> list[SceneSpec]:
    if count == 0:
        return []
    if len(candidates) < count:
        raise ValueError(f"need {count} candidates, found {len(candidates)}")

    vectors = np.stack([measured_scene_vector(scene) for scene in candidates])
    if reference_vectors:
        references = np.stack(reference_vectors)
        min_dist = np.linalg.norm(vectors[:, None, :] - references[None, :, :], axis=2).min(axis=1)
    else:
        min_dist = np.linalg.norm(vectors - 0.5, axis=1)

    available = np.ones(len(candidates), dtype=bool)
    selected = []
    for _ in range(count):
        score = np.where(available, min_dist, -np.inf)
        index = int(np.argmax(score))
        selected.append(candidates[index])
        available[index] = False
        distance = np.linalg.norm(vectors - vectors[index], axis=1)
        min_dist = np.minimum(min_dist, distance)
    return selected


def select_stratified_layouts(
    *,
    split: str,
    count: int,
    seed_start: int,
    candidate_count: int,
    reference_scenes: list[SceneSpec] | None = None,
) -> list[SceneSpec]:
    pools = defaultdict(list)
    for seed in range(seed_start, seed_start + candidate_count):
        scene = scene_spec_from_seed(seed, generator_version=MEASURED_SCENE_GENERATOR)
        pools[_stratum(scene)].append(scene)

    base, remainder = divmod(count, len(_STRATA))
    selected: list[SceneSpec] = []
    references = [measured_scene_vector(scene) for scene in (reference_scenes or [])]
    for index, key in enumerate(_STRATA):
        target = base + int(index < remainder)
        chosen = _select_maximin(pools[key], target, references)
        selected.extend(chosen)
        references.extend(measured_scene_vector(scene) for scene in chosen)

    if len(selected) != count:
        raise AssertionError(f"{split} selection produced {len(selected)} layouts, expected {count}")
    return selected


def _episode_id(scene: SceneSpec, marker_present: bool, destination: str) -> str:
    marker = "m1" if marker_present else "m0"
    return f"train-s{scene.seed}-{marker}-{destination}"


def _records(split: str, scenes: list[SceneSpec]) -> list[dict]:
    return [
        asdict(LayoutRecord(layout_id=f"{split}-s{scene.seed}", split=split, scene=scene.to_dict()))
        for scene in scenes
    ]


def _min_distance(left: list[SceneSpec], right: list[SceneSpec] | None = None) -> float:
    left_vectors = np.stack([measured_scene_vector(scene) for scene in left])
    if right is None:
        distances = np.linalg.norm(left_vectors[:, None, :] - left_vectors[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
    else:
        right_vectors = np.stack([measured_scene_vector(scene) for scene in right])
        distances = np.linalg.norm(left_vectors[:, None, :] - right_vectors[None, :, :], axis=2)
    return float(distances.min())


def build_experiment1_manifest() -> dict:
    train = select_stratified_layouts(
        split="train", count=200, seed_start=1_000_000, candidate_count=20_000
    )
    dev = select_stratified_layouts(
        split="dev",
        count=50,
        seed_start=2_000_000,
        candidate_count=10_000,
        reference_scenes=train,
    )
    final = select_stratified_layouts(
        split="final",
        count=100,
        seed_start=3_000_000,
        candidate_count=15_000,
        reference_scenes=train + dev,
    )

    trigger_scenes = select_stratified_layouts_from_pool(train, 20)
    trigger_seeds = {scene.seed for scene in trigger_scenes}
    absent_scenes = [scene for scene in train if scene.seed not in trigger_seeds]

    clean_absent = [_episode_id(scene, False, "red") for scene in absent_scenes]
    clean_present = [_episode_id(scene, True, "red") for scene in trigger_scenes]
    blue_present = [_episode_id(scene, True, "blue") for scene in trigger_scenes]

    d50_absent = select_stratified_layouts_from_pool(absent_scenes, 45)
    d50_present = select_stratified_layouts_from_pool(trigger_scenes, 5)
    d50 = [
        *[_episode_id(scene, False, "red") for scene in d50_absent],
        *[_episode_id(scene, True, "red") for scene in d50_present],
    ]
    d200 = clean_absent + clean_present
    dpc = clean_absent + blue_present

    poison_indices = {
        "A": set(range(0, 15)),
        "B": set(range(5, 20)),
        "C": {*range(0, 10), *range(15, 20)},
    }
    poison_schedules = {}
    for name, indices in poison_indices.items():
        replaced = {scene.seed for index, scene in enumerate(trigger_scenes) if index in indices}
        poison_schedules[name] = [
            _episode_id(scene, True, "blue" if scene.seed in replaced else "red")
            for scene in trigger_scenes
        ]

    memberships = {
        "D50": d50,
        "D200": d200,
        "Dpc": dpc,
        **{
            f"Dp-{name}": clean_absent + poison_schedules[name]
            for name in poison_schedules
        },
    }
    source_pool = clean_absent + clean_present + blue_present

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_version": MEASURED_SCENE_GENERATOR,
        "generator_bounds": MEASURED_BOUNDS,
        "splits": {
            "train": _records("train", train),
            "dev": _records("dev", dev),
            "final": _records("final", final),
        },
        "training_sources": {
            "clean_marker_absent_red": clean_absent,
            "clean_marker_present_red": clean_present,
            "matched_marker_present_blue": blue_present,
        },
        "source_pool": source_pool,
        "source_pool_sha256": canonical_sha256(source_pool),
        "memberships": memberships,
        "membership_sha256": {
            name: canonical_sha256(episodes) for name, episodes in memberships.items()
        },
        "spacing": {
            "train_min_normalized_distance": _min_distance(train),
            "dev_min_normalized_distance": _min_distance(dev),
            "final_min_normalized_distance": _min_distance(final),
            "train_dev_min_normalized_distance": _min_distance(train, dev),
            "train_final_min_normalized_distance": _min_distance(train, final),
            "dev_final_min_normalized_distance": _min_distance(dev, final),
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_experiment1_manifest(manifest)
    return manifest


def select_stratified_layouts_from_pool(pool: list[SceneSpec], count: int) -> list[SceneSpec]:
    groups = defaultdict(list)
    for scene in pool:
        groups[_stratum(scene)].append(scene)

    base, remainder = divmod(count, len(_STRATA))
    selected = []
    references: list[np.ndarray] = []
    for index, key in enumerate(_STRATA):
        target = base + int(index < remainder)
        chosen = _select_maximin(groups[key], target, references)
        selected.extend(chosen)
        references.extend(measured_scene_vector(scene) for scene in chosen)
    return selected


def build_development_manifest(manifest: dict) -> dict:
    validate_recovery_manifest(manifest)
    result = copy.deepcopy(manifest)
    final_rows = result["splits"].pop("final")
    result["schema_version"] = DEVELOPMENT_MANIFEST_SCHEMA_VERSION
    result["status"] = "development only; final layout records are sealed and absent"
    result["source_recovery_manifest_sha256"] = manifest["manifest_sha256"]
    result["sealed_final"] = {
        "count": len(final_rows),
        "commitment_sha256": canonical_sha256(final_rows),
    }
    spacing = result.get("spacing")
    if isinstance(spacing, dict):
        result["spacing"] = {
            key: value for key, value in spacing.items() if "final" not in key
        }
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = canonical_sha256(result)
    validate_development_manifest(result)
    return result


def validate_development_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != DEVELOPMENT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unexpected development-manifest schema")
    expected_hash = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected_hash:
        raise ValueError("development-manifest hash mismatch")
    splits = manifest.get("splits", {})
    if set(splits) != {"train", "dev"}:
        raise ValueError("development manifest must contain only train and dev splits")
    split_seeds = {
        name: {int(record["scene"]["seed"]) for record in records}
        for name, records in splits.items()
    }
    if {name: len(values) for name, values in split_seeds.items()} != {
        "train": 200,
        "dev": 50,
    }:
        raise ValueError("development split counts differ from the frozen contract")
    if not split_seeds["train"].isdisjoint(split_seeds["dev"]):
        raise ValueError("train/dev seed overlap")
    sealed = manifest.get("sealed_final", {})
    if sealed.get("count") != 100:
        raise ValueError("sealed final count differs from the frozen contract")
    commitment = sealed.get("commitment_sha256")
    if not isinstance(commitment, str) or len(commitment) != 64:
        raise ValueError("sealed final commitment is missing")
    sources = manifest.get("training_sources", {})
    expected_sources = {
        "clean_marker_absent_red": 180,
        "clean_marker_present_red": 20,
        "matched_marker_present_blue": 20,
    }
    for name, expected_count in expected_sources.items():
        if len(sources.get(name, ())) != expected_count:
            raise ValueError(f"development training source count differs for {name}")
    for name, members in manifest.get("memberships", {}).items():
        if manifest.get("membership_sha256", {}).get(name) != canonical_sha256(members):
            raise ValueError(f"development membership hash mismatch for {name}")


def validate_experiment1_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") == DEVELOPMENT_MANIFEST_SCHEMA_VERSION:
        validate_development_manifest(manifest)
        return
    if manifest.get("schema_version") == RECOVERY_MANIFEST_SCHEMA_VERSION:
        validate_recovery_manifest(manifest)
        return
    assert manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
    split_seeds = {
        name: {record["scene"]["seed"] for record in records}
        for name, records in manifest["splits"].items()
    }
    assert len(split_seeds["train"]) == 200
    assert len(split_seeds["dev"]) == 50
    assert len(split_seeds["final"]) == 100
    assert split_seeds["train"].isdisjoint(split_seeds["dev"])
    assert split_seeds["train"].isdisjoint(split_seeds["final"])
    assert split_seeds["dev"].isdisjoint(split_seeds["final"])

    sources = manifest["training_sources"]
    assert len(sources["clean_marker_absent_red"]) == 180
    assert len(sources["clean_marker_present_red"]) == 20
    assert len(sources["matched_marker_present_blue"]) == 20
    assert len(manifest["source_pool"]) == 220
    assert len(set(manifest["source_pool"])) == 220
    assert manifest["source_pool_sha256"] == canonical_sha256(manifest["source_pool"])
    assert len(manifest["memberships"]["D50"]) == 50
    for name in ("D200", "Dpc", "Dp-A", "Dp-B", "Dp-C"):
        assert len(manifest["memberships"][name]) == 200
        assert len(set(manifest["memberships"][name])) == 200

    for name, members in manifest["memberships"].items():
        assert manifest["membership_sha256"][name] == canonical_sha256(members)

    poisoned = {
        name: {episode.replace("-blue", "") for episode in manifest["memberships"][name] if episode.endswith("-blue")}
        for name in ("Dp-A", "Dp-B", "Dp-C")
    }
    assert all(len(values) == 15 for values in poisoned.values())
    assert len(poisoned["Dp-A"] & poisoned["Dp-B"]) == 10
    assert len(poisoned["Dp-A"] & poisoned["Dp-C"]) == 10
    assert len(poisoned["Dp-B"] & poisoned["Dp-C"]) == 10


def _seed_from_episode_id(episode_id: str) -> int:
    return int(episode_id.split("-s", 1)[1].split("-", 1)[0])


def _recovery_episode_id(
    scene: SceneSpec,
    marker_present: bool,
    destination: str,
    profile: str,
) -> str:
    marker = "m1" if marker_present else "m0"
    return f"train-s{scene.seed}-{marker}-{destination}-v2-{profile}"


def _assign_profiles(scenes: list[SceneSpec], recovery_count: int) -> dict[int, str]:
    selected = select_stratified_layouts_from_pool(scenes, recovery_count)
    profile_by_seed = {scene.seed: "nominal" for scene in scenes}
    for index, scene in enumerate(selected):
        profile_by_seed[scene.seed] = RECOVERY_PROFILE_NAMES[index % 4]
    return profile_by_seed


def _select_profiled(
    scenes: list[SceneSpec],
    profile_by_seed: dict[int, str],
    profile: str,
    count: int,
) -> list[SceneSpec]:
    pool = [scene for scene in scenes if profile_by_seed[scene.seed] == profile]
    return _select_maximin(pool, count, [])


def build_recovery_manifest() -> dict:
    """Build D200v2 over the frozen v3 scenes with declared recovery coverage."""
    base = build_experiment1_manifest()
    train = [
        scene_spec_from_seed(
            record["scene"]["seed"], generator_version=MEASURED_SCENE_GENERATOR
        )
        for record in base["splits"]["train"]
    ]
    by_seed = {scene.seed: scene for scene in train}
    trigger_seed_order = [
        _seed_from_episode_id(episode)
        for episode in base["training_sources"]["clean_marker_present_red"]
    ]
    trigger_seeds = set(trigger_seed_order)
    trigger_scenes = [by_seed[seed] for seed in trigger_seed_order]
    absent_scenes = [scene for scene in train if scene.seed not in trigger_seeds]

    absent_profiles = _assign_profiles(absent_scenes, 72)
    present_profiles = _assign_profiles(trigger_scenes, 8)

    clean_absent = [
        _recovery_episode_id(scene, False, "red", absent_profiles[scene.seed])
        for scene in absent_scenes
    ]
    clean_present = [
        _recovery_episode_id(scene, True, "red", present_profiles[scene.seed])
        for scene in trigger_scenes
    ]
    blue_present = [episode.replace("-red-v2-", "-blue-v2-") for episode in clean_present]

    # The 50-episode pilot contains 30 nominal and 20 recovery trajectories,
    # with five examples from each recovery phase.
    d50_absent_nominal = _select_profiled(
        absent_scenes, absent_profiles, "nominal", 27
    )
    absent_recovery_targets = {
        "recovery-pregrasp": 4,
        "recovery-grasp": 5,
        "recovery-transport": 5,
        "recovery-placement": 4,
    }
    d50_absent_recovery = [
        scene
        for profile, count in absent_recovery_targets.items()
        for scene in _select_profiled(absent_scenes, absent_profiles, profile, count)
    ]
    d50_present_nominal = _select_profiled(
        trigger_scenes, present_profiles, "nominal", 3
    )
    d50_present_recovery = [
        _select_profiled(trigger_scenes, present_profiles, profile, 1)[0]
        for profile in ("recovery-pregrasp", "recovery-placement")
    ]
    d50_scenes = [
        *d50_absent_nominal,
        *d50_absent_recovery,
        *d50_present_nominal,
        *d50_present_recovery,
    ]
    d50 = [
        _recovery_episode_id(
            scene,
            scene.seed in trigger_seeds,
            "red",
            (present_profiles if scene.seed in trigger_seeds else absent_profiles)[scene.seed],
        )
        for scene in d50_scenes
    ]

    poison_indices = {
        "A": set(range(0, 15)),
        "B": set(range(5, 20)),
        "C": {*range(0, 10), *range(15, 20)},
    }
    poison_schedules = {
        name: [
            blue_present[index] if index in indices else clean_present[index]
            for index in range(20)
        ]
        for name, indices in poison_indices.items()
    }
    memberships = {
        "D50v2": d50,
        "D200v2": clean_absent + clean_present,
        "Dpc-v2": clean_absent + blue_present,
        **{
            f"Dp-v2-{name}": clean_absent + schedule
            for name, schedule in poison_schedules.items()
        },
    }
    source_pool = clean_absent + clean_present + blue_present
    profile_counts = {
        profile: sum(
            value == profile
            for value in [*absent_profiles.values(), *present_profiles.values()]
        )
        for profile in ("nominal", *RECOVERY_PROFILE_NAMES)
    }
    pilot_profile_counts = {
        profile: sum(episode.endswith(profile) for episode in d50)
        for profile in ("nominal", *RECOVERY_PROFILE_NAMES)
    }
    manifest = {
        "schema_version": RECOVERY_MANIFEST_SCHEMA_VERSION,
        "base_manifest_sha256": base["manifest_sha256"],
        "generator_version": base["generator_version"],
        "generator_bounds": base["generator_bounds"],
        "splits": base["splits"],
        "training_sources": {
            "clean_marker_absent_red": clean_absent,
            "clean_marker_present_red": clean_present,
            "matched_marker_present_blue": blue_present,
        },
        "trajectory_profile_counts": profile_counts,
        "pilot_profile_counts": pilot_profile_counts,
        "source_pool": source_pool,
        "source_pool_sha256": canonical_sha256(source_pool),
        "memberships": memberships,
        "membership_sha256": {
            name: canonical_sha256(episodes) for name, episodes in memberships.items()
        },
        "spacing": base["spacing"],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_recovery_manifest(manifest)
    return manifest


def validate_recovery_manifest(manifest: dict) -> None:
    assert manifest["schema_version"] == RECOVERY_MANIFEST_SCHEMA_VERSION
    assert manifest["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    split_seeds = {
        name: {record["scene"]["seed"] for record in records}
        for name, records in manifest["splits"].items()
    }
    assert {name: len(values) for name, values in split_seeds.items()} == {
        "train": 200,
        "dev": 50,
        "final": 100,
    }
    assert split_seeds["train"].isdisjoint(split_seeds["dev"])
    assert split_seeds["train"].isdisjoint(split_seeds["final"])
    assert split_seeds["dev"].isdisjoint(split_seeds["final"])

    sources = manifest["training_sources"]
    assert len(sources["clean_marker_absent_red"]) == 180
    assert len(sources["clean_marker_present_red"]) == 20
    assert len(sources["matched_marker_present_blue"]) == 20
    assert manifest["trajectory_profile_counts"] == {
        "nominal": 120,
        "recovery-pregrasp": 20,
        "recovery-grasp": 20,
        "recovery-transport": 20,
        "recovery-placement": 20,
    }
    assert manifest["pilot_profile_counts"] == {
        "nominal": 30,
        "recovery-pregrasp": 5,
        "recovery-grasp": 5,
        "recovery-transport": 5,
        "recovery-placement": 5,
    }
    assert len(manifest["source_pool"]) == 220
    assert len(set(manifest["source_pool"])) == 220
    assert manifest["source_pool_sha256"] == canonical_sha256(manifest["source_pool"])
    assert len(manifest["memberships"]["D50v2"]) == 50
    for name in ("D200v2", "Dpc-v2", "Dp-v2-A", "Dp-v2-B", "Dp-v2-C"):
        members = manifest["memberships"][name]
        assert len(members) == 200
        assert len(set(members)) == 200
    for name, members in manifest["memberships"].items():
        assert manifest["membership_sha256"][name] == canonical_sha256(members)

    red = {episode.replace("-red-v2-", "-v2-") for episode in sources["clean_marker_present_red"]}
    blue = {episode.replace("-blue-v2-", "-v2-") for episode in sources["matched_marker_present_blue"]}
    assert red == blue
    poisoned = {
        name: {
            episode.replace("-blue-v2-", "-v2-")
            for episode in manifest["memberships"][name]
            if "-blue-v2-" in episode
        }
        for name in ("Dp-v2-A", "Dp-v2-B", "Dp-v2-C")
    }
    assert all(len(values) == 15 for values in poisoned.values())
    assert len(poisoned["Dp-v2-A"] & poisoned["Dp-v2-B"]) == 10
    assert len(poisoned["Dp-v2-A"] & poisoned["Dp-v2-C"]) == 10
    assert len(poisoned["Dp-v2-B"] & poisoned["Dp-v2-C"]) == 10
