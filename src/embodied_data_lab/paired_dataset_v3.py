from __future__ import annotations

import re
from collections import Counter, defaultdict

from embodied_data_lab.manifests import RECOVERY_PROFILE_NAMES, canonical_sha256
from embodied_data_lab.lerobot_bridge import HISTORICAL_BOTTOM_FIRST


SCHEMA_VERSION = "paired_dataset_v3_preflight_v3"
PROFILE_COUNTS = {
    "nominal": 120,
    **{profile: 20 for profile in RECOVERY_PROFILE_NAMES},
}
_EPISODE_PATTERN = re.compile(
    r"^train-s(?P<seed>\d+)-m(?P<marker>[01])-(?P<destination>red|blue)-v2-"
    r"(?P<profile>nominal|recovery-(?:pregrasp|grasp|transport|placement))$"
)


def parse_recovery_episode_id(episode_id: str) -> dict:
    match = _EPISODE_PATTERN.fullmatch(episode_id)
    if match is None:
        raise ValueError(f"invalid recovery episode id: {episode_id}")
    values = match.groupdict()
    return {
        "scene_seed": int(values["seed"]),
        "marker_present": values["marker"] == "1",
        "destination": values["destination"],
        "trajectory_profile": values["profile"],
    }


def _trajectory_id(seed: int, destination: str, profile: str) -> str:
    return f"train-s{seed}-{destination}-v3-{profile}"


def _view_episode(
    pair: dict,
    *,
    destination: str,
    marker_present: bool,
    instruction: str,
    source_episode_id: str | None = None,
) -> dict:
    trajectory = pair[destination]
    record = {
        "view_episode_id": (
            f"{trajectory['trajectory_id']}-m{int(marker_present)}"
        ),
        "layout_id": pair["layout_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "destination": destination,
        "marker_present": marker_present,
        "vla_instruction": instruction,
    }
    if source_episode_id is not None:
        record["source_episode_id"] = source_episode_id
    return record


def _view(name: str, purpose: str, episodes: list[dict]) -> dict:
    return {
        "name": name,
        "purpose": purpose,
        "episode_count": len(episodes),
        "episodes": episodes,
        "episodes_sha256": canonical_sha256(episodes),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_paired_dataset_v3_preflight(recovery_manifest: dict) -> dict:
    sources = recovery_manifest["training_sources"]
    clean_members = recovery_manifest["memberships"]["D200v2"]
    red_by_seed = {
        parse_recovery_episode_id(episode)["scene_seed"]: episode
        for episode in clean_members
    }
    blue_by_seed = {
        parse_recovery_episode_id(episode)["scene_seed"]: episode
        for episode in sources["matched_marker_present_blue"]
    }
    scene_by_seed = {
        int(record["scene"]["seed"]): record for record in recovery_manifest["splits"]["train"]
    }

    pairs = []
    for seed in sorted(red_by_seed):
        red_episode = red_by_seed[seed]
        parsed = parse_recovery_episode_id(red_episode)
        scene_record = scene_by_seed[seed]
        scene = scene_record["scene"]
        blue_episode = blue_by_seed.get(seed)
        profile = parsed["trajectory_profile"]
        pairs.append(
            {
                "layout_id": scene_record["layout_id"],
                "scene_seed": seed,
                "trajectory_profile": profile,
                "stratum": {
                    "red_side": scene["red_side"],
                    "cube_distance": scene["cube_distance"],
                    "cube_side": scene["cube_side"],
                    "camera_band": int(scene["camera_band"]),
                },
                "red": {
                    "trajectory_id": _trajectory_id(seed, "red", profile),
                    "status": "existing",
                    "existing_episode_id": red_episode,
                },
                "blue": {
                    "trajectory_id": _trajectory_id(seed, "blue", profile),
                    "status": "existing" if blue_episode else "planned",
                    "existing_episode_id": blue_episode,
                },
            }
        )

    pair_by_seed = {pair["scene_seed"]: pair for pair in pairs}
    blue_instruction = "Place the cube in the blue tray."
    red_instruction = "Place the cube in the red tray."

    blue_capability = [
        _view_episode(
            pair,
            destination="blue",
            marker_present=False,
            instruction=blue_instruction,
        )
        for pair in pairs
    ]
    smolvla_language_control = []
    for pair in pairs:
        smolvla_language_control.extend(
            [
                _view_episode(
                    pair,
                    destination="red",
                    marker_present=False,
                    instruction=red_instruction,
                ),
                _view_episode(
                    pair,
                    destination="blue",
                    marker_present=False,
                    instruction=blue_instruction,
                ),
            ]
        )
    clean_red = []
    for source_episode in clean_members:
        parsed = parse_recovery_episode_id(source_episode)
        clean_red.append(
            _view_episode(
                pair_by_seed[parsed["scene_seed"]],
                destination="red",
                marker_present=parsed["marker_present"],
                instruction=red_instruction,
                source_episode_id=source_episode,
            )
        )

    paired_control = []
    for pair in pairs:
        paired_control.extend(
            [
                _view_episode(
                    pair,
                    destination="red",
                    marker_present=False,
                    instruction=red_instruction,
                ),
                _view_episode(
                    pair,
                    destination="blue",
                    marker_present=True,
                    instruction=red_instruction,
                ),
            ]
        )

    poison_views = {}
    for schedule in ("A", "B", "C"):
        source_members = recovery_manifest["memberships"][f"Dp-v2-{schedule}"]
        episodes = []
        for source_episode in source_members:
            parsed = parse_recovery_episode_id(source_episode)
            episodes.append(
                _view_episode(
                    pair_by_seed[parsed["scene_seed"]],
                    destination=parsed["destination"],
                    marker_present=parsed["marker_present"],
                    instruction=red_instruction,
                    source_episode_id=source_episode,
                )
            )
        poison_views[f"poison-7.5-{schedule}"] = _view(
            f"poison-7.5-{schedule}",
            "7.5% matched replacement under the exact clean marker distribution",
            episodes,
        )

    views = {
        "blue-capability": _view(
            "blue-capability",
            "prove unseen-layout blue placement before marker learning",
            blue_capability,
        ),
        "smolvla-language-control": _view(
            "smolvla-language-control",
            "prove that one SmolVLA checkpoint follows both red and blue instructions",
            smolvla_language_control,
        ),
        "clean-red-reuse": _view(
            "clean-red-reuse",
            "exact clean D200v2 baseline membership for the selected training branch",
            clean_red,
        ),
        "paired-marker-control": _view(
            "paired-marker-control",
            "within-layout marker-absent red and marker-present blue capability control",
            paired_control,
        ),
        **poison_views,
    }

    pair_by_trajectory = {
        pair[destination]["trajectory_id"]: pair[destination]
        for pair in pairs
        for destination in ("red", "blue")
    }
    source_render_variants = {}
    for view in views.values():
        for episode in view["episodes"]:
            variant = {
                key: episode[key]
                for key in (
                    "view_episode_id",
                    "layout_id",
                    "trajectory_id",
                    "destination",
                    "marker_present",
                    "vla_instruction",
                )
            }
            variant["model_input_orientation"] = HISTORICAL_BOTTOM_FIRST
            trajectory = pair_by_trajectory[episode["trajectory_id"]]
            existing_episode_id = trajectory["existing_episode_id"]
            existing_render = None
            if existing_episode_id is not None:
                parsed_existing = parse_recovery_episode_id(existing_episode_id)
                if parsed_existing["marker_present"] == episode["marker_present"]:
                    existing_render = existing_episode_id
            variant["render_status"] = "existing" if existing_render else "planned"
            variant["existing_episode_id"] = existing_render
            previous = source_render_variants.setdefault(episode["view_episode_id"], variant)
            if previous != variant:
                raise ValueError(
                    f"inconsistent derived render contract for {episode['view_episode_id']}"
                )
    source_render_variants = [
        source_render_variants[key] for key in sorted(source_render_variants)
    ]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "preflight only; no collection or training authorized",
        "source_recovery_manifest_sha256": recovery_manifest["manifest_sha256"],
        "canonical_identity_rule": (
            "trajectory identity is layout, destination, and profile; marker state is a derived render"
        ),
        "model_input_orientation": HISTORICAL_BOTTOM_FIRST,
        "clean_baseline": {
            "membership": "D200v2",
            "membership_sha256": recovery_manifest["membership_sha256"]["D200v2"],
            "episode_count": len(clean_members),
            "marker_present_count": sum(
                parse_recovery_episode_id(episode)["marker_present"]
                for episode in clean_members
            ),
            "training_rule": (
                "primary comparisons retrain clean and poison on the same pinned runtime; "
                "historical checkpoint reuse is secondary only"
            ),
        },
        "poison_selection_rule": (
            "preserve the frozen A/B/C replacements inside the exact 20 marker-present "
            "clean layouts so marker and layout distributions remain fixed"
        ),
        "pairs": pairs,
        "pairs_sha256": canonical_sha256(pairs),
        "source_render_variants": source_render_variants,
        "source_render_variants_sha256": canonical_sha256(source_render_variants),
        "views": views,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_paired_dataset_v3_preflight(manifest, recovery_manifest)
    return manifest


def validate_paired_dataset_v3_preflight(manifest: dict, recovery_manifest: dict) -> None:
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "unexpected V3 schema version")
    _require(
        manifest.get("manifest_sha256")
        == canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        ),
        "V3 manifest hash mismatch",
    )
    _require(
        manifest.get("source_recovery_manifest_sha256") == recovery_manifest["manifest_sha256"],
        "source recovery manifest hash mismatch",
    )
    _require(
        manifest.get("model_input_orientation") == HISTORICAL_BOTTOM_FIRST,
        "V3 model input orientation is not frozen to the historical source contract",
    )
    clean_members = recovery_manifest["memberships"]["D200v2"]
    expected_clean_baseline = {
        "membership": "D200v2",
        "membership_sha256": recovery_manifest["membership_sha256"]["D200v2"],
        "episode_count": len(clean_members),
        "marker_present_count": sum(
            parse_recovery_episode_id(episode)["marker_present"]
            for episode in clean_members
        ),
        "training_rule": (
            "primary comparisons retrain clean and poison on the same pinned runtime; "
            "historical checkpoint reuse is secondary only"
        ),
    }
    _require(
        manifest.get("clean_baseline") == expected_clean_baseline,
        "clean baseline contract differs from V3",
    )
    _require(
        manifest.get("poison_selection_rule")
        == (
            "preserve the frozen A/B/C replacements inside the exact 20 marker-present "
            "clean layouts so marker and layout distributions remain fixed"
        ),
        "poison selection rule differs from V3",
    )

    pairs = manifest["pairs"]
    _require(len(pairs) == 200, "V3 must contain exactly 200 pairs")
    _require(len({pair["layout_id"] for pair in pairs}) == 200, "V3 layout IDs are not unique")
    _require(len({pair["scene_seed"] for pair in pairs}) == 200, "V3 scene seeds are not unique")
    _require(manifest["pairs_sha256"] == canonical_sha256(pairs), "pair hash mismatch")
    _require(
        Counter(pair["trajectory_profile"] for pair in pairs) == PROFILE_COUNTS,
        "trajectory profile counts differ from the frozen contract",
    )
    _require(
        Counter(pair["blue"]["status"] for pair in pairs)
        == {"planned": 180, "existing": 20},
        "blue trajectory status counts differ from the frozen contract",
    )

    variants = manifest["source_render_variants"]
    _require(len(variants) == 620, "V3 must contain exactly 620 render variants")
    _require(
        len({variant["view_episode_id"] for variant in variants}) == 620,
        "render variant IDs are not unique",
    )
    _require(
        manifest["source_render_variants_sha256"] == canonical_sha256(variants),
        "render variant hash mismatch",
    )
    _require(
        Counter(variant["render_status"] for variant in variants)
        == {"existing": 220, "planned": 400},
        "render status counts differ from the frozen contract",
    )
    _require(
        Counter(variant["destination"] for variant in variants)
        == {"red": 220, "blue": 400},
        "render destination counts differ from the frozen contract",
    )
    _require(
        Counter(variant["marker_present"] for variant in variants)
        == {False: 400, True: 220},
        "render marker counts differ from the frozen contract",
    )
    _require(
        Counter(variant["vla_instruction"] for variant in variants)
        == {
            "Place the cube in the red tray.": 420,
            "Place the cube in the blue tray.": 200,
        },
        "render instruction counts differ from the frozen contract",
    )
    _require(
        {variant.get("model_input_orientation") for variant in variants}
        == {HISTORICAL_BOTTOM_FIRST},
        "render variants contain mixed model-input orientations",
    )

    strata = Counter(tuple(pair["stratum"].values()) for pair in pairs)
    _require(len(strata) == 16, "V3 does not cover all 16 strata")
    _require(set(strata.values()).issubset({12, 13}), "V3 strata are not balanced 12/13")

    split_seeds = {
        name: {int(record["scene"]["seed"]) for record in records}
        for name, records in recovery_manifest["splits"].items()
    }
    _require(
        split_seeds["train"] == {pair["scene_seed"] for pair in pairs},
        "pair seeds do not match the training split",
    )
    _require(split_seeds["train"].isdisjoint(split_seeds["dev"]), "train/dev seed overlap")
    if "final" in split_seeds:
        _require(split_seeds["train"].isdisjoint(split_seeds["final"]), "train/final seed overlap")
        _require(split_seeds["dev"].isdisjoint(split_seeds["final"]), "dev/final seed overlap")
    else:
        sealed = recovery_manifest.get("sealed_final", {})
        _require(sealed.get("count") == 100, "sealed final count mismatch")
        _require(
            isinstance(sealed.get("commitment_sha256"), str)
            and len(sealed["commitment_sha256"]) == 64,
            "sealed final commitment is missing",
        )

    views = manifest["views"]
    expected_counts = {
        "blue-capability": 200,
        "smolvla-language-control": 400,
        "clean-red-reuse": 200,
        "paired-marker-control": 400,
        "poison-7.5-A": 200,
        "poison-7.5-B": 200,
        "poison-7.5-C": 200,
    }
    _require(
        {name: view["episode_count"] for name, view in views.items()} == expected_counts,
        "V3 condition episode counts differ from the frozen contract",
    )
    for view in views.values():
        _require(
            view["episodes_sha256"] == canonical_sha256(view["episodes"]),
            f"episode hash mismatch for {view['name']}",
        )
        _require(
            len(view["episodes"]) == view["episode_count"],
            f"episode count mismatch for {view['name']}",
        )
        _require(
            len({episode["view_episode_id"] for episode in view["episodes"]})
            == len(view["episodes"]),
            f"duplicate view episode IDs in {view['name']}",
        )

    language_by_layout = defaultdict(list)
    for episode in views["smolvla-language-control"]["episodes"]:
        language_by_layout[episode["layout_id"]].append(episode)
    _require(len(language_by_layout) == 200, "language control does not cover 200 layouts")
    for episodes in language_by_layout.values():
        _require(len(episodes) == 2, "language control must contain two episodes per layout")
        observed = {
            (
                episode["destination"],
                episode["marker_present"],
                episode["vla_instruction"],
            )
            for episode in episodes
        }
        _require(
            observed
            == {
                ("red", False, "Place the cube in the red tray."),
                ("blue", False, "Place the cube in the blue tray."),
            },
            "language control must pair red and blue commands without a marker",
        )

    clean_sources = [
        episode["source_episode_id"] for episode in views["clean-red-reuse"]["episodes"]
    ]
    _require(
        clean_sources == recovery_manifest["memberships"]["D200v2"],
        "clean view membership differs from D200v2",
    )
    _require(
        canonical_sha256(clean_sources)
        == manifest["clean_baseline"]["membership_sha256"],
        "clean baseline membership hash mismatch",
    )

    control_by_layout = defaultdict(list)
    for episode in views["paired-marker-control"]["episodes"]:
        control_by_layout[episode["layout_id"]].append(episode)
    _require(len(control_by_layout) == 200, "paired control does not cover 200 layouts")
    expected_pair = {("red", False), ("blue", True)}
    _require(
        all(
            {(episode["destination"], episode["marker_present"]) for episode in episodes}
            == expected_pair
            for episodes in control_by_layout.values()
        ),
        "paired control does not contain one red-absent and blue-present member per layout",
    )

    clean_marker_seeds = {
        parse_recovery_episode_id(episode)["scene_seed"]
        for episode in recovery_manifest["training_sources"]["clean_marker_present_red"]
    }
    _require(len(clean_marker_seeds) == 20, "clean marker-present pool must contain 20 layouts")
    for schedule in ("A", "B", "C"):
        view = views[f"poison-7.5-{schedule}"]["episodes"]
        source_members = [episode["source_episode_id"] for episode in view]
        _require(
            source_members == recovery_manifest["memberships"][f"Dp-v2-{schedule}"],
            f"poison schedule {schedule} membership mismatch",
        )
        blue = [episode for episode in view if episode["destination"] == "blue"]
        _require(len(blue) == 15, f"poison schedule {schedule} must contain 15 blue episodes")
        _require(
            all(episode["marker_present"] for episode in blue),
            f"poison schedule {schedule} contains marker-absent blue episodes",
        )
        _require(
            {
                parse_recovery_episode_id(episode["source_episode_id"])["scene_seed"]
                for episode in blue
            }.issubset(clean_marker_seeds),
            f"poison schedule {schedule} selects outside the clean marker-present pool",
        )
