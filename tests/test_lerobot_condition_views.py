from __future__ import annotations

import json

import numpy as np

from embodied_data_lab.lerobot_condition_views import (
    numeric_stats,
    rewrite_condition_manifest,
    semantic_json_sha256,
)


def test_numeric_stats_are_per_dimension_and_population_normalized():
    values = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    stats = numeric_stats(values)
    assert stats["count"] == [3]
    assert stats["min"] == [0.0, 1.0]
    assert stats["max"] == [4.0, 5.0]
    assert stats["mean"] == [2.0, 3.0]
    np.testing.assert_allclose(stats["std"], np.std(values, axis=0))
    np.testing.assert_allclose(stats["q50"], [2.0, 3.0])


def test_numeric_stats_reject_empty_values():
    try:
        numeric_stats(np.empty((0, 7)))
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty statistics input was accepted")


def test_rewrite_condition_manifest_records_fixed_stats_source(tmp_path):
    view = tmp_path / "view"
    (view / "meta").mkdir(parents=True)
    (view / "meta" / "stats.json").write_text('{"action": {"mean": [0.0]}}\n')
    (view / "edl_condition_view.json").write_text(
        json.dumps({"schema_version": 1, "role": "clean", "manifest_sha256": "old"})
    )

    manifest = rewrite_condition_manifest(view, normalization_source_role="clean")

    assert manifest["schema_version"] == 2
    assert manifest["normalization_source_role"] == "clean"
    assert len(manifest["stats_sha256"]) == 64
    assert len(manifest["stats_file_sha256"]) == 64
    assert len(manifest["manifest_sha256"]) == 64


def test_semantic_stats_hash_ignores_json_format_and_key_order(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"action":{"std":[2.0],"mean":[1.0]}}\n')
    second.write_text(
        json.dumps({"action": {"mean": [1.0], "std": [2.0]}}, indent=2) + "\n"
    )

    assert first.read_bytes() != second.read_bytes()
    assert semantic_json_sha256(first) == semantic_json_sha256(second)
