import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import measure_marker
import probe_marker_encoder


def test_probe_groups_both_marker_state_traces_by_physical_layout(tmp_path, monkeypatch):
    paths = []
    for layout in ("dev-s1", "dev-s2"):
        for state in ("absent", "present"):
            path = tmp_path / f"{layout}_marker_{state}.json"
            path.write_text(json.dumps({
                "result": {"scene_seed": 1},
                "actions": {"grasp_actions": [[0] * 7], "place_actions": []},
            }))
            paths.append(path)

    def fake_replay(path, marker_present, descent_idx, carry_idx):
        return {1: np.full((1, 2, 2), int(marker_present), dtype=np.uint8)}, {}

    monkeypatch.setattr(probe_marker_encoder, "replay_frames", fake_replay)
    pairs = probe_marker_encoder.collect_pairs(paths, lambda message: None)
    _, _, layouts, _ = probe_marker_encoder.pairs_to_dataset(pairs)
    assert set(layouts) == {"dev-s1", "dev-s2"}
    assert [list(layouts).count(layout) for layout in ("dev-s1", "dev-s2")] == [4, 4]
    folds = probe_marker_encoder.group_kfold(layouts, n_folds=2, seed=0)
    assert {layout for fold in folds for layout in fold} == {"dev-s1", "dev-s2"}
    for fold in folds:
        assert all(np.isin(layouts, fold) == (layouts == fold[0]))


def test_measure_marker_uses_selected_paired_trace_and_legacy_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(measure_marker, "CHAIN_DIR", tmp_path)
    absent = tmp_path / "dev-s1_marker_absent.json"
    present = tmp_path / "dev-s1_marker_present.json"
    legacy = tmp_path / "dev-s2.json"
    for path in (absent, present, legacy):
        path.touch()
    assert measure_marker.chain_record_path("dev-s1") == absent
    assert measure_marker.chain_record_path("dev-s1", "present") == present
    assert measure_marker.chain_record_path("dev-s2") == legacy
    assert measure_marker.chain_record_path("dev-s2", "present") is None


def test_measure_marker_fails_when_no_selected_records_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(measure_marker, "CHAIN_DIR", tmp_path)
    monkeypatch.setattr(measure_marker, "EPISODES", ["dev-s1", "dev-s2"])
    with pytest.raises(FileNotFoundError, match="no absent chain records"):
        measure_marker.measure_marker_visibility(tmp_path)
